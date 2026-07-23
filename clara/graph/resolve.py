"""
CLARA graph — entity resolution.

Resolution order (spec'd in the graph design):
normalize → exact canonical match → alias table → seed + clara.yml aliases →
trigram-FTS fuzzy candidates accepted iff ``name_sim >= 0.90``. Multiple
fuzzy candidates create a NEW node that records the candidate ids in
``properties.possible_duplicates`` (merging is a human/agent decision, never
a silent guess). ``user`` and ``skill`` resolve to singleton nodes created
with ``expandable = 0``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from clara.graph.normalize import SEED_ALIASES, name_sim, normalize_name

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.90
_SINGLETONS = {"user", "skill"}
_FUZZY_SCAN_LIMIT = 500

_alias_map_cache: dict[str, str] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


def alias_map() -> dict[str, str]:
    """Seed aliases merged with the repo policy's ``aliases:`` (policy wins)."""
    global _alias_map_cache
    if _alias_map_cache is None:
        merged = dict(SEED_ALIASES)
        try:
            from clara.policy import load_policy

            for alias, target in load_policy().aliases.items():
                merged[normalize_name(alias)] = normalize_name(target)
        except Exception:  # noqa: BLE001 — a broken clara.yml must not break resolution
            logger.warning("could not load clara.yml aliases; using seed aliases only")
        _alias_map_cache = merged
    return _alias_map_cache


async def _graph_ready(session: AsyncSession) -> bool:
    """Probe (cached per session) whether the graph tables exist."""
    cached = session.info.get("_clara_graph_ready")
    if cached is not None:
        return bool(cached)
    row = (
        await session.execute(
            sa_text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_nodes'")
        )
    ).first()
    session.info["_clara_graph_ready"] = row is not None
    return row is not None


async def _has_fts(session: AsyncSession) -> bool:
    cached = session.info.get("_clara_graph_fts")
    if cached is not None:
        return bool(cached)
    row = (
        await session.execute(
            sa_text(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_nodes_fts'"
            )
        )
    ).first()
    session.info["_clara_graph_fts"] = row is not None
    return row is not None


async def _fts_upsert(session: AsyncSession, node_id: str, names: str, entity_type: str) -> None:
    if not await _has_fts(session):
        return
    await session.execute(
        sa_text("DELETE FROM graph_nodes_fts WHERE node_id = :nid"), {"nid": node_id}
    )
    await session.execute(
        sa_text(
            "INSERT INTO graph_nodes_fts (node_id, names, entity_type) "
            "VALUES (:nid, :names, :etype)"
        ),
        {"nid": node_id, "names": names, "etype": entity_type},
    )


async def _node_by_id(session: AsyncSession, node_id: str) -> dict[str, Any] | None:
    row = (
        await session.execute(
            sa_text("SELECT * FROM graph_nodes WHERE node_id = :nid"), {"nid": node_id}
        )
    ).mappings().first()
    if row is None:
        return None
    node = dict(row)
    if node.get("status") == "merged" and node.get("merged_into"):
        target = (
            await session.execute(
                sa_text("SELECT * FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node["merged_into"]},
            )
        ).mappings().first()
        if target is not None:
            return dict(target)
    return node


async def _bump_mention(session: AsyncSession, node_id: str) -> None:
    await session.execute(
        sa_text(
            "UPDATE graph_nodes SET mention_count = mention_count + 1, updated_at = :now "
            "WHERE node_id = :nid"
        ),
        {"nid": node_id, "now": _now()},
    )


async def add_alias(
    session: AsyncSession,
    node_id: str,
    alias: str,
    *,
    user_id: str | None = None,
    source: str = "auto",
) -> None:
    """Record an alias for a node and refresh its FTS names row."""
    alias_norm = normalize_name(alias)
    if not alias_norm:
        return
    await session.execute(
        sa_text(
            "INSERT OR IGNORE INTO graph_aliases (alias_norm, node_id, user_id, source) "
            "VALUES (:alias, :nid, :uid, :src)"
        ),
        {"alias": alias_norm, "nid": node_id, "uid": user_id or "", "src": source},
    )
    node = await _node_by_id(session, node_id)
    if node is None:
        return
    aliases = (
        await session.execute(
            sa_text("SELECT alias_norm FROM graph_aliases WHERE node_id = :nid"),
            {"nid": node_id},
        )
    ).scalars().all()
    names = " ".join(
        dict.fromkeys([node["canonical_name"], normalize_name(node["display_name"]), *aliases])
    )
    await _fts_upsert(session, node_id, names, node["entity_type"])


async def _exact_canonical(
    session: AsyncSession,
    canonical: str,
    *,
    user_id: str | None,
    entity_type: str | None,
) -> dict[str, Any] | None:
    sql = (
        "SELECT * FROM graph_nodes WHERE status = 'active' "
        "AND canonical_name = :canon AND coalesce(user_id, '') = :uid"
    )
    params: dict[str, Any] = {"canon": canonical, "uid": user_id or ""}
    if entity_type is not None:
        sql += " AND entity_type = :etype"
        params["etype"] = entity_type
    sql += " ORDER BY mention_count DESC"
    row = (await session.execute(sa_text(sql), params)).mappings().first()
    return dict(row) if row else None


async def _alias_lookup(
    session: AsyncSession, alias_norm: str, *, user_id: str | None
) -> dict[str, Any] | None:
    row = (
        await session.execute(
            sa_text(
                "SELECT node_id FROM graph_aliases WHERE alias_norm = :alias "
                "AND user_id = :uid"
            ),
            {"alias": alias_norm, "uid": user_id or ""},
        )
    ).first()
    if row is None:
        return None
    node = await _node_by_id(session, row[0])
    if node is not None and node.get("status") == "active":
        return node
    return None


async def _fuzzy_candidates(
    session: AsyncSession, norm: str, *, user_id: str | None
) -> list[dict[str, Any]]:
    rows: Sequence[Any] = []
    if len(norm) >= 3 and await _has_fts(session):
        try:
            # A phrase query would demand a contiguous substring; OR-ing a
            # sample of the query's trigrams gives fuzzy recall instead
            # (precision comes from the name_sim >= 0.90 filter below).
            grams = [norm[i : i + 3] for i in range(len(norm) - 2)]
            if len(grams) > 8:
                step = len(grams) / 8
                grams = [grams[int(i * step)] for i in range(8)]
            match = " OR ".join('"' + g.replace('"', '""') + '"' for g in grams)
            fts_rows = (
                await session.execute(
                    sa_text(
                        "SELECT node_id FROM graph_nodes_fts "
                        "WHERE graph_nodes_fts MATCH :q LIMIT 40"
                    ),
                    {"q": match},
                )
            ).scalars().all()
            if fts_rows:
                placeholders = ", ".join(f":n{i}" for i in range(len(fts_rows)))
                rows = (
                    await session.execute(
                        sa_text(
                            f"SELECT * FROM graph_nodes WHERE node_id IN ({placeholders}) "
                            "AND status = 'active' AND coalesce(user_id, '') = :uid"
                        ),
                        {**{f"n{i}": nid for i, nid in enumerate(fts_rows)}, "uid": user_id or ""},
                    )
                ).mappings().all()
        except Exception:  # noqa: BLE001 — FTS problems degrade to the scan below
            logger.debug("graph FTS lookup failed; falling back to scan", exc_info=True)
            rows = []
    if not rows:
        rows = (
            await session.execute(
                sa_text(
                    "SELECT * FROM graph_nodes WHERE status = 'active' "
                    "AND coalesce(user_id, '') = :uid LIMIT :lim"
                ),
                {"uid": user_id or "", "lim": _FUZZY_SCAN_LIMIT},
            )
        ).mappings().all()
    scored = [
        (name_sim(norm, r["canonical_name"]), dict(r))
        for r in rows
        if name_sim(norm, r["canonical_name"]) >= SIM_THRESHOLD
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in scored]


async def create_node(
    session: AsyncSession,
    *,
    canonical: str,
    display: str,
    entity_type: str = "concept",
    user_id: str | None = None,
    expandable: int = 1,
    world_model_id: str | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a graph node (+ its FTS names row) and return it."""
    node_id = uuid.uuid4().hex
    now = _now()
    await session.execute(
        sa_text(
            "INSERT INTO graph_nodes (node_id, user_id, canonical_name, display_name, "
            "entity_type, world_model_id, properties, mention_count, expandable, status, "
            "created_at, updated_at) VALUES (:nid, :uid, :canon, :disp, :etype, :wmid, "
            ":props, 1, :expandable, 'active', :now, :now)"
        ),
        {
            "nid": node_id,
            "uid": user_id,
            "canon": canonical,
            "disp": display,
            "etype": entity_type,
            "wmid": world_model_id,
            "props": json.dumps(properties or {}),
            "expandable": expandable,
            "now": now,
        },
    )
    names = " ".join(dict.fromkeys([canonical, normalize_name(display)]))
    await _fts_upsert(session, node_id, names, entity_type)
    node = await _node_by_id(session, node_id)
    assert node is not None
    node["_created"] = True
    return node


async def resolve_node(
    session: AsyncSession,
    name: str,
    *,
    user_id: str | None = None,
    entity_type: str | None = None,
    create: bool = True,
    bump_mention: bool = True,
) -> dict[str, Any] | None:
    """Resolve *name* to a graph node, optionally creating one.

    Returns the node row as a dict (``_created: True`` when new), or ``None``
    when the graph tables are absent, the name is empty, or ``create=False``
    and nothing matched.
    """
    if not await _graph_ready(session):
        return None
    norm = normalize_name(name)
    if not norm:
        return None
    canonical = alias_map().get(norm, norm)

    if canonical in _SINGLETONS:
        node = await _exact_canonical(
            session, canonical, user_id=user_id, entity_type=canonical
        )
        if node is None and create:
            return await create_node(
                session,
                canonical=canonical,
                display=canonical,
                entity_type=canonical,
                user_id=user_id,
                expandable=0,
            )
        if node is not None and bump_mention:
            await _bump_mention(session, node["node_id"])
        return node

    node = await _exact_canonical(session, canonical, user_id=user_id, entity_type=entity_type)
    if node is None and entity_type is not None:
        # A node of another entity_type with the same canonical name still
        # beats creating a duplicate concept node.
        node = await _exact_canonical(session, canonical, user_id=user_id, entity_type=None)
    if node is not None:
        if bump_mention:
            await _bump_mention(session, node["node_id"])
        return node

    for candidate_alias in dict.fromkeys([norm, canonical]):
        node = await _alias_lookup(session, candidate_alias, user_id=user_id)
        if node is not None:
            if bump_mention:
                await _bump_mention(session, node["node_id"])
            return node

    candidates = await _fuzzy_candidates(session, canonical, user_id=user_id)
    if len(candidates) == 1:
        node = candidates[0]
        await add_alias(session, node["node_id"], canonical, user_id=user_id, source="auto")
        if bump_mention:
            await _bump_mention(session, node["node_id"])
        return node
    if len(candidates) > 1 and create:
        return await create_node(
            session,
            canonical=canonical,
            display=name.strip() or canonical,
            entity_type=entity_type or "concept",
            user_id=user_id,
            properties={
                "possible_duplicates": [c["node_id"] for c in candidates],
            },
        )
    if not create:
        return candidates[0] if candidates else None

    return await create_node(
        session,
        canonical=canonical,
        display=name.strip() or canonical,
        entity_type=entity_type or "concept",
        user_id=user_id,
    )
