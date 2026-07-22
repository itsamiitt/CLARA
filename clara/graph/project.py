"""
CLARA graph — projection of memory writes into graph tables.

Every public ``project_*`` function is FAIL-SOFT by contract: any exception
is logged and swallowed, because a graph error must never fail a memory
write (the memories table is the source of truth; the graph is derived and
rebuildable via :func:`rebuild`).
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory
from clara.graph.normalize import normalize_relation
from clara.graph.resolve import _graph_ready, resolve_node

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


def _stamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return str(value.isoformat(sep=" "))
    return str(value) if value else _now()


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _fail_soft(
    fn: Callable[..., Awaitable[_T]],
) -> Callable[..., Awaitable[_T | None]]:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> _T | None:
        try:
            return await fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 — graph errors must never fail a memory write
            logger.warning("graph projection failed in %s", fn.__name__, exc_info=True)
            return None

    return wrapper


# ---------------------------------------------------------------------------
# Edge primitives
# ---------------------------------------------------------------------------


async def _insert_edge(
    session: AsyncSession,
    *,
    user_id: str | None,
    src_id: str,
    dst_id: str,
    relation: str,
    belief_id: str | None,
    confidence: float,
    valid_from: str,
    temporal_precision: str = "exact",
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> str:
    edge_id = uuid.uuid4().hex
    await session.execute(
        sa_text(
            "INSERT INTO graph_edges (edge_id, user_id, src_id, dst_id, relation, "
            "belief_id, confidence, weight, valid_from, temporal_precision, metadata) "
            "VALUES (:eid, :uid, :src, :dst, :rel, :bid, :conf, :weight, :vfrom, "
            ":precision, :meta)"
        ),
        {
            "eid": edge_id,
            "uid": user_id,
            "src": src_id,
            "dst": dst_id,
            "rel": relation,
            "bid": belief_id,
            "conf": confidence,
            "weight": weight,
            "vfrom": valid_from,
            "precision": temporal_precision,
            "meta": json.dumps(metadata or {}),
        },
    )
    return edge_id


async def _invalidate_matching(
    session: AsyncSession,
    *,
    src_id: str,
    dst_id: str,
    relation: str,
    invalidated_by: str,
    user_id: str | None,
) -> int:
    result = await session.execute(
        sa_text(
            "UPDATE graph_edges SET invalid_at = :now, "
            "metadata = json_set(coalesce(metadata, '{}'), '$.invalidated_by', :by) "
            "WHERE src_id = :src AND dst_id = :dst AND relation = :rel "
            "AND invalid_at IS NULL AND coalesce(user_id, '') = :uid"
        ),
        {
            "now": _now(),
            "by": invalidated_by,
            "src": src_id,
            "dst": dst_id,
            "rel": relation,
            "uid": user_id or "",
        },
    )
    return _rowcount(result)


async def _invalidate_by_belief(
    session: AsyncSession, belief_id: str, *, reason: str
) -> int:
    result = await session.execute(
        sa_text(
            "UPDATE graph_edges SET invalid_at = :now, "
            "metadata = json_set(coalesce(metadata, '{}'), '$.invalidated_by', :by) "
            "WHERE belief_id = :bid AND invalid_at IS NULL"
        ),
        {"now": _now(), "by": reason, "bid": belief_id},
    )
    return _rowcount(result)


# ---------------------------------------------------------------------------
# Projection entry points (all fail-soft)
# ---------------------------------------------------------------------------


async def _belief_endpoints(
    session: AsyncSession, record: Memory, *, create: bool
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    content = record.content if isinstance(record.content, dict) else {}
    relation = normalize_relation(content.get("relation", ""))
    src = await resolve_node(
        session, content.get("subject", ""), user_id=record.user_id, create=create
    )
    dst = await resolve_node(
        session, content.get("object", ""), user_id=record.user_id, create=create
    )
    return src, dst, relation


@_fail_soft
async def project_belief_created(session: AsyncSession, record: Memory) -> None:
    """New belief → edge; new negation belief → invalidate the matching edge."""
    if not await _graph_ready(session):
        return
    content = record.content if isinstance(record.content, dict) else {}
    if content.get("is_negation"):
        src, dst, relation = await _belief_endpoints(session, record, create=False)
        if src is not None and dst is not None and relation:
            await _invalidate_matching(
                session,
                src_id=src["node_id"],
                dst_id=dst["node_id"],
                relation=relation,
                invalidated_by=str(record.memory_id),
                user_id=record.user_id,
            )
        return
    src, dst, relation = await _belief_endpoints(session, record, create=True)
    if src is None or dst is None or not relation:
        return
    await _insert_edge(
        session,
        user_id=record.user_id,
        src_id=src["node_id"],
        dst_id=dst["node_id"],
        relation=relation,
        belief_id=str(record.memory_id),
        confidence=float(record.confidence),
        valid_from=_stamp(record.created_at),
    )


@_fail_soft
async def project_belief_superseded(
    session: AsyncSession, old_record: Memory, new_record: Memory | None = None
) -> None:
    """Old belief superseded → its edge gets ``invalid_at`` + provenance."""
    if not await _graph_ready(session):
        return
    reason = str(new_record.memory_id) if new_record is not None else "superseded"
    await _invalidate_by_belief(session, str(old_record.memory_id), reason=reason)


@_fail_soft
async def project_confidence_changed(
    session: AsyncSession, memory_id: str, confidence: float
) -> None:
    """Mirror a belief confidence change onto its valid edges."""
    if not await _graph_ready(session):
        return
    await session.execute(
        sa_text(
            "UPDATE graph_edges SET confidence = :conf "
            "WHERE belief_id = :bid AND invalid_at IS NULL"
        ),
        {"conf": float(confidence), "bid": str(memory_id)},
    )


@_fail_soft
async def project_memory_invalidated(
    session: AsyncSession, memory_id: str, *, reason: str = "forgotten"
) -> None:
    """memory_forget / deprecate → invalidate the projected edge(s)."""
    if not await _graph_ready(session):
        return
    await _invalidate_by_belief(session, str(memory_id), reason=reason)


@_fail_soft
async def project_world_model_upserted(session: AsyncSession, record: Memory) -> None:
    """World-model row → node upsert linking ``world_model_id``."""
    if not await _graph_ready(session):
        return
    content = record.content if isinstance(record.content, dict) else {}
    name = content.get("name") or content.get("subject")
    if not name:
        return
    node = await resolve_node(
        session,
        str(name),
        user_id=record.user_id,
        entity_type=str(content.get("entity_type") or "entity"),
        create=True,
    )
    if node is None:
        return
    existing_props: dict[str, Any] = {}
    with contextlib.suppress(ValueError):
        existing_props = json.loads(node.get("properties") or "{}")
    wm_props = content.get("properties")
    merged = {**existing_props, **wm_props} if isinstance(wm_props, dict) else existing_props
    await session.execute(
        sa_text(
            "UPDATE graph_nodes SET world_model_id = :wmid, properties = :props, "
            "entity_type = CASE WHEN entity_type = 'concept' THEN :etype "
            "ELSE entity_type END, "
            "updated_at = :now WHERE node_id = :nid"
        ),
        {
            "wmid": str(record.memory_id),
            "props": json.dumps(merged),
            "etype": str(content.get("entity_type") or "entity"),
            "now": _now(),
            "nid": node["node_id"],
        },
    )


# ---------------------------------------------------------------------------
# Rebuild / repair (loud, not fail-soft — this is an explicit CLI action)
# ---------------------------------------------------------------------------


async def rebuild(session: AsyncSession, *, from_scratch: bool = False) -> dict[str, int]:
    """Regenerate graph tables from active memories, in insertion order.

    Edges created here carry ``temporal_precision='reconstructed'``.
    Idempotent: beliefs whose edge already exists are skipped.
    """
    if not await _graph_ready(session):
        raise RuntimeError("graph tables missing — run migrations first")
    if from_scratch:
        for table in ("graph_edges", "graph_aliases", "graph_nodes"):
            await session.execute(sa_text(f"DELETE FROM {table}"))
        try:
            await session.execute(sa_text("DELETE FROM graph_nodes_fts"))
        except Exception:  # noqa: BLE001 — fts table may not exist
            session.info["_clara_graph_fts"] = None

    rows = (
        await session.execute(
            sa_text(
                "SELECT memory_id, user_id, memory_type, content, confidence, created_at "
                "FROM memories WHERE status = 'active' "
                "AND memory_type IN ('belief', 'world_model') ORDER BY created_at"
            )
        )
    ).all()

    existing_belief_ids = {
        row[0]
        for row in (
            await session.execute(sa_text("SELECT DISTINCT belief_id FROM graph_edges"))
        ).all()
        if row[0]
    }

    edges = 0
    for memory_id, user_id, memory_type, content_raw, confidence, created_at in rows:
        content = content_raw if isinstance(content_raw, dict) else json.loads(content_raw)
        mem_type = str(memory_type)
        if mem_type == "world_model":
            fake = _RowShim(memory_id, user_id, content, confidence, created_at)
            await project_world_model_upserted.__wrapped__(session, fake)  # type: ignore[attr-defined]
            continue
        belief_id = str(memory_id)
        if content.get("is_negation"):
            relation = normalize_relation(content.get("relation", ""))
            src = await resolve_node(
                session, content.get("subject", ""), user_id=user_id, create=False
            )
            dst = await resolve_node(
                session, content.get("object", ""), user_id=user_id, create=False
            )
            if src is not None and dst is not None and relation:
                await _invalidate_matching(
                    session,
                    src_id=src["node_id"],
                    dst_id=dst["node_id"],
                    relation=relation,
                    invalidated_by=belief_id,
                    user_id=user_id,
                )
            continue
        if belief_id in existing_belief_ids:
            continue
        relation = normalize_relation(content.get("relation", ""))
        src = await resolve_node(session, content.get("subject", ""), user_id=user_id, create=True)
        dst = await resolve_node(session, content.get("object", ""), user_id=user_id, create=True)
        if src is None or dst is None or not relation:
            continue
        await _insert_edge(
            session,
            user_id=user_id,
            src_id=src["node_id"],
            dst_id=dst["node_id"],
            relation=relation,
            belief_id=belief_id,
            confidence=float(confidence),
            valid_from=_stamp(created_at),
            temporal_precision="reconstructed",
        )
        edges += 1

    counts = {}
    for label, sql in (
        ("nodes", "SELECT COUNT(*) FROM graph_nodes WHERE status = 'active'"),
        ("edges", "SELECT COUNT(*) FROM graph_edges WHERE invalid_at IS NULL"),
    ):
        counts[label] = int((await session.execute(sa_text(sql))).scalar_one())
    counts["edges_created"] = edges
    return counts


class _RowShim:
    """Duck-typed stand-in for a Memory row when rebuilding from raw SQL."""

    def __init__(
        self, memory_id: Any, user_id: str | None, content: dict[str, Any],
        confidence: float, created_at: Any,
    ) -> None:
        self.memory_id = memory_id
        self.user_id = user_id
        self.content = content
        self.confidence = confidence
        self.created_at = created_at
