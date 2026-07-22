"""
CLARA — LocalMemory: zero-backend memory facade

The engine behind the Claude Code / MCP integration. Unlike :class:`ClaraMemory`
it uses **no LLM, no embedding model, and no LanceDB** — the calling agent
(Claude) supplies already-structured memories and plain-text queries.

* Writes go straight to SQLite through the existing typed memory stores
  (``BeliefMemory``, ``EventStore``, ``SkillStore``, ``WorldModelStore``) with
  ``embedding=None``.
* Reads use :class:`LexicalRetriever` (keyword search over SQLite).

The only state on disk is a single SQLite file. LanceDB stays dormant via the
``_clara_disable_lance`` session flag (see ``clara/retrieval/engine.py``).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import func, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clara.agent import _make_engine, format_context
from clara.db.fts import ensure_fts
from clara.db.migrations import ensure_schema
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.graph import project as graph_project
from clara.graph import render as graph_render
from clara.graph import traverse as graph_traverse
from clara.graph.resolve import resolve_node
from clara.memory.belief import BeliefMemory, SourceType
from clara.memory.event import EventStore
from clara.memory.skill import SkillStore
from clara.memory.world_model import WorldModelStore
from clara.repoid import repo_id
from clara.retrieval.engine import ScoredMemory
from clara.retrieval.lexical import LexicalRetriever

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in MemoryType}

# repo_id shells out to git; cache per working directory so long-lived MCP
# servers pay the subprocess cost once, not per save.
_REPO_ID_CACHE: dict[str, str] = {}


def _cached_repo_id() -> str:
    cwd = str(Path.cwd())
    cached = _REPO_ID_CACHE.get(cwd)
    if cached is None:
        cached = repo_id(cwd)
        _REPO_ID_CACHE[cwd] = cached
    return cached


def _ensure_versioned_schema(db_path: str) -> None:
    """Apply versioned migrations (graph tables etc.) to a file-backed store.

    Fail-soft: the memory store must work even if migrations cannot run
    (read-only mount, schema from a newer CLARA) — graph features then
    degrade to no-ops.
    """
    if db_path in ("", ":memory:"):
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            ensure_schema(conn)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — memory availability beats graph availability
        logger.warning("versioned schema migration failed for %s", db_path, exc_info=True)


def _coerce_types(types: Sequence[str] | None) -> list[MemoryType] | None:
    """Map a list of type strings to :class:`MemoryType`, ignoring unknowns."""
    if not types:
        return None
    coerced: list[MemoryType] = []
    for raw in types:
        try:
            coerced.append(MemoryType(raw))
        except ValueError:
            logger.debug("Ignoring unknown memory type %r", raw)
    return coerced or None


def _source_type(source: str) -> SourceType:
    try:
        return SourceType(source)
    except ValueError:
        return SourceType.user_direct


def _serialize(sm: ScoredMemory) -> dict[str, Any]:
    return {
        "memory_id": str(sm.memory.memory_id),
        "type": sm.memory.memory_type.value,
        "score": round(sm.score, 4),
        "confidence": round(sm.confidence, 4),
        "content": sm.memory.content,
    }


class LocalMemory:
    """SQLite-only memory store driven entirely by the calling agent."""

    def __init__(
        self,
        *,
        engine,
        session_factory: async_sessionmaker[AsyncSession],
        db_path: str,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, db_path: str) -> LocalMemory:
        """Open (or create) the SQLite store at *db_path* and build the schema."""
        path_obj = Path(db_path)
        # Forward-slash form keeps the SQLAlchemy URL valid on Windows paths.
        db_url = f"sqlite+aiosqlite:///{path_obj.as_posix()}"
        db_path = str(path_obj)
        engine = _make_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await ensure_fts(engine)
        await asyncio.to_thread(_ensure_versioned_schema, db_path)

        # The info dict is shared by every session this factory makes, so the
        # LanceDB commit listeners short-circuit and never touch a vector store.
        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
            info={"_clara_disable_lance": True},
        )
        logger.info("LocalMemory ready (db=%s, backend=none)", db_path)
        return cls(engine=engine, session_factory=session_factory, db_path=db_path)

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("LocalMemory closed")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(
        self,
        *,
        mem_type: str = "belief",
        # belief
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        is_negation: bool = False,
        # event
        event_type: str | None = None,
        # skill
        name: str | None = None,
        trigger_conditions: list[str] | None = None,
        steps: list[str] | None = None,
        # world_model
        entity_type: str | None = None,
        properties: dict[str, Any] | None = None,
        # shared
        description: str | None = None,
        domain: str | None = None,
        confidence: float | None = None,
        source: str = "user_direct",
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one structured memory and return ``{memory_id, type, action}``.

        Raises ``ValueError`` if *mem_type* is unknown or required fields for the
        chosen type are missing.
        """
        if mem_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown mem_type {mem_type!r}. Expected one of {sorted(VALID_TYPES)}."
            )

        async with self._session_factory() as session:
            async with session.begin():
                record = await self._route_save(
                    session,
                    mem_type=mem_type,
                    subject=subject,
                    relation=relation,
                    object=object,
                    is_negation=is_negation,
                    event_type=event_type,
                    name=name,
                    trigger_conditions=trigger_conditions,
                    steps=steps,
                    entity_type=entity_type,
                    properties=properties,
                    description=description,
                    domain=domain,
                    confidence=confidence,
                    source=source,
                    user_id=user_id,
                )
                if confidence is not None:
                    record.confidence = max(0.0, min(1.0, float(confidence)))
                meta = dict(record.metadata_) if record.metadata_ else {}
                if tags:
                    meta["tags"] = list(tags)
                # Stamp provenance on new writes; world-model upserts of an
                # existing record keep their original repo_id.
                meta.setdefault("repo_id", _cached_repo_id())
                record.metadata_ = meta
                memory_id = str(record.memory_id)
                rec_type = record.memory_type.value

        logger.info("Saved %s memory %s", rec_type, memory_id)
        return {"memory_id": memory_id, "type": rec_type, "action": "saved"}

    @staticmethod
    async def _route_save(
        session: AsyncSession,
        *,
        mem_type: str,
        subject: str | None,
        relation: str | None,
        object: str | None,
        is_negation: bool,
        event_type: str | None,
        name: str | None,
        trigger_conditions: list[str] | None,
        steps: list[str] | None,
        entity_type: str | None,
        properties: dict[str, Any] | None,
        description: str | None,
        domain: str | None,
        confidence: float | None,
        source: str,
        user_id: str | None,
    ) -> Memory:
        if mem_type == MemoryType.belief.value:
            if not (subject and relation and object):
                raise ValueError(
                    "belief requires 'subject', 'relation', and 'object'."
                )
            return await BeliefMemory(session).store(
                subject=subject,
                relation=relation,
                object_=object,
                domain=domain,
                is_negation=is_negation,
                source=_source_type(source),
                raw_text=description,
                user_id=user_id,
            )

        if mem_type == MemoryType.event.value:
            if not (subject and event_type):
                raise ValueError("event requires 'subject' and 'event_type'.")
            return await EventStore(session).create(
                subject=subject,
                event_type=event_type,
                description=description or "",
                domain=domain,
                user_id=user_id,
                confidence=confidence if confidence is not None else 1.0,
                source_type=source,
                raw_text=description,
            )

        if mem_type == MemoryType.skill.value:
            if not name:
                raise ValueError("skill requires 'name'.")
            return await SkillStore(session).create(
                name=name,
                trigger_conditions=trigger_conditions,
                steps=steps,
                description=description or "",
                domain=domain,
                user_id=user_id,
                confidence=confidence if confidence is not None else 0.5,
                source_type=source,
                raw_text=description,
            )

        # world_model
        if not (entity_type and name):
            raise ValueError("world_model requires 'entity_type' and 'name'.")
        return await WorldModelStore(session).upsert(
            entity_type=entity_type,
            name=name,
            properties=properties,
            domain=domain,
            user_id=user_id,
            confidence=confidence if confidence is not None else 0.9,
            source_type=source,
            raw_text=description,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        types: Sequence[str] | None = None,
        user_id: str | None = None,
        graph_depth: int = 0,
    ) -> dict[str, Any]:
        """Keyword-search memories. Returns a formatted context block + hits.

        ``graph_depth > 0`` appends a ``[GRAPH]`` section built by traversing
        the knowledge graph from the entities of the top hits (fail-soft).
        """
        async with self._session_factory() as session:
            retriever = LexicalRetriever(session)
            result = await retriever.search(
                query,
                top_k=top_k,
                memory_types=_coerce_types(types),
                user_id=user_id,
            )
        payload: dict[str, Any] = {
            "query": query,
            "total": result.total,
            "context": format_context(result),
            "hits": [_serialize(sm) for sm in result.all],
        }
        if graph_depth > 0 and result.total:
            try:
                graph = await self._graph_augment(result, depth=graph_depth, user_id=user_id)
            except Exception:  # noqa: BLE001 — graph must never break search
                logger.warning("graph augmentation failed", exc_info=True)
                graph = None
            if graph:
                payload["context"] += "\n\n" + graph["section"]
                payload["graph"] = {"edges": graph["edges"]}
        return payload

    async def _graph_augment(
        self, result: Any, *, depth: int, user_id: str | None
    ) -> dict[str, Any] | None:
        """Traverse from the top hits' entities and render a [GRAPH] section."""
        seed_names: list[str] = []
        for sm in result.all[:5]:
            content = sm.memory.content if isinstance(sm.memory.content, dict) else {}
            for key in ("subject", "object", "name"):
                value = content.get(key)
                if isinstance(value, str) and value and value not in seed_names:
                    seed_names.append(value)
        seed_names = [n for n in seed_names if n.lower() != "user"][:3]
        if not seed_names:
            return None
        async with self._session_factory() as session:
            async with session.begin():
                groups: list[tuple[str, list[dict[str, Any]]]] = []
                all_edges: list[dict[str, Any]] = []
                node_ids: list[str] = []
                for name in seed_names:
                    node = await resolve_node(
                        session, name, user_id=user_id, create=False, bump_mention=False
                    )
                    if node is None:
                        continue
                    edges = await graph_traverse.traverse(
                        session,
                        [node["node_id"]],
                        depth=depth,
                        user_id=user_id,
                        limit=12,
                    )
                    if not edges:
                        continue
                    groups.append((node["display_name"], edges))
                    all_edges.extend(edges)
                    node_ids.extend(
                        [e["src_id"] for e in edges] + [e["dst_id"] for e in edges]
                    )
                if not all_edges:
                    return None
                nodes = await graph_traverse.fetch_nodes(session, node_ids)
                await graph_traverse.bump_traversed(session, all_edges)
                section = graph_render.render_graph_section(groups, nodes)
        if not section:
            return None
        edges_payload = [
            {
                "edge_id": e["edge_id"],
                "src": nodes.get(e["src_id"], {}).get("display_name", e["src_id"]),
                "relation": e["relation"],
                "dst": nodes.get(e["dst_id"], {}).get("display_name", e["dst_id"]),
                "confidence": e["confidence"],
                "depth": e["depth"],
                "score": round(float(e["score"]), 4),
            }
            for e in all_edges
        ]
        return {"section": section, "edges": edges_payload}

    async def recent(
        self,
        *,
        n: int = 10,
        types: Sequence[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Most relevant recent memories (recency + confidence ranked)."""
        return await self.search("", top_k=n, types=types, user_id=user_id)

    # ------------------------------------------------------------------
    # Mutate / lifecycle
    # ------------------------------------------------------------------

    async def update(
        self,
        memory_id: str,
        *,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update confidence and/or tags on an existing memory."""
        mid = uuid.UUID(str(memory_id))
        async with self._session_factory() as session:
            async with session.begin():
                record = await session.get(Memory, mid)
                if record is None:
                    raise ValueError(f"Memory {memory_id} not found.")
                if confidence is not None:
                    record.confidence = max(0.0, min(1.0, float(confidence)))
                    await graph_project.project_confidence_changed(
                        session, str(mid), record.confidence
                    )
                if tags is not None:
                    meta = dict(record.metadata_) if record.metadata_ else {}
                    meta["tags"] = list(tags)
                    record.metadata_ = meta
                rec_type = record.memory_type.value
        return {"memory_id": str(mid), "type": rec_type, "action": "updated"}

    async def forget(self, memory_id: str, *, archive: bool = False) -> dict[str, Any]:
        """Deprecate (default) or archive a memory. Never hard-deletes."""
        mid = uuid.UUID(str(memory_id))
        new_status = MemoryStatus.archived if archive else MemoryStatus.deprecated
        async with self._session_factory() as session:
            async with session.begin():
                record = await session.get(Memory, mid)
                if record is None:
                    raise ValueError(f"Memory {memory_id} not found.")
                record.status = new_status
                await graph_project.project_memory_invalidated(
                    session, str(mid), reason=new_status.value
                )
        return {"memory_id": str(mid), "action": new_status.value}

    async def stats(self) -> dict[str, Any]:
        """Counts of active memories per type, plus total rows."""
        async with self._session_factory() as session:
            by_type_stmt = (
                select(Memory.memory_type, func.count())
                .where(Memory.status == MemoryStatus.active)
                .group_by(Memory.memory_type)
            )
            by_type = {
                mtype.value: count
                for mtype, count in (await session.execute(by_type_stmt)).all()
            }
            total = (
                await session.execute(select(func.count()).select_from(Memory))
            ).scalar_one()
            graph_counts: dict[str, int] = {}
            try:
                for label, sql in (
                    ("nodes", "SELECT COUNT(*) FROM graph_nodes WHERE status = 'active'"),
                    ("edges", "SELECT COUNT(*) FROM graph_edges WHERE invalid_at IS NULL"),
                ):
                    graph_counts[label] = int(
                        (await session.execute(sa_text(sql))).scalar_one()
                    )
            except Exception:  # noqa: BLE001 — graph tables may be absent
                graph_counts = {}
        payload = {
            "db_path": self._db_path,
            "backend": "none (lexical + sqlite)",
            "active_by_type": by_type,
            "total_rows": int(total),
        }
        if graph_counts:
            payload["graph"] = graph_counts
        return payload

    # ------------------------------------------------------------------
    # Knowledge graph API
    # ------------------------------------------------------------------

    async def graph_entity(
        self, name: str, *, include_history: bool = False, user_id: str | None = None
    ) -> dict[str, Any]:
        """Entity card: node fields, aliases, possible_duplicates, top edges."""
        import json as _json

        async with self._session_factory() as session:
            async with session.begin():
                node = await resolve_node(
                    session, name, user_id=user_id, create=False, bump_mention=False
                )
                if node is None:
                    return {"found": False, "name": name}
                aliases = (
                    await session.execute(
                        sa_text("SELECT alias_norm FROM graph_aliases WHERE node_id = :nid"),
                        {"nid": node["node_id"]},
                    )
                ).scalars().all()
                edges = await graph_traverse.traverse(
                    session, [node["node_id"]], depth=1, limit=10, user_id=user_id
                )
                history: list[dict[str, Any]] = []
                if include_history:
                    rows = (
                        await session.execute(
                            sa_text(
                                "SELECT * FROM graph_edges WHERE invalid_at IS NOT NULL "
                                "AND (src_id = :nid OR dst_id = :nid) "
                                "ORDER BY invalid_at DESC LIMIT 10"
                            ),
                            {"nid": node["node_id"]},
                        )
                    ).mappings().all()
                    history = [dict(r) for r in rows]
                node_ids = [node["node_id"]]
                for e in [*edges, *history]:
                    node_ids.extend([e["src_id"], e["dst_id"]])
                nodes = await graph_traverse.fetch_nodes(session, node_ids)
        try:
            properties = _json.loads(node.get("properties") or "{}")
        except ValueError:
            properties = {}
        card_lines = [
            graph_render.format_edge_line(e, nodes) for e in [*edges, *history]
        ]
        return {
            "found": True,
            "node_id": node["node_id"],
            "name": node["display_name"],
            "canonical_name": node["canonical_name"],
            "entity_type": node["entity_type"],
            "mention_count": node["mention_count"],
            "expandable": bool(node["expandable"]),
            "world_model_id": node.get("world_model_id"),
            "aliases": list(aliases),
            "possible_duplicates": properties.get("possible_duplicates", []),
            "properties": properties,
            "edges": card_lines,
        }

    async def graph_neighbors(
        self,
        name: str,
        *,
        depth: int = 1,
        relation: str | None = None,
        as_of: str | None = None,
        limit: int = 20,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Neighborhood traversal from a named entity, rendered + structured."""
        async with self._session_factory() as session:
            async with session.begin():
                node = await resolve_node(
                    session, name, user_id=user_id, create=False, bump_mention=False
                )
                if node is None:
                    return {"found": False, "name": name, "edges": []}
                edges = await graph_traverse.traverse(
                    session,
                    [node["node_id"]],
                    depth=depth,
                    relation=relation,
                    as_of=as_of,
                    limit=limit,
                    user_id=user_id,
                )
                node_ids = [node["node_id"]]
                for e in edges:
                    node_ids.extend([e["src_id"], e["dst_id"]])
                nodes = await graph_traverse.fetch_nodes(session, node_ids)
                if edges and as_of is None:
                    await graph_traverse.bump_traversed(session, edges)
        section = graph_render.render_graph_section(
            [(node["display_name"], edges)], nodes
        )
        return {
            "found": True,
            "name": node["display_name"],
            "context": section,
            "edges": [
                {
                    "edge_id": e["edge_id"],
                    "line": graph_render.format_edge_line(e, nodes),
                    "depth": e["depth"],
                    "score": round(float(e["score"]), 4),
                }
                for e in edges
            ],
        }

    async def graph_path(
        self,
        from_name: str,
        to_name: str,
        *,
        max_hops: int = 4,
        as_of: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Best path between two named entities, or found=False."""
        async with self._session_factory() as session:
            async with session.begin():
                src = await resolve_node(
                    session, from_name, user_id=user_id, create=False, bump_mention=False
                )
                dst = await resolve_node(
                    session, to_name, user_id=user_id, create=False, bump_mention=False
                )
                if src is None or dst is None:
                    return {
                        "found": False,
                        "missing": [
                            n for n, r in ((from_name, src), (to_name, dst)) if r is None
                        ],
                    }
                path = await graph_traverse.find_path(
                    session,
                    src["node_id"],
                    dst["node_id"],
                    max_hops=max_hops,
                    as_of=as_of,
                    user_id=user_id,
                )
                if path is None:
                    return {"found": False, "hops": None}
                node_ids = [src["node_id"], dst["node_id"]]
                for e in path:
                    node_ids.extend([e["src_id"], e["dst_id"]])
                nodes = await graph_traverse.fetch_nodes(session, node_ids)
        return {
            "found": True,
            "hops": len(path),
            "path": [graph_render.format_edge_line(e, nodes) for e in path],
        }

    async def memory_link(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        entity_types: list[str] | None = None,
        confidence: float | None = None,
        description: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Belief-save sugar: pre-resolve typed endpoints, save, return ids."""
        src_type = entity_types[0] if entity_types and len(entity_types) > 0 else None
        dst_type = entity_types[1] if entity_types and len(entity_types) > 1 else None
        async with self._session_factory() as session:
            async with session.begin():
                src_node = await resolve_node(
                    session, src, user_id=user_id, entity_type=src_type, create=True
                )
                dst_node = await resolve_node(
                    session, dst, user_id=user_id, entity_type=dst_type, create=True
                )
        saved = await self.save(
            mem_type="belief",
            subject=src,
            relation=relation,
            object=dst,
            confidence=confidence,
            description=description,
            user_id=user_id,
        )
        async with self._session_factory() as session:
            edge_row = (
                await session.execute(
                    sa_text(
                        "SELECT edge_id FROM graph_edges WHERE belief_id = :bid "
                        "ORDER BY rowid DESC LIMIT 1"
                    ),
                    {"bid": saved["memory_id"]},
                )
            ).first()
        return {
            "belief_id": saved["memory_id"],
            "edge_id": edge_row[0] if edge_row else None,
            "src_node": src_node["display_name"] if src_node else src,
            "dst_node": dst_node["display_name"] if dst_node else dst,
            "action": "linked",
        }

    async def graph_rebuild(self, *, from_scratch: bool = False) -> dict[str, int]:
        """Regenerate graph tables from active memories (CLI entry point)."""
        async with self._session_factory() as session:
            async with session.begin():
                return await graph_project.rebuild(session, from_scratch=from_scratch)
