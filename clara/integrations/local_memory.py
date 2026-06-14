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

import logging
import uuid
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clara.agent import _make_engine, format_context
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.memory.belief import BeliefMemory, SourceType
from clara.memory.event import EventStore
from clara.memory.skill import SkillStore
from clara.memory.world_model import WorldModelStore
from clara.retrieval.engine import ScoredMemory
from clara.retrieval.lexical import LexicalRetriever

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in MemoryType}


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
                if tags:
                    meta = dict(record.metadata_) if record.metadata_ else {}
                    meta["tags"] = list(tags)
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
    ) -> dict[str, Any]:
        """Keyword-search memories. Returns a formatted context block + hits."""
        async with self._session_factory() as session:
            retriever = LexicalRetriever(session)
            result = await retriever.search(
                query,
                top_k=top_k,
                memory_types=_coerce_types(types),
                user_id=user_id,
            )
        return {
            "query": query,
            "total": result.total,
            "context": format_context(result),
            "hits": [_serialize(sm) for sm in result.all],
        }

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
        return {
            "db_path": self._db_path,
            "backend": "none (lexical + sqlite)",
            "active_by_type": by_type,
            "total_rows": int(total),
        }
