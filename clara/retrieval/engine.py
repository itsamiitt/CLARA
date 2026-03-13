"""
CLARA - Vector Retrieval Engine

Retrieves relevant memories from the Unified Memory Store given a natural
language query. Vector search is handled by embedded LanceDB, while SQLite
remains the source of truth for relational metadata and lifecycle fields.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import uuid
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Sequence

import lancedb
import pyarrow as pa
from sqlalchemy import and_, event, inspect as sa_inspect, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from clara.db.models import Memory, MemoryStatus, MemoryType, VECTOR_DIMENSIONS
from clara.retrieval.cache import CachedScoredMemory, MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine, normalize_embedding_dimensions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring weights (from CONTEXT.md §Vector Retrieval Engine)
# ---------------------------------------------------------------------------

W_SIMILARITY: float = 0.65
W_CONFIDENCE: float = 0.20
W_RECENCY: float = 0.10
W_USAGE: float = 0.05

RECENCY_LAMBDA: float = 0.01  # decay constant for recency_score

# How many candidates to pull from the vector index before re-ranking
DEFAULT_CANDIDATE_MULTIPLIER: int = 4  # top_k × this = ANN candidates

DEFAULT_LANCE_PATH = "./clara_vectors"
LANCE_TABLE_NAME = "memories"
LANCE_SCHEMA = pa.schema([
    pa.field("memory_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIMENSIONS)),
    pa.field("user_id", pa.string()),
    pa.field("memory_type", pa.string()),
    pa.field("status", pa.string()),
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A :class:`Memory` record together with its composite retrieval score."""

    memory: Memory
    score: float
    similarity: float
    confidence: float
    recency_score: float
    usage_frequency: float


@dataclass(slots=True)
class RetrievalResult:
    """Top-k results grouped by memory type."""

    beliefs: list[ScoredMemory] = field(default_factory=list)
    events: list[ScoredMemory] = field(default_factory=list)
    skills: list[ScoredMemory] = field(default_factory=list)
    world_model: list[ScoredMemory] = field(default_factory=list)

    @property
    def all(self) -> list[ScoredMemory]:
        """All scored memories in a single flat list, highest score first."""
        combined = self.beliefs + self.events + self.skills + self.world_model
        return sorted(combined, key=lambda s: s.score, reverse=True)

    @property
    def total(self) -> int:
        return len(self.beliefs) + len(self.events) + len(self.skills) + len(self.world_model)


@dataclass(frozen=True, slots=True)
class LanceMemoryRecord:
    """Minimal record persisted in the embedded LanceDB vector table."""

    memory_id: str
    vector: list[float] | None
    user_id: str
    memory_type: str
    status: str
    is_new: bool = False


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_recency_score(updated_at: datetime, now: datetime | None = None) -> float:
    """``recency_score = e^(−λ × days_since_last_accessed)``"""
    now = now or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - updated_at).total_seconds() / 86_400.0)
    return math.exp(-RECENCY_LAMBDA * delta_days)


def compute_usage_frequency(access_count: int, max_access_count: int) -> float:
    """``usage_frequency = log(1 + access_count) / log(1 + max_access_count)``"""
    if max_access_count <= 0:
        return 0.0
    denominator = math.log(1 + max_access_count)
    if denominator == 0.0:
        return 0.0
    return math.log(1 + access_count) / denominator


def compute_final_score(
    similarity: float,
    confidence: float,
    recency_score: float,
    usage_frequency: float,
) -> float:
    """Composite retrieval score from CONTEXT.md."""
    return (
        W_SIMILARITY * similarity
        + W_CONFIDENCE * confidence
        + W_RECENCY * recency_score
        + W_USAGE * usage_frequency
    )


# ---------------------------------------------------------------------------
# Embedded LanceDB vector store
# ---------------------------------------------------------------------------

class LanceRetrievalEngine:
    """Handles vector indexing and ANN search against embedded LanceDB."""

    _default_path: str = DEFAULT_LANCE_PATH
    _shared: dict[str, "LanceRetrievalEngine"] = {}
    _shared_lock = Lock()

    def __init__(self, lance_path: str = DEFAULT_LANCE_PATH) -> None:
        self._lance_path = lance_path
        self._table_lock = Lock()
        self._sync_lock = Lock()
        self._pending_lock = Lock()
        self._pending: dict[str, LanceMemoryRecord] = {}
        self._flush_thread: Thread | None = None
        self._db = None
        self._table = None

    @property
    def lance_path(self) -> str:
        return self._lance_path

    @classmethod
    def configure_default_path(cls, lance_path: str) -> None:
        cls._default_path = lance_path
        os.environ["CLARA_LANCE_PATH"] = lance_path

    @classmethod
    def get_default(cls) -> "LanceRetrievalEngine":
        lance_path = os.environ.get("CLARA_LANCE_PATH", cls._default_path or DEFAULT_LANCE_PATH)
        with cls._shared_lock:
            engine = cls._shared.get(lance_path)
            if engine is None:
                engine = cls(lance_path)
                cls._shared[lance_path] = engine
            return engine

    @classmethod
    def reset_defaults(cls) -> None:
        with cls._shared_lock:
            for engine in cls._shared.values():
                engine.close()
            cls._shared.clear()
        cls._default_path = os.environ.get("CLARA_LANCE_PATH", DEFAULT_LANCE_PATH)

    def close(self) -> None:
        thread = self._flush_thread
        self.flush_pending_sync()
        if thread is not None and thread.is_alive():
            thread.join(timeout=30)
        # LanceDB embedded connections do not currently expose a close() API.
        self._db = None
        self._table = None

    async def search_candidates(
        self,
        query_vector: list[float],
        *,
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._search_candidates_sync,
                query_vector=query_vector,
                n_candidates=n_candidates,
                memory_types=memory_types,
                user_id=user_id,
            ),
        )

    async def sync_records(self, records: Sequence[LanceMemoryRecord]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            functools.partial(self._sync_records_sync, records=list(records)),
        )

    async def flush_pending(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.flush_pending_sync)

    def sync_records_sync(self, records: Sequence[LanceMemoryRecord]) -> None:
        self._sync_records_sync(records)

    def enqueue_records(self, records: Sequence[LanceMemoryRecord]) -> None:
        if not records:
            return
        with self._pending_lock:
            for record in records:
                if record.memory_id:
                    self._pending[record.memory_id] = record
        self._ensure_background_flush()

    def flush_pending_sync(self) -> None:
        while True:
            with self._pending_lock:
                records = list(self._pending.values())
                self._pending.clear()
            if not records:
                with self._sync_lock:
                    return
            self._sync_records_sync(records)

    def _ensure_background_flush(self) -> None:
        with self._pending_lock:
            if self._flush_thread is not None and self._flush_thread.is_alive():
                return
            self._flush_thread = Thread(
                target=self._background_flush_worker,
                name="clara-lance-sync",
                daemon=True,
            )
            self._flush_thread.start()

    def _background_flush_worker(self) -> None:
        while True:
            try:
                self.flush_pending_sync()
            except Exception:
                logger.exception("Background LanceDB sync failed")
            with self._pending_lock:
                if not self._pending:
                    self._flush_thread = None
                    return

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "''")

    def _ensure_table_sync(self):
        with self._table_lock:
            if self._table is not None:
                return self._table

            Path(self._lance_path).mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(self._lance_path)
            if hasattr(db, "list_tables"):
                table_names = db.list_tables()
                tables = set(getattr(table_names, "tables", table_names) or [])
            elif hasattr(db, "table_names"):
                tables = set(db.table_names())
            else:
                tables = set(getattr(db.list_tables(), "tables", []) or [])
            if LANCE_TABLE_NAME not in tables:
                table = db.create_table(LANCE_TABLE_NAME, schema=LANCE_SCHEMA)
            else:
                table = db.open_table(LANCE_TABLE_NAME)

            self._db = db
            self._table = table
            return table

    def _build_where_clause(
        self,
        *,
        user_id: str | None,
        memory_types: Sequence[MemoryType] | None,
    ) -> str:
        parts = ["status = 'active'"]
        if user_id is not None:
            parts.append(f"user_id = '{self._escape(user_id)}'")
        if memory_types:
            values = ", ".join(
                f"'{self._escape(mem_type.value if isinstance(mem_type, MemoryType) else str(mem_type))}'"
                for mem_type in memory_types
            )
            parts.append(f"memory_type IN ({values})")
        return " AND ".join(parts)

    def _search_candidates_sync(
        self,
        *,
        query_vector: list[float],
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None,
        user_id: str | None,
    ) -> list[tuple[str, float]]:
        if n_candidates <= 0:
            return []
        if memory_types is not None and len(memory_types) == 0:
            return []

        self.flush_pending_sync()
        table = self._ensure_table_sync()
        where_clause = self._build_where_clause(
            user_id=user_id,
            memory_types=memory_types,
        )

        try:
            rows = (
                table.search([float(value) for value in query_vector])
                .metric("cosine")
                .where(where_clause, prefilter=True)
                .limit(n_candidates)
                .select(["memory_id", "_distance"])
                .to_list()
            )
        except Exception:
            logger.exception("LanceDB ANN search failed")
            return []

        pairs: list[tuple[str, float]] = []
        for row in rows:
            memory_id = row.get("memory_id")
            if not memory_id:
                continue
            try:
                distance = float(row.get("_distance", 1.0))
            except (TypeError, ValueError):
                distance = 1.0
            pairs.append((str(memory_id), 1.0 - distance))
        return pairs

    def _sync_records_sync(self, records: Sequence[LanceMemoryRecord]) -> None:
        if not records:
            return

        with self._sync_lock:
            table = self._ensure_table_sync()
            deduped: dict[str, LanceMemoryRecord] = {
                record.memory_id: record for record in records if record.memory_id
            }
            if not deduped:
                return

            upserts = [
                {
                    "memory_id": record.memory_id,
                    "vector": [float(value) for value in record.vector],
                    "user_id": record.user_id,
                    "memory_type": record.memory_type,
                    "status": record.status,
                }
                for record in deduped.values()
                if record.vector is not None
            ]
            if upserts:
                (
                    table.merge_insert("memory_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(upserts)
                )

            for record in deduped.values():
                if record.vector is not None:
                    continue
                table.update(
                    where=f"memory_id = '{self._escape(record.memory_id)}'",
                    values={
                        "user_id": record.user_id,
                        "memory_type": record.memory_type,
                        "status": record.status,
                    },
                )


# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    """Semantic retrieval over the Unified Memory Store."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        *,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
        cache: MemoryCache | None = None,
        lance_engine: LanceRetrievalEngine | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedding_engine
        self._candidate_multiplier = candidate_multiplier
        self._cache = cache
        self._lance = lance_engine or LanceRetrievalEngine.get_default()
        session_info = getattr(getattr(session, "sync_session", session), "info", None)
        if isinstance(session_info, dict):
            session_info.setdefault("_lance_engine", self._lance)
            session_info.setdefault("_clara_embedding_engine", self._embedder)
            if self._cache is not None:
                session_info.setdefault("_clara_cache", self._cache)

    @property
    def lance(self) -> LanceRetrievalEngine:
        return self._lance

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
        track_access: bool = True,
    ) -> RetrievalResult:
        """Search the memory store and return ranked, grouped results."""
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = self._cache.make_query_hash(
                query,
                top_k=top_k,
                memory_types=memory_types,
            )
            cached = await self._cache.get(user_id, cache_key)
            if cached is not None:
                cached_result = await self._hydrate_cached_hits(cached, user_id=user_id)
                if cached_result is not None:
                    if track_access:
                        await self._increment_access_counts([sm.memory for sm in cached_result.all])
                    return cached_result

        query_vector = normalize_embedding_dimensions(
            self._embedder.embed(query),
            target_dimensions=VECTOR_DIMENSIONS,
        )
        n_candidates = top_k * self._candidate_multiplier
        candidates = await self._fetch_candidates(
            query_vector,
            n_candidates=n_candidates,
            memory_types=memory_types,
            user_id=user_id,
        )

        if not candidates:
            return RetrievalResult()

        now = datetime.now(timezone.utc)
        max_access_count = max(self._get_access_count(memory) for memory, _ in candidates)

        scored: list[ScoredMemory] = []
        for memory, similarity in candidates:
            recency = compute_recency_score(memory.updated_at, now)
            access_count = self._get_access_count(memory)
            usage_freq = compute_usage_frequency(access_count, max_access_count)
            final = compute_final_score(
                similarity=similarity,
                confidence=memory.confidence,
                recency_score=recency,
                usage_frequency=usage_freq,
            )
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=final,
                    similarity=similarity,
                    confidence=memory.confidence,
                    recency_score=recency,
                    usage_frequency=usage_freq,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = scored[:top_k]

        if track_access:
            await self._increment_access_counts([item.memory for item in scored])

        result = self._group_scored(scored)
        if self._cache is not None and cache_key is not None:
            await self._cache.set(user_id, cache_key, self._to_cached_hits(scored))
        return result

    async def record_accesses(self, memory_ids: Sequence[Any]) -> None:
        """Fetch memories by ID and bump their access metadata."""
        if not memory_ids:
            return

        stmt = select(Memory).where(Memory.memory_id.in_(list(memory_ids)))
        result = await self._session.execute(stmt)
        memories = result.scalars().all()
        if memories:
            await self._increment_access_counts(list(memories))

    async def _fetch_candidates(
        self,
        query_vector: list[float],
        *,
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Query LanceDB, then hydrate full ORM rows from SQLite."""
        id_pairs = await self._lance.search_candidates(
            query_vector,
            n_candidates=n_candidates,
            memory_types=memory_types,
            user_id=user_id,
        )
        if not id_pairs:
            return []

        memory_ids: list[uuid.UUID] = []
        ordered_ids: list[str] = []
        for memory_id, _similarity in id_pairs:
            try:
                parsed = uuid.UUID(str(memory_id))
            except (TypeError, ValueError):
                continue
            memory_ids.append(parsed)
            ordered_ids.append(str(parsed))

        if not memory_ids:
            return []

        filters = [Memory.memory_id.in_(memory_ids), Memory.status == MemoryStatus.active]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if memory_types:
            filters.append(Memory.memory_type.in_(memory_types))

        stmt = select(Memory).where(and_(*filters))
        result = await self._session.execute(stmt)
        rows = {
            str(memory.memory_id): memory
            for memory in result.scalars().all()
            if memory.status == MemoryStatus.active
        }

        hydrated: list[tuple[Memory, float]] = []
        for memory_id, similarity in id_pairs:
            memory = rows.get(str(memory_id))
            if memory is not None:
                hydrated.append((memory, similarity))
        return hydrated

    @staticmethod
    def _embedding_to_list(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [float(v) for v in value]
        if isinstance(value, tuple):
            return [float(v) for v in value]
        if isinstance(value, str):
            try:
                parsed = literal_eval(value)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed]
            return None
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            parsed = tolist()
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _get_access_count(memory: Memory) -> int:
        meta = memory.metadata_ or {}
        return int(meta.get("access_count", 0))

    async def _increment_access_counts(
        self,
        memories: list[Memory],
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for memory in memories:
            meta: dict[str, Any] = dict(memory.metadata_) if memory.metadata_ else {}
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            meta["last_accessed"] = now_iso
            memory.metadata_ = meta
        await self._session.flush()

    @staticmethod
    def _group_scored(scored: Sequence[ScoredMemory]) -> RetrievalResult:
        result = RetrievalResult()
        for scored_memory in scored:
            match scored_memory.memory.memory_type:
                case MemoryType.belief:
                    result.beliefs.append(scored_memory)
                case MemoryType.event:
                    result.events.append(scored_memory)
                case MemoryType.skill:
                    result.skills.append(scored_memory)
                case MemoryType.world_model:
                    result.world_model.append(scored_memory)
        return result

    @staticmethod
    def _to_cached_hits(scored: Sequence[ScoredMemory]) -> list[CachedScoredMemory]:
        return [
            CachedScoredMemory(
                memory_id=str(item.memory.memory_id),
                score=item.score,
                similarity=item.similarity,
                confidence=item.confidence,
                recency_score=item.recency_score,
                usage_frequency=item.usage_frequency,
            )
            for item in scored
        ]

    async def _hydrate_cached_hits(
        self,
        hits: Sequence[CachedScoredMemory],
        *,
        user_id: str | None,
    ) -> RetrievalResult | None:
        if not hits:
            return RetrievalResult()

        memory_ids = [uuid.UUID(hit.memory_id) for hit in hits]
        stmt = select(Memory).where(Memory.memory_id.in_(memory_ids))
        if user_id is not None:
            stmt = stmt.where(Memory.user_id == user_id)

        result = await self._session.execute(stmt)
        rows = {
            str(memory.memory_id): memory
            for memory in result.scalars().all()
            if memory.status == MemoryStatus.active
        }
        if len(rows) != len(hits):
            return None

        scored: list[ScoredMemory] = []
        for hit in hits:
            memory = rows.get(hit.memory_id)
            if memory is None:
                return None
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=hit.score,
                    similarity=hit.similarity,
                    confidence=hit.confidence,
                    recency_score=hit.recency_score,
                    usage_frequency=hit.usage_frequency,
                )
            )
        return self._group_scored(scored)


# ---------------------------------------------------------------------------
# Automatic SQLite → LanceDB sync on successful commit
# ---------------------------------------------------------------------------

def _snapshot_memory(memory: Memory, *, is_new: bool) -> LanceMemoryRecord | None:
    memory_id = getattr(memory, "memory_id", None)
    memory_type = getattr(memory, "memory_type", None)
    status = getattr(memory, "status", None)
    if memory_id is None or memory_type is None or status is None:
        return None

    vector = RetrievalEngine._embedding_to_list(memory.embedding)
    return LanceMemoryRecord(
        memory_id=str(memory_id),
        vector=vector,
        user_id=memory.user_id or "",
        memory_type=memory_type.value if hasattr(memory_type, "value") else str(memory_type),
        status=status.value if hasattr(status, "value") else str(status),
        is_new=is_new,
    )


def _needs_lance_sync(memory: Memory, *, is_new: bool) -> bool:
    if is_new:
        return True

    if getattr(memory, "_embedding_cache", None) is not None:
        return True

    state = sa_inspect(memory)
    for field_name in ("status", "user_id", "memory_type"):
        if state.attrs[field_name].history.has_changes():
            return True
    return False


def _install_lance_commit_sync() -> None:
    if getattr(_install_lance_commit_sync, "_installed", False):
        return

    @event.listens_for(Session, "before_flush")
    def _capture_memory_objects(session, flush_context, instances) -> None:
        tracked = session.info.setdefault("_lance_tracked_objects", {})
        for obj in list(session.new) + list(session.dirty):
            is_new = obj in session.new
            if isinstance(obj, Memory) and _needs_lance_sync(obj, is_new=is_new):
                tracked[id(obj)] = (obj, obj in session.new)

    @event.listens_for(Session, "after_flush_postexec")
    def _snapshot_tracked_objects(session, flush_context) -> None:
        tracked = session.info.pop("_lance_tracked_objects", {})
        if not tracked:
            return

        snapshots = session.info.setdefault("_lance_pending_snapshots", {})
        for obj, is_new in tracked.values():
            snapshot = _snapshot_memory(obj, is_new=is_new)
            if snapshot is not None:
                snapshots[snapshot.memory_id] = snapshot

    @event.listens_for(Session, "after_commit")
    def _sync_lance_after_commit(session) -> None:
        snapshots = list(session.info.pop("_lance_pending_snapshots", {}).values())
        session.info.pop("_lance_tracked_objects", None)
        if not snapshots:
            return

        try:
            lance = session.info.get("_lance_engine") or LanceRetrievalEngine.get_default()
            lance.enqueue_records(snapshots)
        except Exception:
            logger.exception("Failed to sync memories into LanceDB after commit")

    @event.listens_for(Session, "after_rollback")
    def _clear_lance_pending(session) -> None:
        session.info.pop("_lance_tracked_objects", None)
        session.info.pop("_lance_pending_snapshots", None)

    _install_lance_commit_sync._installed = True


_install_lance_commit_sync()
