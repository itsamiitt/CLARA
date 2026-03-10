"""
CLARA — Vector Retrieval Engine

Retrieves relevant memories from the Unified Memory Store given a natural
language query.  The pipeline:

    1. Embed the query text via :class:`~clara.retrieval.embeddings.EmbeddingEngine`.
    2. Run a pgvector cosine similarity search filtered to ``status = 'active'``.
    3. Rank candidates using the composite scoring formula from CONTEXT.md:

       ``final_score = 0.65 × similarity + 0.20 × confidence
                     + 0.10 × recency + 0.05 × usage_frequency``

    4. Return the top-k results grouped by :class:`~clara.db.models.MemoryType`.
    5. Increment ``access_count`` on every retrieved record.
"""

from __future__ import annotations

import math
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, text as sa_text, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory, MemoryStatus, MemoryType, VECTOR_DIMENSIONS
from clara.retrieval.embeddings import EmbeddingEngine, normalize_embedding_dimensions


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


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_recency_score(updated_at: datetime, now: datetime | None = None) -> float:
    """``recency_score = e^(−λ × days_since_last_accessed)``"""
    now = now or datetime.now(timezone.utc)
    # Ensure both datetimes are timezone-aware (SQLite may return naive ones)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - updated_at).total_seconds() / 86_400.0)
    return math.exp(-RECENCY_LAMBDA * delta_days)


def compute_usage_frequency(access_count: int, max_access_count: int) -> float:
    """``usage_frequency = log(1 + access_count) / log(1 + max_access_count)``

    Returns 0.0 when *max_access_count* is 0 (no accesses recorded yet).
    """
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
# RetrievalEngine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    """Semantic retrieval over the Unified Memory Store.

    Usage::

        engine = RetrievalEngine(session, embedding_engine)
        results = await engine.search("What language does the user prefer?")
        for sm in results.beliefs:
            print(sm.memory.content, sm.score)
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        *,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self._session = session
        self._embedder = embedding_engine
        self._candidate_multiplier = candidate_multiplier

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        memory_types: Sequence[MemoryType] | None = None,
    ) -> RetrievalResult:
        """Search the memory store and return ranked, grouped results.

        Args:
            query: Natural language query text.
            top_k: Maximum number of results to return (across all types).
            memory_types: Optional filter — restrict search to specific
                memory types.  ``None`` means search all types.

        Returns:
            A :class:`RetrievalResult` with results grouped by memory type.
        """
        # 1. Embed the query
        query_vector = normalize_embedding_dimensions(
            self._embedder.embed(query),
            target_dimensions=VECTOR_DIMENSIONS,
        )

        # 2. Vector similarity search (ANN) — pull more candidates than
        #    top_k so the re-ranking stage has room to reshuffle.
        n_candidates = top_k * self._candidate_multiplier

        candidates = await self._fetch_candidates(
            query_vector,
            n_candidates=n_candidates,
            memory_types=memory_types,
        )

        if not candidates:
            return RetrievalResult()

        # 3. Composite scoring
        now = datetime.now(timezone.utc)

        # Determine max_access_count across the candidate set for normalization
        max_access_count = max(
            self._get_access_count(m) for m, _ in candidates
        )

        scored: list[ScoredMemory] = []
        for mem, sim in candidates:
            recency = compute_recency_score(mem.updated_at, now)

            access_count = self._get_access_count(mem)
            usage_freq = compute_usage_frequency(access_count, max_access_count)

            final = compute_final_score(
                similarity=sim,
                confidence=mem.confidence,
                recency_score=recency,
                usage_frequency=usage_freq,
            )

            scored.append(ScoredMemory(
                memory=mem,
                score=final,
                similarity=sim,
                confidence=mem.confidence,
                recency_score=recency,
                usage_frequency=usage_freq,
            ))

        # 4. Sort descending and take top_k
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = scored[:top_k]

        # 5. Increment access_count on retrieved records
        await self._increment_access_counts([s.memory for s in scored])

        # 6. Group by memory type
        result = RetrievalResult()
        for sm in scored:
            match sm.memory.memory_type:
                case MemoryType.belief:
                    result.beliefs.append(sm)
                case MemoryType.event:
                    result.events.append(sm)
                case MemoryType.skill:
                    result.skills.append(sm)
                case MemoryType.world_model:
                    result.world_model.append(sm)

        return result

    # ------------------------------------------------------------------
    # Internal: fetch candidates via pgvector
    # ------------------------------------------------------------------

    async def _fetch_candidates(
        self,
        query_vector: list[float],
        *,
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Run an ANN search and return ``(Memory, cosine_similarity)`` pairs.

        Uses pgvector's ``<=>`` (cosine distance) operator. Similarity is
        calculated as ``1 − distance``.
        """
        if self._dialect_name() == "sqlite":
            return await self._fetch_candidates_sqlite(
                query_vector,
                n_candidates=n_candidates,
                memory_types=memory_types,
            )

        # cosine_distance column expression
        distance_expr = Memory.embedding.cosine_distance(query_vector).label(
            "cosine_distance"
        )

        filters = [
            Memory.status == MemoryStatus.active,
            Memory.embedding.is_not(None),
        ]
        if memory_types:
            filters.append(Memory.memory_type.in_(memory_types))

        stmt = (
            select(Memory, distance_expr)
            .where(and_(*filters))
            .order_by(distance_expr.asc())
            .limit(n_candidates)
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        # Convert distance → similarity
        return [(mem, 1.0 - dist) for mem, dist in rows]

    def _dialect_name(self) -> str | None:
        bind = getattr(self._session, "bind", None)
        dialect = getattr(bind, "dialect", None)
        name = getattr(dialect, "name", None)
        return name if isinstance(name, str) else None

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

    async def _fetch_candidates_sqlite(
        self,
        query_vector: list[float],
        *,
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None = None,
    ) -> list[tuple[Memory, float]]:
        filters = [Memory.status == MemoryStatus.active]
        if memory_types:
            filters.append(Memory.memory_type.in_(memory_types))

        stmt = select(Memory).where(and_(*filters))
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        pairs: list[tuple[Memory, float]] = []
        for mem in rows:
            mem_vector = self._embedding_to_list(mem.embedding)
            if mem_vector is None:
                continue
            pairs.append((mem, self._cosine_similarity(query_vector, mem_vector)))

        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:n_candidates]

    # ------------------------------------------------------------------
    # Internal: access tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _get_access_count(memory: Memory) -> int:
        """Read ``access_count`` from the metadata JSONB (default 0)."""
        meta = memory.metadata_ or {}
        return int(meta.get("access_count", 0))

    async def _increment_access_counts(
        self,
        memories: list[Memory],
    ) -> None:
        """Bump ``access_count`` and ``last_accessed`` in metadata for each record."""
        now_iso = datetime.now(timezone.utc).isoformat()

        for mem in memories:
            meta: dict[str, Any] = dict(mem.metadata_) if mem.metadata_ else {}
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            meta["last_accessed"] = now_iso
            mem.metadata_ = meta

        await self._session.flush()
