"""
Tests for clara.retrieval.engine — RetrievalEngine with mocked backends.

All tests mock the AsyncSession hydration query, the LanceDB ANN search, and
the EmbeddingEngine. This validates the scoring formula, grouping, access
tracking, and edge cases entirely in-process.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from clara.db.models import MemoryStatus, MemoryType
from clara.retrieval.engine import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    RECENCY_LAMBDA,
    W_CONFIDENCE,
    W_RECENCY,
    W_SIMILARITY,
    W_USAGE,
    RetrievalEngine,
    RetrievalResult,
    ScoredMemory,
    compute_final_score,
    compute_recency_score,
    compute_usage_frequency,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-in for Memory (avoids SQLAlchemy instrumentation)
# ---------------------------------------------------------------------------

class FakeMemory:
    """Plain object mimicking the Memory model for testing without SQLAlchemy
    instrumented attribute machinery."""

    def __init__(
        self,
        *,
        memory_type: MemoryType = MemoryType.belief,
        confidence: float = 0.8,
        status: MemoryStatus = MemoryStatus.active,
        updated_at: datetime | None = None,
        access_count: int = 0,
        memory_id: uuid.UUID | None = None,
    ) -> None:
        self.memory_id = memory_id or uuid.uuid4()
        self.memory_type = memory_type
        self.content: dict[str, Any] = {
            "subject": "user", "relation": "uses", "object": "Rust",
        }
        self.embedding = [0.0] * 10
        self.confidence = confidence
        self.status = status
        self.decay_rate = 0.02
        self.created_at = updated_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.metadata_: dict[str, Any] = {"access_count": access_count}


def _make_embedder(dims: int = 1536) -> MagicMock:
    """Return a mocked EmbeddingEngine."""
    embedder = MagicMock()
    embedder.dimensions = dims
    embedder.embed.return_value = [0.0] * dims
    return embedder


def _make_async_session(
    memories: list[FakeMemory] | None = None,
) -> AsyncMock:
    """Return a mocked AsyncSession whose execute() hydrates Memory rows."""
    session = AsyncMock()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = memories or []
    mock_result.scalars.return_value = mock_scalars
    session.execute = AsyncMock(return_value=mock_result)
    session.flush = AsyncMock()

    return session


def _stub_lance(engine: RetrievalEngine, pairs: list[tuple[FakeMemory, float]]) -> AsyncMock:
    mock = AsyncMock(return_value=[(str(mem.memory_id), similarity) for mem, similarity in pairs])
    engine._lance.search_candidates = mock  # type: ignore[method-assign]
    return mock


# ---------------------------------------------------------------------------
# Pure-function tests — scoring helpers
# ---------------------------------------------------------------------------

class TestComputeRecencyScore:
    def test_now_returns_one(self):
        now = datetime.now(timezone.utc)
        assert compute_recency_score(now, now) == pytest.approx(1.0)

    def test_old_record_decays(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=100)
        score = compute_recency_score(old, now)
        expected = math.exp(-RECENCY_LAMBDA * 100)
        assert score == pytest.approx(expected)

    def test_monotonically_decreasing(self):
        now = datetime.now(timezone.utc)
        s1 = compute_recency_score(now - timedelta(days=1), now)
        s10 = compute_recency_score(now - timedelta(days=10), now)
        s100 = compute_recency_score(now - timedelta(days=100), now)
        assert s1 > s10 > s100

    def test_never_negative(self):
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=100_000)
        assert compute_recency_score(ancient, now) >= 0.0


class TestComputeUsageFrequency:
    def test_zero_max_returns_zero(self):
        assert compute_usage_frequency(0, 0) == 0.0
        assert compute_usage_frequency(5, 0) == 0.0

    def test_max_count_returns_one(self):
        assert compute_usage_frequency(10, 10) == pytest.approx(1.0)

    def test_half_count(self):
        result = compute_usage_frequency(5, 10)
        expected = math.log(6) / math.log(11)
        assert result == pytest.approx(expected)

    def test_monotonically_increasing(self):
        f1 = compute_usage_frequency(1, 100)
        f50 = compute_usage_frequency(50, 100)
        f100 = compute_usage_frequency(100, 100)
        assert f1 < f50 < f100


class TestComputeFinalScore:
    def test_weights_sum_to_one(self):
        assert W_SIMILARITY + W_CONFIDENCE + W_RECENCY + W_USAGE == pytest.approx(1.0)

    def test_perfect_scores(self):
        score = compute_final_score(
            similarity=1.0, confidence=1.0, recency_score=1.0, usage_frequency=1.0,
        )
        assert score == pytest.approx(1.0)

    def test_zero_scores(self):
        score = compute_final_score(
            similarity=0.0, confidence=0.0, recency_score=0.0, usage_frequency=0.0,
        )
        assert score == pytest.approx(0.0)

    def test_similarity_weight_dominates(self):
        """Similarity (0.65) should have the largest effect."""
        high_sim = compute_final_score(1.0, 0.0, 0.0, 0.0)
        high_conf = compute_final_score(0.0, 1.0, 0.0, 0.0)
        assert high_sim > high_conf

    def test_known_values(self):
        score = compute_final_score(
            similarity=0.9,
            confidence=0.8,
            recency_score=0.7,
            usage_frequency=0.5,
        )
        expected = 0.65 * 0.9 + 0.20 * 0.8 + 0.10 * 0.7 + 0.05 * 0.5
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RetrievalResult data class
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_empty_result(self):
        r = RetrievalResult()
        assert r.total == 0
        assert r.all == []

    def test_total_counts_all_types(self):
        r = RetrievalResult()
        r.beliefs = [MagicMock() for _ in range(2)]
        r.events = [MagicMock() for _ in range(3)]
        r.skills = [MagicMock()]
        assert r.total == 6

    def test_all_sorted_descending(self):
        mem1 = FakeMemory()
        mem2 = FakeMemory()
        sm1 = ScoredMemory(
            memory=mem1, score=0.5,
            similarity=0.5, confidence=0.5, recency_score=0.5, usage_frequency=0.5,
        )
        sm2 = ScoredMemory(
            memory=mem2, score=0.9,
            similarity=0.9, confidence=0.9, recency_score=0.9, usage_frequency=0.9,
        )
        r = RetrievalResult(beliefs=[sm1], events=[sm2])
        assert r.all[0].score > r.all[1].score


# ---------------------------------------------------------------------------
# RetrievalEngine — integration with mocks
# ---------------------------------------------------------------------------

class TestRetrievalEngineSearch:
    """Tests for RetrievalEngine.search() with mocked session + embedder."""

    # -- Basic search behavior --

    @pytest.mark.asyncio
    async def test_search_returns_grouped_results(self):
        belief = FakeMemory(memory_type=MemoryType.belief, confidence=0.9)
        event = FakeMemory(memory_type=MemoryType.event, confidence=0.8)
        skill = FakeMemory(memory_type=MemoryType.skill, confidence=0.7)

        session = _make_async_session([belief, event, skill])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [
            (belief, 0.9),
            (event, 0.8),
            (skill, 0.7),
        ])
        result = await engine.search("test query")

        assert len(result.beliefs) == 1
        assert len(result.events) == 1
        assert len(result.skills) == 1
        assert result.total == 3

    @pytest.mark.asyncio
    async def test_search_embeds_query(self):
        session = _make_async_session([])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [])
        await engine.search("What language does user prefer?")

        embedder.embed.assert_called_once_with("What language does user prefer?")

    @pytest.mark.asyncio
    async def test_search_empty_candidates(self):
        session = _make_async_session([])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [])
        result = await engine.search("test")

        assert result.total == 0
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        memories = [
            (FakeMemory(confidence=0.9 - i * 0.05), 0.9 - i * 0.05)
            for i in range(10)
        ]
        session = _make_async_session([mem for mem, _similarity in memories])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, memories)
        result = await engine.search("test", top_k=3)

        assert result.total == 3

    # -- Scoring correctness --

    @pytest.mark.asyncio
    async def test_scoring_formula(self):
        """Verify that the exact CONTEXT.md formula is applied."""
        now = datetime.now(timezone.utc)
        mem = FakeMemory(
            confidence=0.8,
            updated_at=now - timedelta(days=10),
            access_count=5,
        )
        session = _make_async_session([mem])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(mem, 0.85)])
        result = await engine.search("test", top_k=1)

        sm = result.beliefs[0]

        # Verify individual components
        assert sm.similarity == pytest.approx(0.85)
        assert sm.confidence == pytest.approx(0.8)

        expected_recency = math.exp(-0.01 * 10)
        assert sm.recency_score == pytest.approx(expected_recency, abs=0.01)

        # Only one candidate → max_access_count = 5 → usage = log(6)/log(6) = 1.0
        assert sm.usage_frequency == pytest.approx(1.0)

        expected_score = (
            0.65 * 0.85 + 0.20 * 0.8 + 0.10 * expected_recency + 0.05 * 1.0
        )
        assert sm.score == pytest.approx(expected_score, abs=0.01)

    @pytest.mark.asyncio
    async def test_higher_similarity_ranks_higher(self):
        """More similar memories should rank above less similar ones."""
        now = datetime.now(timezone.utc)
        high_sim = FakeMemory(confidence=0.5, updated_at=now)
        low_sim = FakeMemory(confidence=0.5, updated_at=now)

        session = _make_async_session([high_sim, low_sim])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [
            (high_sim, 0.95),
            (low_sim, 0.50),
        ])
        result = await engine.search("test")

        all_results = result.all
        assert all_results[0].similarity > all_results[1].similarity

    # -- Access tracking --

    @pytest.mark.asyncio
    async def test_increments_access_count(self):
        mem = FakeMemory(access_count=3)
        session = _make_async_session([mem])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(mem, 0.9)])
        await engine.search("test")

        # access_count should have been bumped from 3 → 4
        assert mem.metadata_["access_count"] == 4

    @pytest.mark.asyncio
    async def test_sets_last_accessed(self):
        mem = FakeMemory()
        session = _make_async_session([mem])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(mem, 0.9)])
        await engine.search("test")

        assert "last_accessed" in mem.metadata_

    @pytest.mark.asyncio
    async def test_access_count_starts_at_zero(self):
        """If metadata has no access_count, it should default to 0 then bump to 1."""
        mem = FakeMemory()
        mem.metadata_ = {}  # no access_count key

        session = _make_async_session([mem])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(mem, 0.9)])
        await engine.search("test")

        assert mem.metadata_["access_count"] == 1

    @pytest.mark.asyncio
    async def test_flush_called_after_access_update(self):
        mem = FakeMemory()
        session = _make_async_session([mem])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(mem, 0.9)])
        await engine.search("test")

        session.flush.assert_called()

    # -- World model grouping --

    @pytest.mark.asyncio
    async def test_world_model_grouped_correctly(self):
        wm = FakeMemory(memory_type=MemoryType.world_model, confidence=0.9)
        session = _make_async_session([wm])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [(wm, 0.9)])
        result = await engine.search("test")

        assert len(result.world_model) == 1
        assert result.beliefs == []
        assert result.events == []
        assert result.skills == []

    # -- Multiple types mixed --

    @pytest.mark.asyncio
    async def test_mixed_types_all_grouped(self):
        now = datetime.now(timezone.utc)
        b1 = FakeMemory(memory_type=MemoryType.belief, confidence=0.9, updated_at=now)
        b2 = FakeMemory(memory_type=MemoryType.belief, confidence=0.8, updated_at=now)
        e1 = FakeMemory(memory_type=MemoryType.event, confidence=0.7, updated_at=now)
        s1 = FakeMemory(memory_type=MemoryType.skill, confidence=0.6, updated_at=now)
        w1 = FakeMemory(memory_type=MemoryType.world_model, confidence=0.5, updated_at=now)

        session = _make_async_session([b1, b2, e1, s1, w1])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [
            (b1, 0.90),
            (b2, 0.85),
            (e1, 0.80),
            (s1, 0.75),
            (w1, 0.70),
        ])
        result = await engine.search("test", top_k=10)

        assert len(result.beliefs) == 2
        assert len(result.events) == 1
        assert len(result.skills) == 1
        assert len(result.world_model) == 1
        assert result.total == 5

    # -- Usage frequency normalization --

    @pytest.mark.asyncio
    async def test_usage_frequency_normalized(self):
        """The most-accessed record should have usage_frequency = 1.0."""
        now = datetime.now(timezone.utc)
        popular = FakeMemory(confidence=0.5, updated_at=now, access_count=100)
        unpopular = FakeMemory(confidence=0.5, updated_at=now, access_count=1)

        session = _make_async_session([popular, unpopular])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder)
        _stub_lance(engine, [
            (popular, 0.8),
            (unpopular, 0.8),
        ])
        result = await engine.search("test")

        all_results = result.all
        popular_scored = [s for s in all_results if s.memory is popular][0]
        assert popular_scored.usage_frequency == pytest.approx(1.0)

        unpopular_scored = [s for s in all_results if s.memory is unpopular][0]
        expected = math.log(2) / math.log(101)
        assert unpopular_scored.usage_frequency == pytest.approx(expected)

    # -- Candidate multiplier --

    @pytest.mark.asyncio
    async def test_candidate_multiplier(self):
        """Engine should fetch top_k × multiplier candidates from the index."""
        session = _make_async_session([])
        embedder = _make_embedder()

        engine = RetrievalEngine(session, embedder, candidate_multiplier=5)
        lance = _stub_lance(engine, [])
        await engine.search("test", top_k=4)

        assert lance.await_args.kwargs["n_candidates"] == 20
