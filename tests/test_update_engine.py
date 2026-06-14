"""
Tests for clara.update.engine — MemoryUpdateEngine with mocked dependencies.

All tests mock the AsyncSession, EmbeddingEngine, and RetrievalEngine.
No real database, LLM, or embedding calls are made.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clara.db.models import MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.engine import RetrievalResult, ScoredMemory
from clara.update.engine import (
    CONFLICT_CONFIDENCE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    ActionTaken,
    MemoryUpdateEngine,
    UpdateResult,
    _same_belief,
    classify_memory_type,
    _is_conflicting,
    _domains_differ,
    _map_source_type,
)
from clara.memory.belief import SourceType


class TestClassifyNormalization:
    """Relations are normalised before keyword matching (P1 hardening)."""

    def test_multiword_and_hyphen_relations_match(self):
        assert classify_memory_type(_make_fact(relation="proficient in")) == MemoryType.skill
        assert classify_memory_type(_make_fact(relation="runs on")) == MemoryType.world_model
        assert classify_memory_type(_make_fact(relation="DEPLOYED")) == MemoryType.event

    def test_unknown_relation_defaults_to_belief(self):
        assert classify_memory_type(_make_fact(relation="prefers")) == MemoryType.belief


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    *,
    subject: str = "user",
    relation: str = "uses",
    object: str = "Rust",
    domain: str | None = "systems",
    source_type: str = "user_direct",
    confidence: float = 0.9,
    is_negation: bool = False,
    raw_text: str = "I use Rust for systems work.",
) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        relation=relation,
        object=object,
        domain=domain,
        source_type=source_type,
        confidence=confidence,
        is_negation=is_negation,
        raw_text=raw_text,
    )


class FakeMemory:
    """Stand-in for Memory model (avoids SQLAlchemy instrumentation)."""

    def __init__(
        self,
        *,
        memory_type: MemoryType = MemoryType.belief,
        confidence: float = 0.8,
        status: MemoryStatus = MemoryStatus.active,
        updated_at: datetime | None = None,
        content: dict[str, Any] | None = None,
        memory_id: uuid.UUID | None = None,
    ) -> None:
        self.memory_id = memory_id or uuid.uuid4()
        self.memory_type = memory_type
        self.content = content or {
            "subject": "user", "relation": "uses", "object": "Python",
        }
        self.embedding = [0.0] * 10
        self.confidence = confidence
        self.status = status
        self.decay_rate = 0.02
        self.created_at = updated_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.metadata_: dict[str, Any] = {"access_count": 0}


def _make_scored(
    memory: FakeMemory,
    similarity: float = 0.9,
) -> ScoredMemory:
    return ScoredMemory(
        memory=memory,
        score=similarity * 0.65 + memory.confidence * 0.20,
        similarity=similarity,
        confidence=memory.confidence,
        recency_score=1.0,
        usage_frequency=0.0,
    )


def _empty_retrieval_result() -> RetrievalResult:
    return RetrievalResult()


def _retrieval_result_with(scored_list: list[ScoredMemory]) -> RetrievalResult:
    """Build a RetrievalResult that returns the given items via .all."""
    r = RetrievalResult()
    for sm in scored_list:
        match sm.memory.memory_type:
            case MemoryType.belief:
                r.beliefs.append(sm)
            case MemoryType.event:
                r.events.append(sm)
            case MemoryType.skill:
                r.skills.append(sm)
            case MemoryType.world_model:
                r.world_model.append(sm)
    return r


# ---------------------------------------------------------------------------
# classify_memory_type
# ---------------------------------------------------------------------------

class TestClassifyMemoryType:
    def test_belief_default(self):
        fact = _make_fact(relation="uses")
        assert classify_memory_type(fact) == MemoryType.belief

    def test_event_relations(self):
        for rel in ("started", "completed", "failed", "deployed", "shipped"):
            fact = _make_fact(relation=rel)
            assert classify_memory_type(fact) == MemoryType.event, rel

    def test_skill_relations(self):
        for rel in ("knows", "learned", "proficient_in", "mastered"):
            fact = _make_fact(relation=rel)
            assert classify_memory_type(fact) == MemoryType.skill, rel

    def test_world_model_relations(self):
        for rel in ("is", "has", "contains", "part_of"):
            fact = _make_fact(relation=rel)
            assert classify_memory_type(fact) == MemoryType.world_model, rel

    def test_relation_case_insensitive(self):
        fact = _make_fact(relation="STARTED")
        assert classify_memory_type(fact) == MemoryType.event

    def test_belief_for_preferences(self):
        fact = _make_fact(relation="prefers")
        assert classify_memory_type(fact) == MemoryType.belief


# ---------------------------------------------------------------------------
# _is_conflicting
# ---------------------------------------------------------------------------

class TestIsConflicting:
    def test_same_subject_relation_different_object(self):
        fact = _make_fact(subject="user", relation="uses", object="Rust")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is True

    def test_same_subject_relation_same_object_no_conflict(self):
        fact = _make_fact(subject="user", relation="uses", object="Rust")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Rust",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is False

    def test_different_relation_no_conflict(self):
        fact = _make_fact(subject="user", relation="uses", object="Rust")
        mem = FakeMemory(content={
            "subject": "user", "relation": "prefers", "object": "Python",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is False

    def test_negation_flag_detects_conflict(self):
        fact = _make_fact(
            subject="user", relation="uses", object="Python",
            is_negation=True,
        )
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is True

    def test_case_insensitive(self):
        fact = _make_fact(subject="User", relation="Uses", object="Rust")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is True

    def test_opposite_polarity_same_object_conflicts(self):
        fact = _make_fact(subject="user", relation="uses", object="Python")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
            "is_negation": True,
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is True

    def test_negation_different_object_not_conflict(self):
        fact = _make_fact(subject="user", relation="uses", object="Python", is_negation=True)
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Rust",
        })
        sm = _make_scored(mem)
        assert _is_conflicting(fact, sm) is False


# ---------------------------------------------------------------------------
# _domains_differ
# ---------------------------------------------------------------------------

class TestDomainsDiffer:
    def test_different_domains(self):
        fact = _make_fact(domain="systems")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
            "domain": "web",
        })
        sm = _make_scored(mem)
        assert _domains_differ(fact, sm) is True

    def test_same_domain(self):
        fact = _make_fact(domain="systems")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
            "domain": "systems",
        })
        sm = _make_scored(mem)
        assert _domains_differ(fact, sm) is False


class TestSameBelief:
    def test_same_positive_belief(self):
        fact = _make_fact(subject="user", relation="uses", object="Rust", domain="systems")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Rust", "domain": "systems",
        })
        sm = _make_scored(mem)
        assert _same_belief(fact, sm) is True

    def test_negation_mismatch_not_same(self):
        fact = _make_fact(subject="user", relation="uses", object="Rust", is_negation=True)
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Rust",
        })
        sm = _make_scored(mem)
        assert _same_belief(fact, sm) is False

    def test_fact_no_domain(self):
        fact = _make_fact(domain=None)
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
            "domain": "web",
        })
        sm = _make_scored(mem)
        assert _domains_differ(fact, sm) is False

    def test_candidate_no_domain(self):
        fact = _make_fact(domain="systems")
        mem = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
        })
        sm = _make_scored(mem)
        assert _domains_differ(fact, sm) is False


# ---------------------------------------------------------------------------
# _map_source_type
# ---------------------------------------------------------------------------

class TestMapSourceType:
    def test_valid_mapping(self):
        assert _map_source_type("user_direct") == SourceType.user_direct
        assert _map_source_type("tool_api") == SourceType.tool_api

    def test_unknown_falls_back(self):
        assert _map_source_type("unknown_type") == SourceType.agent_inference


# ---------------------------------------------------------------------------
# UpdateResult
# ---------------------------------------------------------------------------

class TestUpdateResult:
    def test_fields(self):
        uid = uuid.uuid4()
        r = UpdateResult(
            action_taken=ActionTaken.created,
            memory_id=uid,
            conflict_detected=False,
        )
        assert r.action_taken == ActionTaken.created
        assert r.memory_id == uid
        assert r.conflict_detected is False
        assert r.superseded_id is None

    def test_with_superseded(self):
        old_id = uuid.uuid4()
        new_id = uuid.uuid4()
        r = UpdateResult(
            action_taken=ActionTaken.superseded,
            memory_id=new_id,
            conflict_detected=True,
            superseded_id=old_id,
        )
        assert r.superseded_id == old_id


# ---------------------------------------------------------------------------
# MemoryUpdateEngine.process — full pipeline mocked
# ---------------------------------------------------------------------------

class TestProcessNovelFact:
    """When there are no similar memories, a new record should be created."""

    @pytest.mark.asyncio
    async def test_novel_fact_creates_new(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=_empty_retrieval_result())

        engine = MemoryUpdateEngine(session, embedder, retriever)

        # Mock BeliefMemory.store
        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ):
            result = await engine.process(_make_fact())

        assert result.action_taken == ActionTaken.created
        assert result.memory_id == new_id
        assert result.conflict_detected is False
        assert result.superseded_id is None


class TestProcessReinforce:
    """When a similar non-conflicting belief exists, it should be reinforced."""

    @pytest.mark.asyncio
    async def test_reinforces_existing_belief(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        # Existing memory with same content (not conflicting)
        existing = FakeMemory(
            content={"subject": "user", "relation": "uses", "object": "Rust", "domain": "systems"},
            confidence=0.7,
        )
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        # Mock BeliefMemory.update
        updated_mem = FakeMemory(memory_id=existing.memory_id, confidence=0.85)
        with patch.object(
            engine._belief_memory, "update",
            new_callable=AsyncMock, return_value=updated_mem,
        ):
            result = await engine.process(_make_fact(
                subject="user", relation="uses", object="Rust",
            ))

        assert result.action_taken == ActionTaken.reinforced
        assert result.memory_id == existing.memory_id
        assert result.conflict_detected is False


class TestProcessConflictSupersede:
    """When a conflicting memory exists and confidence > 0.6, supersede it."""

    @pytest.mark.asyncio
    async def test_supersedes_old_belief(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        old_id = uuid.uuid4()
        existing = FakeMemory(
            memory_id=old_id,
            content={"subject": "user", "relation": "uses", "object": "Python"},
            confidence=0.85,
        )
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        old_record = FakeMemory(
            memory_id=old_id, status=MemoryStatus.superseded,
        )
        new_record = FakeMemory(memory_id=new_id)

        with patch.object(
            engine._belief_memory, "supersede",
            new_callable=AsyncMock,
            return_value=(old_record, new_record),
        ):
            fact = _make_fact(
                subject="user", relation="uses", object="Rust",
                confidence=0.9,  # > 0.6
            )
            result = await engine.process(fact)

        assert result.action_taken == ActionTaken.superseded
        assert result.memory_id == new_id
        assert result.superseded_id == old_id
        assert result.conflict_detected is True

    @pytest.mark.asyncio
    async def test_exact_match_conflict_without_similarity_search(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=_empty_retrieval_result())

        engine = MemoryUpdateEngine(session, embedder, retriever)

        old_id = uuid.uuid4()
        existing = FakeMemory(
            memory_id=old_id,
            content={"subject": "user", "relation": "uses", "object": "Python", "domain": "systems"},
            confidence=0.85,
        )
        exact_candidate = _make_scored(existing, similarity=1.0)

        new_id = uuid.uuid4()
        old_record = FakeMemory(memory_id=old_id, status=MemoryStatus.superseded)
        new_record = FakeMemory(memory_id=new_id, content={
            "subject": "user", "relation": "uses", "object": "Rust", "domain": "systems",
        })

        with patch.object(
            engine, "_find_exact_belief_candidates",
            new_callable=AsyncMock, return_value=[exact_candidate],
        ), patch.object(
            engine._belief_memory, "supersede",
            new_callable=AsyncMock, return_value=(old_record, new_record),
        ):
            result = await engine.process(_make_fact(
                subject="user", relation="uses", object="Rust", domain="systems", confidence=0.95,
            ))

        assert result.action_taken == ActionTaken.superseded
        assert result.superseded_id == old_id

    @pytest.mark.asyncio
    async def test_same_object_polarity_conflict_beats_other_object_conflict(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=_empty_retrieval_result())

        engine = MemoryUpdateEngine(session, embedder, retriever)

        negated_id = uuid.uuid4()
        rust_id = uuid.uuid4()
        negated_python = FakeMemory(
            memory_id=negated_id,
            content={
                "subject": "user", "relation": "uses", "object": "Python",
                "domain": "systems", "is_negation": True,
            },
            confidence=0.9,
        )
        active_rust = FakeMemory(
            memory_id=rust_id,
            content={
                "subject": "user", "relation": "uses", "object": "Rust",
                "domain": "systems",
            },
            confidence=0.9,
        )

        new_id = uuid.uuid4()
        old_record = FakeMemory(memory_id=negated_id, status=MemoryStatus.superseded)
        new_record = FakeMemory(memory_id=new_id, content={
            "subject": "user", "relation": "uses", "object": "Python", "domain": "systems",
        })

        with patch.object(
            engine, "_find_exact_belief_candidates",
            new_callable=AsyncMock,
            return_value=[_make_scored(negated_python, 1.0), _make_scored(active_rust, 1.0)],
        ), patch.object(
            engine._belief_memory, "supersede",
            new_callable=AsyncMock, return_value=(old_record, new_record),
        ) as mock_supersede:
            result = await engine.process(_make_fact(
                subject="user", relation="uses", object="Python", domain="systems", confidence=0.95,
            ))

        assert result.action_taken == ActionTaken.superseded
        assert result.superseded_id == negated_id
        assert mock_supersede.await_args.args[0] == negated_id


class TestProcessConflictAmbiguous:
    """When confidence is ≤ 0.6, store both."""

    @pytest.mark.asyncio
    async def test_ambiguous_retains_both(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        existing = FakeMemory(
            content={"subject": "user", "relation": "uses", "object": "Python"},
        )
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ):
            fact = _make_fact(
                subject="user", relation="uses", object="Rust",
                confidence=0.5,  # ≤ 0.6
            )
            result = await engine.process(fact)

        assert result.action_taken == ActionTaken.retained_both
        assert result.memory_id == new_id
        assert result.conflict_detected is True
        assert result.superseded_id is None


class TestProcessConflictDomainScoped:
    """When domains differ, retain both with domain tags."""

    @pytest.mark.asyncio
    async def test_domain_scoped_retains_both(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        existing = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
            "domain": "web",
        })
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ):
            fact = _make_fact(
                subject="user", relation="uses", object="Rust",
                domain="systems",  # differs from "web"
                confidence=0.95,  # high confidence, but domains differ
            )
            result = await engine.process(fact)

        assert result.action_taken == ActionTaken.retained_both
        assert result.conflict_detected is True
        assert result.superseded_id is None


class TestProcessNegation:
    """Negation facts should trigger conflict detection."""

    @pytest.mark.asyncio
    async def test_negation_supersedes(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        old_id = uuid.uuid4()
        existing = FakeMemory(
            memory_id=old_id,
            content={"subject": "user", "relation": "uses", "object": "Python"},
        )
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        old_record = FakeMemory(memory_id=old_id, status=MemoryStatus.superseded)
        new_record = FakeMemory(memory_id=new_id)

        with patch.object(
            engine._belief_memory, "supersede",
            new_callable=AsyncMock,
            return_value=(old_record, new_record),
        ) as mock_supersede:
            fact = _make_fact(
                subject="user", relation="uses", object="Python",
                is_negation=True,
                confidence=0.85,  # > 0.6
            )
            result = await engine.process(fact)

        assert result.action_taken == ActionTaken.superseded
        assert result.conflict_detected is True
        assert mock_supersede.await_args.kwargs["is_negation"] is True


class TestProcessEventType:
    """Event facts should always create new records."""

    @pytest.mark.asyncio
    async def test_event_creates_new_even_with_similar(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        # Existing event with same content
        existing = FakeMemory(
            memory_type=MemoryType.event,
            content={"subject": "user", "relation": "deployed", "object": "app"},
        )
        scored = _make_scored(existing, similarity=0.90)
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)
        fact = _make_fact(
            subject="user", relation="deployed", object="app",
        )

        result = await engine.process(fact)

        # Events don't reinforce — they create new records
        # (since same content isn't conflicting and events aren't reinforced)
        assert result.action_taken == ActionTaken.created
        assert result.conflict_detected is False


class TestProcessSearchIntegration:
    """Verify search is called with the right parameters."""

    @pytest.mark.asyncio
    async def test_search_called_with_fact_text(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=_empty_retrieval_result())

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ):
            await engine.process(_make_fact(
                subject="user", relation="uses", object="Rust",
                domain="systems",
            ))

        retriever.search.assert_called_once()
        call_args = retriever.search.call_args
        search_text = call_args[0][0] if call_args[0] else call_args[1].get("query", "")
        assert "user" in search_text
        assert "uses" in search_text
        assert "Rust" in search_text

    @pytest.mark.asyncio
    async def test_process_passes_user_id_to_search_and_store(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=_empty_retrieval_result())

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ) as mock_store:
            await engine.process(_make_fact(), user_id="alice")

        assert retriever.search.await_args.kwargs["user_id"] == "alice"
        assert mock_store.await_args.kwargs["user_id"] == "alice"


class TestProcessBelowThreshold:
    """Candidates below the similarity threshold should be ignored."""

    @pytest.mark.asyncio
    async def test_low_similarity_treated_as_novel(self):
        session = AsyncMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 1536

        # Similar candidate but below threshold
        existing = FakeMemory(content={
            "subject": "user", "relation": "uses", "object": "Python",
        })
        scored = _make_scored(existing, similarity=0.75)  # below 0.82
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=_retrieval_result_with([scored]),
        )

        engine = MemoryUpdateEngine(session, embedder, retriever)

        new_id = uuid.uuid4()
        fake_record = FakeMemory(memory_id=new_id)
        with patch.object(
            engine._belief_memory, "store",
            new_callable=AsyncMock, return_value=fake_record,
        ):
            result = await engine.process(_make_fact())

        assert result.action_taken == ActionTaken.created
        assert result.conflict_detected is False
