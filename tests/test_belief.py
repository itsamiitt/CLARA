"""
Tests for clara.memory.belief — BeliefMemory CRUD + confidence scoring.

Uses an in-memory SQLite database (via aiosqlite) so no PostgreSQL instance
is needed to run the test suite.  pgvector-specific features (embedding column,
vector indexes) are skipped; standard SQLAlchemy column types still work.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.memory.belief import (
    DEFAULT_DECAY_RATE,
    DEFAULT_OBSERVATION_STRENGTH,
    SOURCE_WEIGHTS,
    BeliefMemory,
    SourceType,
    compute_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    """Create an in-memory async SQLite engine for testing.

    Registers custom type compilations so PostgreSQL-specific types
    (JSONB, UUID, Vector) map to SQLite-compatible equivalents.
    """
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

    # Teach SQLite's compiler how to render PostgreSQL-only types.
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(type_, compiler, **kw):
        return "TEXT"

    # pgvector's Vector type — render as TEXT in SQLite (we don't run
    # vector operations in these tests).
    try:
        from pgvector.sqlalchemy import Vector

        @compiles(Vector, "sqlite")
        def _compile_vector_sqlite(type_, compiler, **kw):
            return "TEXT"
    except Exception:
        pass

    eng = create_async_engine("sqlite+aiosqlite://", echo=False)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    """Provide a fresh async session per test."""
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as sess:
        yield sess
        await sess.rollback()


@pytest_asyncio.fixture
async def beliefs(session: AsyncSession):
    """Return a BeliefMemory instance bound to the test session."""
    return BeliefMemory(session)


# ---------------------------------------------------------------------------
# Unit tests — compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    """Pure-function tests for the Bayesian confidence formula."""

    def test_first_observation_user_direct(self):
        """First observation from a direct user statement → confidence = 1.0."""
        result = compute_confidence(
            prior_confidence=0.0,
            prior_weight=0.0,
            decay_rate=0.02,
            days_since_last_seen=0.0,
            source_weight=1.0,
            observation_strength=1.0,
        )
        assert result == pytest.approx(1.0)

    def test_first_observation_agent_inference(self):
        """First observation from agent inference → confidence = 0.5."""
        result = compute_confidence(
            prior_confidence=0.0,
            prior_weight=0.0,
            decay_rate=0.02,
            days_since_last_seen=0.0,
            source_weight=0.5,
            observation_strength=1.0,
        )
        assert result == pytest.approx(0.5)

    def test_second_observation_reinforces(self):
        """A second confirming observation should raise confidence."""
        first = compute_confidence(
            prior_confidence=0.0,
            prior_weight=0.0,
            decay_rate=0.0,
            days_since_last_seen=0.0,
            source_weight=0.5,
            observation_strength=1.0,
        )
        second = compute_confidence(
            prior_confidence=first,
            prior_weight=1.0,
            decay_rate=0.0,
            days_since_last_seen=0.0,
            source_weight=0.85,
            observation_strength=1.0,
        )
        assert second > first

    def test_decay_lowers_confidence(self):
        """With no new evidence, elapsed time should lower the effective prior."""
        no_decay = compute_confidence(
            prior_confidence=0.8,
            prior_weight=2.0,
            decay_rate=0.0,
            days_since_last_seen=30.0,
            source_weight=0.5,
            observation_strength=1.0,
        )
        with_decay = compute_confidence(
            prior_confidence=0.8,
            prior_weight=2.0,
            decay_rate=0.05,
            days_since_last_seen=30.0,
            source_weight=0.5,
            observation_strength=1.0,
        )
        assert with_decay < no_decay

    def test_confidence_clamped_to_unit_interval(self):
        """Result should never exceed 1.0 or go below 0.0."""
        high = compute_confidence(
            prior_confidence=1.0,
            prior_weight=0.0,
            decay_rate=0.0,
            days_since_last_seen=0.0,
            source_weight=1.0,
            observation_strength=5.0,
        )
        assert high <= 1.0

        low = compute_confidence(
            prior_confidence=0.0,
            prior_weight=100.0,
            decay_rate=1.0,
            days_since_last_seen=1000.0,
            source_weight=0.0,
            observation_strength=0.0,
        )
        assert low >= 0.0

    def test_decay_factor_formula(self):
        """Verify explicit decay_factor = e^(−rate × days)."""
        rate = 0.02
        days = 10.0
        expected_decay = math.exp(-rate * days)

        result = compute_confidence(
            prior_confidence=1.0,
            prior_weight=1.0,
            decay_rate=rate,
            days_since_last_seen=days,
            source_weight=0.0,
            observation_strength=0.0,
        )
        # numerator = 1.0 * decay_factor + 0 = decay_factor
        # denominator = 1.0 + 1.0 = 2
        assert result == pytest.approx(expected_decay / 2.0)


# ---------------------------------------------------------------------------
# Integration tests — BeliefMemory.store
# ---------------------------------------------------------------------------

class TestStore:
    """Tests for creating new beliefs."""

    @pytest.mark.asyncio
    async def test_store_creates_record(self, beliefs: BeliefMemory, session: AsyncSession):
        record = await beliefs.store(
            subject="user",
            relation="uses",
            object_="Rust",
            domain="systems",
            source=SourceType.user_direct,
            raw_text="I switched from Python to Rust.",
        )
        await session.commit()

        assert record.memory_id is not None
        assert record.memory_type == MemoryType.belief
        assert record.status == MemoryStatus.active
        assert record.content["subject"] == "user"
        assert record.content["relation"] == "uses"
        assert record.content["object"] == "Rust"
        assert record.content["domain"] == "systems"

    @pytest.mark.asyncio
    async def test_store_initial_confidence_user_direct(self, beliefs: BeliefMemory, session: AsyncSession):
        """User direct statement → initial confidence = 1.0."""
        record = await beliefs.store(
            subject="user",
            relation="prefers",
            object_="dark mode",
            source=SourceType.user_direct,
            raw_text="I prefer dark mode.",
        )
        await session.commit()
        assert record.confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_store_initial_confidence_agent_inference(self, beliefs: BeliefMemory, session: AsyncSession):
        """Agent inference → initial confidence = 0.5."""
        record = await beliefs.store(
            subject="user",
            relation="prefers",
            object_="tabs",
            source=SourceType.agent_inference,
            raw_text=None,
        )
        await session.commit()
        assert record.confidence == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_store_evidence_trail(self, beliefs: BeliefMemory, session: AsyncSession):
        record = await beliefs.store(
            subject="user",
            relation="uses",
            object_="Rust",
            source=SourceType.tool_api,
            raw_text="Detected Cargo.toml in repo.",
        )
        await session.commit()

        evidence = record.metadata_["evidence"]
        assert len(evidence) == 1
        assert evidence[0]["source"] == "tool_api"
        assert evidence[0]["text"] == "Detected Cargo.toml in repo."

    @pytest.mark.asyncio
    async def test_store_without_domain(self, beliefs: BeliefMemory, session: AsyncSession):
        """Domain is optional and should be absent from content when not provided."""
        record = await beliefs.store(
            subject="user",
            relation="uses",
            object_="VS Code",
            source=SourceType.user_direct,
            raw_text="I use VS Code.",
        )
        await session.commit()
        assert "domain" not in record.content

    @pytest.mark.asyncio
    async def test_store_sets_decay_rate(self, beliefs: BeliefMemory, session: AsyncSession):
        record = await beliefs.store(
            subject="user",
            relation="location",
            object_="Berlin",
            source=SourceType.user_direct,
            raw_text="I live in Berlin.",
            decay_rate=0.005,
        )
        await session.commit()
        assert record.decay_rate == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Integration tests — BeliefMemory.get
# ---------------------------------------------------------------------------

class TestGet:
    """Tests for retrieving beliefs by UUID."""

    @pytest.mark.asyncio
    async def test_get_existing(self, beliefs: BeliefMemory, session: AsyncSession):
        stored = await beliefs.store(
            subject="user", relation="uses", object_="Rust",
            source=SourceType.user_direct, raw_text="test",
        )
        await session.commit()

        fetched = await beliefs.get(stored.memory_id)
        assert fetched is not None
        assert fetched.memory_id == stored.memory_id

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, beliefs: BeliefMemory):
        result = await beliefs.get(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests — BeliefMemory.update
# ---------------------------------------------------------------------------

class TestUpdate:
    """Tests for reinforcing existing beliefs with new evidence."""

    @pytest.mark.asyncio
    async def test_update_increases_confidence(self, beliefs: BeliefMemory, session: AsyncSession):
        """Reinforcing an agent-inferred belief with user evidence should boost it."""
        record = await beliefs.store(
            subject="user", relation="uses", object_="Neovim",
            source=SourceType.agent_inference, raw_text="Saw .nvim config.",
        )
        await session.commit()

        initial_conf = record.confidence  # 0.5
        updated = await beliefs.update(
            record.memory_id,
            source=SourceType.user_direct,
            raw_text="Yes, I use Neovim.",
        )
        await session.commit()

        assert updated.confidence > initial_conf

    @pytest.mark.asyncio
    async def test_update_appends_evidence(self, beliefs: BeliefMemory, session: AsyncSession):
        record = await beliefs.store(
            subject="user", relation="prefers", object_="dark mode",
            source=SourceType.user_direct, raw_text="I like dark mode.",
        )
        await session.commit()

        await beliefs.update(
            record.memory_id,
            source=SourceType.system,
            raw_text="System detected dark theme in settings.",
        )
        await session.commit()

        refreshed = await beliefs.get(record.memory_id)
        assert refreshed is not None
        evidence = refreshed.metadata_["evidence"]
        assert len(evidence) == 2
        assert evidence[1]["source"] == "system"

    @pytest.mark.asyncio
    async def test_update_increments_prior_weight(self, beliefs: BeliefMemory, session: AsyncSession):
        record = await beliefs.store(
            subject="user", relation="uses", object_="Linux",
            source=SourceType.user_direct, raw_text="I use Linux.",
        )
        await session.commit()
        assert record.metadata_["prior_weight"] == pytest.approx(1.0)

        updated = await beliefs.update(
            record.memory_id,
            source=SourceType.tool_api,
            raw_text="Detected Linux kernel headers.",
        )
        await session.commit()
        assert updated.metadata_["prior_weight"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self, beliefs: BeliefMemory):
        with pytest.raises(ValueError, match="not found"):
            await beliefs.update(
                uuid.uuid4(),
                source=SourceType.user_direct,
                raw_text="test",
            )

    @pytest.mark.asyncio
    async def test_update_superseded_raises(self, beliefs: BeliefMemory, session: AsyncSession):
        """Cannot update a belief that has already been superseded."""
        old = await beliefs.store(
            subject="user", relation="uses", object_="Python",
            source=SourceType.user_direct, raw_text="I use Python.",
        )
        await session.commit()

        await beliefs.supersede(
            old.memory_id,
            subject="user", relation="uses", object_="Rust",
            source=SourceType.user_direct, raw_text="Switched to Rust.",
        )
        await session.commit()

        with pytest.raises(ValueError, match="superseded"):
            await beliefs.update(
                old.memory_id,
                source=SourceType.user_direct,
                raw_text="Still using Python.",
            )


# ---------------------------------------------------------------------------
# Integration tests — BeliefMemory.supersede
# ---------------------------------------------------------------------------

class TestSupersede:
    """Tests for replacing one belief with another."""

    @pytest.mark.asyncio
    async def test_supersede_marks_old_as_superseded(self, beliefs: BeliefMemory, session: AsyncSession):
        old = await beliefs.store(
            subject="user", relation="uses", object_="Python",
            source=SourceType.user_direct, raw_text="I use Python.",
        )
        await session.commit()

        old_updated, new = await beliefs.supersede(
            old.memory_id,
            subject="user", relation="uses", object_="Rust",
            source=SourceType.user_direct,
            raw_text="I switched to Rust.",
        )
        await session.commit()

        assert old_updated.status == MemoryStatus.superseded
        assert new.status == MemoryStatus.active

    @pytest.mark.asyncio
    async def test_supersede_links_records(self, beliefs: BeliefMemory, session: AsyncSession):
        old = await beliefs.store(
            subject="user", relation="uses", object_="Python",
            source=SourceType.user_direct, raw_text="I use Python.",
        )
        await session.commit()

        old_updated, new = await beliefs.supersede(
            old.memory_id,
            subject="user", relation="uses", object_="Rust",
            source=SourceType.user_direct,
            raw_text="I switched to Rust.",
        )
        await session.commit()

        # Old → new link
        assert old_updated.metadata_["superseded_by"] == str(new.memory_id)
        # New → old back-link
        assert new.metadata_["supersedes"] == str(old_updated.memory_id)

    @pytest.mark.asyncio
    async def test_supersede_creates_new_active_belief(self, beliefs: BeliefMemory, session: AsyncSession):
        old = await beliefs.store(
            subject="user", relation="db", object_="MySQL",
            source=SourceType.user_direct, raw_text="I use MySQL.",
        )
        await session.commit()

        _, new = await beliefs.supersede(
            old.memory_id,
            subject="user", relation="db", object_="PostgreSQL",
            domain="backend",
            source=SourceType.user_direct,
            raw_text="Migrated to PostgreSQL.",
        )
        await session.commit()

        assert new.content["object"] == "PostgreSQL"
        assert new.content.get("domain") == "backend"
        assert new.confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_supersede_nonexistent_raises(self, beliefs: BeliefMemory):
        with pytest.raises(ValueError, match="not found"):
            await beliefs.supersede(
                uuid.uuid4(),
                subject="user", relation="uses", object_="Go",
                source=SourceType.user_direct,
                raw_text="test",
            )

    @pytest.mark.asyncio
    async def test_supersede_already_superseded_raises(self, beliefs: BeliefMemory, session: AsyncSession):
        old = await beliefs.store(
            subject="user", relation="editor", object_="Vim",
            source=SourceType.user_direct, raw_text="I use Vim.",
        )
        await session.commit()

        await beliefs.supersede(
            old.memory_id,
            subject="user", relation="editor", object_="Neovim",
            source=SourceType.user_direct, raw_text="Switched to Neovim.",
        )
        await session.commit()

        with pytest.raises(ValueError, match="superseded"):
            await beliefs.supersede(
                old.memory_id,
                subject="user", relation="editor", object_="Emacs",
                source=SourceType.user_direct, raw_text="test",
            )


# ---------------------------------------------------------------------------
# Integration tests — BeliefMemory.get_active_beliefs
# ---------------------------------------------------------------------------

class TestGetActiveBeliefs:
    """Tests for querying filtered sets of active beliefs."""

    @pytest.mark.asyncio
    async def test_returns_only_active(self, beliefs: BeliefMemory, session: AsyncSession):
        await beliefs.store(
            subject="user", relation="uses", object_="Python",
            source=SourceType.user_direct, raw_text="test",
        )
        old = await beliefs.store(
            subject="user", relation="uses", object_="Java",
            source=SourceType.user_direct, raw_text="test",
        )
        await session.commit()

        await beliefs.supersede(
            old.memory_id,
            subject="user", relation="uses", object_="Kotlin",
            source=SourceType.user_direct, raw_text="test",
        )
        await session.commit()

        active = await beliefs.get_active_beliefs(subject="user")
        statuses = {r.status for r in active}
        assert MemoryStatus.superseded not in statuses

    @pytest.mark.asyncio
    async def test_filter_by_subject(self, beliefs: BeliefMemory, session: AsyncSession):
        await beliefs.store(
            subject="user", relation="uses", object_="Rust",
            source=SourceType.user_direct, raw_text="test",
        )
        await beliefs.store(
            subject="project", relation="uses", object_="Axum",
            source=SourceType.user_direct, raw_text="test",
        )
        await session.commit()

        results = await beliefs.get_active_beliefs(subject="user")
        assert all(r.content["subject"] == "user" for r in results)

    @pytest.mark.asyncio
    async def test_ordered_by_confidence_desc(self, beliefs: BeliefMemory, session: AsyncSession):
        await beliefs.store(
            subject="user", relation="uses", object_="low-conf tool",
            source=SourceType.agent_inference, raw_text="test",
        )
        await beliefs.store(
            subject="user", relation="uses", object_="high-conf tool",
            source=SourceType.user_direct, raw_text="test",
        )
        await session.commit()

        results = await beliefs.get_active_beliefs(subject="user")
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)
