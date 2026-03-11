"""Tests for clara.memory.skill — SkillStore."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.core.enums import SkillOutcome
from clara.core.exceptions import MemoryNotFoundError
from clara.db.models import Base, MemoryStatus, MemoryType
from clara.memory.skill import (
    DEPRECATION_THRESHOLD,
    FAILURE_PENALTY_FACTOR,
    SUCCESS_REINFORCEMENT,
    SkillStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session():
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(type_, compiler, **kw):
        return "TEXT"

    try:
        from pgvector.sqlalchemy import Vector

        @compiles(Vector, "sqlite")
        def _compile_vector_sqlite(type_, compiler, **kw):
            return "TEXT"
    except Exception:
        pass

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


# ---------------------------------------------------------------------------
# Create tests
# ---------------------------------------------------------------------------

class TestSkillCreate:
    @pytest.mark.asyncio
    async def test_create_basic_skill(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(
            name="deploy-api",
            trigger_conditions=["deploy", "push to prod"],
            steps=["build image", "push to registry", "kubectl apply"],
            user_id="alice",
        )
        await session.commit()

        assert record.memory_type == MemoryType.skill
        assert record.user_id == "alice"
        assert record.status == MemoryStatus.active
        assert record.confidence == 0.5
        assert record.content["name"] == "deploy-api"
        assert record.content["trigger_conditions"] == ["deploy", "push to prod"]
        assert record.content["steps"] == ["build image", "push to registry", "kubectl apply"]

    @pytest.mark.asyncio
    async def test_create_skill_with_domain(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(
            name="run-tests",
            domain="engineering",
            user_id="alice",
        )
        await session.commit()

        assert record.content["domain"] == "engineering"

    @pytest.mark.asyncio
    async def test_create_skill_has_metadata(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="test-skill")
        await session.commit()

        meta = record.metadata_
        assert meta["outcomes"] == []
        assert meta["success_count"] == 0
        assert meta["failure_count"] == 0
        assert meta["last_used"] is None

    @pytest.mark.asyncio
    async def test_create_skill_default_confidence(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="basic")
        await session.commit()

        assert record.confidence == 0.5

    @pytest.mark.asyncio
    async def test_create_skill_custom_confidence(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="expert", confidence=0.9)
        await session.commit()

        assert record.confidence == 0.9


# ---------------------------------------------------------------------------
# Outcome feedback tests
# ---------------------------------------------------------------------------

class TestSkillFeedback:
    @pytest.mark.asyncio
    async def test_success_increases_confidence(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy", confidence=0.5)
        await session.commit()

        updated = await store.record_outcome(record.memory_id, SkillOutcome.success)
        await session.commit()

        assert updated.confidence == pytest.approx(0.5 + SUCCESS_REINFORCEMENT)
        meta = updated.metadata_
        assert meta["success_count"] == 1
        assert meta["failure_count"] == 0
        assert len(meta["outcomes"]) == 1
        assert meta["outcomes"][0]["result"] == "success"
        assert meta["last_used"] is not None

    @pytest.mark.asyncio
    async def test_failure_decreases_confidence(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy", confidence=0.5)
        await session.commit()

        updated = await store.record_outcome(record.memory_id, SkillOutcome.failure)
        await session.commit()

        assert updated.confidence == pytest.approx(0.5 * FAILURE_PENALTY_FACTOR)
        meta = updated.metadata_
        assert meta["success_count"] == 0
        assert meta["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_success_caps_at_1(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy", confidence=0.95)
        await session.commit()

        updated = await store.record_outcome(record.memory_id, SkillOutcome.success)
        await session.commit()

        assert updated.confidence == 1.0

    @pytest.mark.asyncio
    async def test_multiple_failures_compound(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy", confidence=0.5)
        await session.commit()

        for _ in range(3):
            record = await store.record_outcome(record.memory_id, SkillOutcome.failure)

        expected = 0.5 * (FAILURE_PENALTY_FACTOR ** 3)
        assert record.confidence == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_auto_deprecation(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="fragile-skill", confidence=0.16)
        await session.commit()

        # One failure should drop below DEPRECATION_THRESHOLD (0.15)
        updated = await store.record_outcome(record.memory_id, SkillOutcome.failure)
        await session.commit()

        new_conf = 0.16 * FAILURE_PENALTY_FACTOR
        assert new_conf < DEPRECATION_THRESHOLD
        assert updated.status == MemoryStatus.deprecated

    @pytest.mark.asyncio
    async def test_outcome_with_detail(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy")
        await session.commit()

        updated = await store.record_outcome(
            record.memory_id,
            SkillOutcome.failure,
            detail="Timeout connecting to registry",
        )
        await session.commit()

        meta = updated.metadata_
        assert meta["outcomes"][0]["detail"] == "Timeout connecting to registry"

    @pytest.mark.asyncio
    async def test_nonexistent_skill_raises(self, session: AsyncSession):
        store = SkillStore(session)
        with pytest.raises(MemoryNotFoundError):
            await store.record_outcome(uuid.uuid4(), SkillOutcome.success)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestSkillQueries:
    @pytest.mark.asyncio
    async def test_get_returns_skill(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="deploy")
        await session.commit()

        fetched = await store.get(record.memory_id)
        assert fetched is not None
        assert fetched.memory_id == record.memory_id

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, session: AsyncSession):
        store = SkillStore(session)
        assert await store.get(uuid.uuid4()) is None

    @pytest.mark.asyncio
    async def test_match_by_trigger(self, session: AsyncSession):
        store = SkillStore(session)
        await store.create(
            name="deploy-api",
            trigger_conditions=["deploy", "push to prod"],
            user_id="alice",
        )
        await store.create(
            name="run-tests",
            trigger_conditions=["test", "run tests"],
            user_id="alice",
        )
        await session.commit()

        matched = await store.match("I need to deploy the API", user_id="alice")
        assert len(matched) == 1
        assert matched[0].content["name"] == "deploy-api"

    @pytest.mark.asyncio
    async def test_match_case_insensitive(self, session: AsyncSession):
        store = SkillStore(session)
        await store.create(
            name="deploy",
            trigger_conditions=["Deploy"],
        )
        await session.commit()

        matched = await store.match("I need to deploy")
        assert len(matched) == 1

    @pytest.mark.asyncio
    async def test_match_no_results(self, session: AsyncSession):
        store = SkillStore(session)
        await store.create(
            name="deploy",
            trigger_conditions=["deploy"],
        )
        await session.commit()

        matched = await store.match("I need to run tests")
        assert len(matched) == 0

    @pytest.mark.asyncio
    async def test_get_active_skills(self, session: AsyncSession):
        store = SkillStore(session)
        await store.create(name="skill-1", user_id="alice")
        await store.create(name="skill-2", user_id="alice")
        await store.create(name="skill-3", user_id="bob")
        await session.commit()

        skills = await store.get_active_skills(user_id="alice")
        assert len(skills) == 2
        assert all(s.user_id == "alice" for s in skills)

    @pytest.mark.asyncio
    async def test_get_active_skills_ordered_by_confidence(self, session: AsyncSession):
        store = SkillStore(session)
        await store.create(name="low", confidence=0.3)
        await store.create(name="high", confidence=0.9)
        await store.create(name="mid", confidence=0.6)
        await session.commit()

        skills = await store.get_active_skills()
        assert skills[0].content["name"] == "high"
        assert skills[-1].content["name"] == "low"

    @pytest.mark.asyncio
    async def test_deprecated_skills_excluded_from_active(self, session: AsyncSession):
        store = SkillStore(session)
        record = await store.create(name="fragile", confidence=0.16)
        await store.record_outcome(record.memory_id, SkillOutcome.failure)
        await session.commit()

        skills = await store.get_active_skills()
        assert all(s.content["name"] != "fragile" for s in skills)
