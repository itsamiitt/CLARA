"""Tests for clara.memory.event — EventStore."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.core.enums import EventStatus
from clara.core.exceptions import InvalidTransitionError, MemoryNotFoundError
from clara.db.models import Base, MemoryStatus, MemoryType
from clara.memory.event import EventStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session():

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

class TestEventCreate:
    @pytest.mark.asyncio
    async def test_create_basic_event(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user",
            event_type="deployed",
            description="Deployed v2.1 to production",
            user_id="alice",
        )
        await session.commit()

        assert record.memory_type == MemoryType.event
        assert record.user_id == "alice"
        assert record.status == MemoryStatus.active
        assert record.decay_rate == 0.0
        assert record.confidence == 1.0
        assert record.content["event_type"] == "deployed"
        assert record.content["event_status"] == "created"
        assert record.content["subject"] == "user"

    @pytest.mark.asyncio
    async def test_create_event_with_domain(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user",
            event_type="committed",
            description="Committed code to main",
            domain="engineering",
            user_id="alice",
        )
        await session.commit()

        assert record.content["domain"] == "engineering"

    @pytest.mark.asyncio
    async def test_create_event_has_history(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user",
            event_type="deployed",
            description="test",
        )
        await session.commit()

        meta = record.metadata_
        assert "event_history" in meta
        assert len(meta["event_history"]) == 1
        assert meta["event_history"][0]["status"] == "created"


# ---------------------------------------------------------------------------
# Lifecycle transition tests
# ---------------------------------------------------------------------------

class TestEventTransitions:
    @pytest.mark.asyncio
    async def test_transition_created_to_in_progress(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user", event_type="migrating", description="DB migration",
        )
        await session.commit()

        updated = await store.update_outcome(record.memory_id, EventStatus.in_progress)
        await session.commit()

        assert updated.content["event_status"] == "in_progress"
        meta = updated.metadata_
        assert len(meta["event_history"]) == 2
        assert meta["event_history"][1]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_transition_created_to_completed(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user", event_type="deployed", description="test",
        )
        await session.commit()

        updated = await store.update_outcome(
            record.memory_id,
            EventStatus.completed,
            outcome_detail="Successfully deployed",
        )
        await session.commit()

        assert updated.content["event_status"] == "completed"
        assert updated.content["outcome_detail"] == "Successfully deployed"
        meta = updated.metadata_
        assert meta["event_history"][-1]["detail"] == "Successfully deployed"

    @pytest.mark.asyncio
    async def test_transition_in_progress_to_failed(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user", event_type="migrating", description="test",
        )
        await store.update_outcome(record.memory_id, EventStatus.in_progress)
        await session.commit()

        updated = await store.update_outcome(record.memory_id, EventStatus.failed)
        await session.commit()

        assert updated.content["event_status"] == "failed"
        meta = updated.metadata_
        assert len(meta["event_history"]) == 3

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user", event_type="deployed", description="test",
        )
        await store.update_outcome(record.memory_id, EventStatus.completed)
        await session.commit()

        with pytest.raises(InvalidTransitionError) as exc_info:
            await store.update_outcome(record.memory_id, EventStatus.in_progress)

        assert exc_info.value.current_status == "completed"
        assert exc_info.value.target_status == "in_progress"

    @pytest.mark.asyncio
    async def test_nonexistent_event_raises(self, session: AsyncSession):
        store = EventStore(session)
        fake_id = uuid.uuid4()

        with pytest.raises(MemoryNotFoundError):
            await store.update_outcome(fake_id, EventStatus.completed)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestEventQueries:
    @pytest.mark.asyncio
    async def test_get_returns_event(self, session: AsyncSession):
        store = EventStore(session)
        record = await store.create(
            subject="user", event_type="deployed", description="test",
        )
        await session.commit()

        fetched = await store.get(record.memory_id)
        assert fetched is not None
        assert fetched.memory_id == record.memory_id

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, session: AsyncSession):
        store = EventStore(session)
        result = await store.get(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_timeline(self, session: AsyncSession):
        store = EventStore(session)
        await store.create(subject="user", event_type="deployed", description="v1", user_id="alice")
        await store.create(subject="user", event_type="committed", description="fix", user_id="alice")
        await store.create(subject="user", event_type="deployed", description="v2", user_id="bob")
        await session.commit()

        timeline = await store.get_timeline(user_id="alice")
        assert len(timeline) == 2
        assert all(e.user_id == "alice" for e in timeline)

    @pytest.mark.asyncio
    async def test_get_timeline_by_subject(self, session: AsyncSession):
        store = EventStore(session)
        await store.create(subject="user", event_type="deployed", description="v1")
        await store.create(subject="system", event_type="restarted", description="crash")
        await session.commit()

        timeline = await store.get_timeline(subject="system")
        assert len(timeline) == 1
        assert timeline[0].content["subject"] == "system"

    @pytest.mark.asyncio
    async def test_get_timeline_respects_limit(self, session: AsyncSession):
        store = EventStore(session)
        for i in range(5):
            await store.create(subject="user", event_type="task", description=f"task-{i}")
        await session.commit()

        timeline = await store.get_timeline(limit=3)
        assert len(timeline) == 3

    @pytest.mark.asyncio
    async def test_get_by_entity(self, session: AsyncSession):
        store = EventStore(session)
        await store.create(subject="alice", event_type="deployed", description="v1")
        await store.create(subject="bob", event_type="committed", description="fix")
        await store.create(subject="alice", event_type="migrated", description="db")
        await session.commit()

        results = await store.get_by_entity("alice")
        assert len(results) == 2
        assert all(r.content["subject"] == "alice" for r in results)
