"""Tests for clara.memory.world_model — WorldModelStore."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.core.exceptions import MemoryNotFoundError
from clara.db.models import Base, MemoryStatus, MemoryType
from clara.memory.world_model import WorldModelStore


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
# Upsert tests — create path
# ---------------------------------------------------------------------------

class TestWorldModelCreate:
    @pytest.mark.asyncio
    async def test_upsert_creates_new_entity(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16", "role": "primary_db"},
            user_id="alice",
        )
        await session.commit()

        assert record.memory_type == MemoryType.world_model
        assert record.user_id == "alice"
        assert record.status == MemoryStatus.active
        assert record.confidence == 0.9
        assert record.content["entity_type"] == "tool"
        assert record.content["name"] == "PostgreSQL"
        assert record.content["properties"]["version"] == "16"
        assert record.content["properties"]["role"] == "primary_db"

    @pytest.mark.asyncio
    async def test_upsert_with_domain(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="Redis",
            domain="infrastructure",
        )
        await session.commit()

        assert record.content["domain"] == "infrastructure"

    @pytest.mark.asyncio
    async def test_upsert_empty_properties(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="person",
            name="Alice",
        )
        await session.commit()

        assert record.content["properties"] == {}

    @pytest.mark.asyncio
    async def test_create_has_empty_mutation_history(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="Node.js",
            properties={"version": "20"},
        )
        await session.commit()

        meta = record.metadata_
        assert meta["mutation_history"] == []


# ---------------------------------------------------------------------------
# Upsert tests — merge path
# ---------------------------------------------------------------------------

class TestWorldModelMerge:
    @pytest.mark.asyncio
    async def test_upsert_merges_properties(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16", "role": "primary_db"},
            user_id="alice",
        )
        await session.commit()

        # Upsert again with updated properties
        updated = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "17", "replicas": 3},
            user_id="alice",
        )
        await session.commit()

        # Same record, merged properties
        assert updated.memory_id == record.memory_id
        props = updated.content["properties"]
        assert props["version"] == "17"         # updated
        assert props["role"] == "primary_db"    # preserved
        assert props["replicas"] == 3           # added

    @pytest.mark.asyncio
    async def test_merge_logs_mutations(self, session: AsyncSession):
        store = WorldModelStore(session)
        await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16"},
            user_id="alice",
        )
        await session.commit()

        await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "17"},
            user_id="alice",
        )
        await session.commit()

        # Fetch and check mutation history
        record = await store.get_by_name("PostgreSQL", user_id="alice")
        assert record is not None
        meta = record.metadata_
        history = meta["mutation_history"]
        assert len(history) == 1
        assert history[0]["key"] == "version"
        assert history[0]["old_value"] == "16"
        assert history[0]["new_value"] == "17"

    @pytest.mark.asyncio
    async def test_merge_does_not_log_unchanged_properties(self, session: AsyncSession):
        store = WorldModelStore(session)
        await store.upsert(
            entity_type="tool",
            name="Redis",
            properties={"version": "7", "role": "cache"},
        )
        await session.commit()

        # Upsert with same values
        await store.upsert(
            entity_type="tool",
            name="Redis",
            properties={"version": "7"},  # unchanged
        )
        await session.commit()

        record = await store.get_by_name("Redis")
        meta = record.metadata_
        assert len(meta["mutation_history"]) == 0

    @pytest.mark.asyncio
    async def test_upsert_different_users_creates_separate_records(self, session: AsyncSession):
        store = WorldModelStore(session)
        rec_a = await store.upsert(
            entity_type="tool", name="PostgreSQL",
            properties={"version": "16"}, user_id="alice",
        )
        rec_b = await store.upsert(
            entity_type="tool", name="PostgreSQL",
            properties={"version": "15"}, user_id="bob",
        )
        await session.commit()

        assert rec_a.memory_id != rec_b.memory_id
        assert rec_a.content["properties"]["version"] == "16"
        assert rec_b.content["properties"]["version"] == "15"

    @pytest.mark.asyncio
    async def test_upsert_recovers_from_raced_insert(self, session: AsyncSession, monkeypatch):
        store = WorldModelStore(session)
        original = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16"},
            user_id="alice",
        )
        await session.commit()

        original_find_existing = store._find_existing
        calls = {"count": 0}

        async def _stale_then_real(entity_type, name, *, user_id=None):
            calls["count"] += 1
            if calls["count"] == 1:
                return None
            return await original_find_existing(entity_type, name, user_id=user_id)

        monkeypatch.setattr(store, "_find_existing", _stale_then_real)

        updated = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "17", "replicas": 2},
            user_id="alice",
        )
        await session.commit()

        assert updated.memory_id == original.memory_id
        assert updated.content["properties"]["version"] == "17"
        assert updated.content["properties"]["replicas"] == 2


# ---------------------------------------------------------------------------
# update_property tests
# ---------------------------------------------------------------------------

class TestUpdateProperty:
    @pytest.mark.asyncio
    async def test_update_single_property(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16", "role": "primary_db"},
        )
        await session.commit()

        updated = await store.update_property(
            record.memory_id, "version", "17",
        )
        await session.commit()

        assert updated.content["properties"]["version"] == "17"
        assert updated.content["properties"]["role"] == "primary_db"

    @pytest.mark.asyncio
    async def test_update_property_logs_mutation(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="Node.js",
            properties={"version": "18"},
        )
        await session.commit()

        await store.update_property(record.memory_id, "version", "20")
        await session.commit()

        history = await store.get_history(record.memory_id)
        assert len(history) == 1
        assert history[0]["key"] == "version"
        assert history[0]["old_value"] == "18"
        assert history[0]["new_value"] == "20"

    @pytest.mark.asyncio
    async def test_update_property_adds_new_key(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool",
            name="Redis",
            properties={"version": "7"},
        )
        await session.commit()

        updated = await store.update_property(record.memory_id, "max_memory", "4GB")
        await session.commit()

        assert updated.content["properties"]["max_memory"] == "4GB"
        history = await store.get_history(record.memory_id)
        assert history[0]["old_value"] is None  # new key
        assert history[0]["new_value"] == "4GB"

    @pytest.mark.asyncio
    async def test_update_property_nonexistent_raises(self, session: AsyncSession):
        store = WorldModelStore(session)
        with pytest.raises(MemoryNotFoundError):
            await store.update_property(uuid.uuid4(), "key", "value")


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestWorldModelQueries:
    @pytest.mark.asyncio
    async def test_get_returns_record(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(entity_type="tool", name="PostgreSQL")
        await session.commit()

        fetched = await store.get(record.memory_id)
        assert fetched is not None
        assert fetched.memory_id == record.memory_id

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, session: AsyncSession):
        store = WorldModelStore(session)
        assert await store.get(uuid.uuid4()) is None

    @pytest.mark.asyncio
    async def test_get_state_filters_by_user(self, session: AsyncSession):
        store = WorldModelStore(session)
        await store.upsert(entity_type="tool", name="Redis", user_id="alice")
        await store.upsert(entity_type="tool", name="Postgres", user_id="alice")
        await store.upsert(entity_type="tool", name="MySQL", user_id="bob")
        await session.commit()

        state = await store.get_state(user_id="alice")
        assert len(state) == 2
        assert all(s.user_id == "alice" for s in state)

    @pytest.mark.asyncio
    async def test_get_state_filters_by_entity_type(self, session: AsyncSession):
        store = WorldModelStore(session)
        await store.upsert(entity_type="tool", name="Redis")
        await store.upsert(entity_type="person", name="Alice")
        await session.commit()

        tools = await store.get_state(entity_type="tool")
        assert len(tools) == 1
        assert tools[0].content["name"] == "Redis"

    @pytest.mark.asyncio
    async def test_get_by_name(self, session: AsyncSession):
        store = WorldModelStore(session)
        await store.upsert(entity_type="tool", name="Redis", user_id="alice")
        await session.commit()

        record = await store.get_by_name("Redis", user_id="alice")
        assert record is not None
        assert record.content["name"] == "Redis"

    @pytest.mark.asyncio
    async def test_get_by_name_returns_none(self, session: AsyncSession):
        store = WorldModelStore(session)
        result = await store.get_by_name("NonExistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_history(self, session: AsyncSession):
        store = WorldModelStore(session)
        record = await store.upsert(
            entity_type="tool", name="Node.js",
            properties={"version": "18"},
        )
        await store.update_property(record.memory_id, "version", "20")
        await store.update_property(record.memory_id, "version", "22")
        await session.commit()

        history = await store.get_history(record.memory_id)
        assert len(history) == 2
        assert history[0]["new_value"] == "20"
        assert history[1]["new_value"] == "22"

    @pytest.mark.asyncio
    async def test_get_history_nonexistent_raises(self, session: AsyncSession):
        store = WorldModelStore(session)
        with pytest.raises(MemoryNotFoundError):
            await store.get_history(uuid.uuid4())
