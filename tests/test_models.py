from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clara.db.models import Base, Memory, MemoryStatus, MemoryType


@pytest_asyncio.fixture
async def model_session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_cache_clears_on_expire(model_session):
    memory = Memory(
        memory_type=MemoryType.belief,
        user_id="alice",
        content={"subject": "user", "relation": "uses", "object": "Rust"},
        embedding=[0.1, 0.2, 0.3],
        confidence=0.9,
        status=MemoryStatus.active,
        decay_rate=0.02,
        metadata_={},
    )
    model_session.add(memory)
    await model_session.flush()

    assert memory.embedding == [0.1, 0.2, 0.3]

    model_session.expire(memory)

    assert memory.embedding is None
