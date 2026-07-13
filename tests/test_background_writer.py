from __future__ import annotations

import hashlib
import math

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clara.db.models import Base, Memory
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.embeddings import EmbeddingEngine
from clara.update.background import BackgroundWriter


FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        magnitude = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / magnitude for value in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@pytest_asyncio.fixture
async def db_parts():

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield engine, factory
    await engine.dispose()


class TestBackgroundWriter:
    @pytest.mark.asyncio
    async def test_enqueue_processes_fact(self, db_parts):
        engine, factory = db_parts
        writer = BackgroundWriter(factory, EmbeddingEngine(_FakeBackend()))

        await writer.enqueue(
            ExtractedFact(
                subject="user",
                relation="uses",
                object="Rust",
                domain="systems",
                source_type="user_direct",
                confidence=0.9,
                is_negation=False,
                raw_text="I use Rust.",
            ),
            user_id="alice",
        )
        await writer.join()
        await writer.stop()

        async with factory() as session:
            rows = (await session.execute(select(Memory).where(Memory.user_id == "alice"))).scalars().all()

        assert rows
