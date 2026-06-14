"""
Concurrency/stress tests for retrieval. Marked @pytest.mark.stress so they are
excluded from the default run (see pyproject addopts -m 'not stress'); run with:

    pytest -m stress

They assert the retrieval + SQLite->LanceDB sync path survives many concurrent
recalls interleaved with writes, without crashing or returning corrupt results.
"""

from __future__ import annotations

import asyncio
import hashlib
import math

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from clara.agent import ClaraMemory, _make_engine
from clara.db.models import Base
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.embeddings import EmbeddingEngine

FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        mag = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / mag for v in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _SeqExtractor:
    """Emits one unique belief per call so each remember() writes a new memory."""

    def __init__(self) -> None:
        self._n = 0

    async def extract(self, text: str) -> list[ExtractedFact]:
        self._n += 1
        return [
            ExtractedFact(
                subject="user",
                relation="prefers",
                object=f"topic-{self._n}",
                domain=None,
                source_type="user_direct",
                confidence=0.9,
                is_negation=False,
                raw_text=text,
            )
        ]


@pytest_asyncio.fixture
async def agent(tmp_path):
    engine = _make_engine(f"sqlite+aiosqlite:///{(tmp_path / 'clara.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    instance = ClaraMemory(
        engine=engine,
        session_factory=factory,
        embedding_engine=EmbeddingEngine(_FakeBackend()),
        extractor=_SeqExtractor(),  # type: ignore[arg-type]
        decay_scheduler=None,
    )
    try:
        yield instance
    finally:
        await instance.close()
        await engine.dispose()


@pytest.mark.stress
class TestRetrievalConcurrency:
    @pytest.mark.asyncio
    async def test_many_concurrent_recalls(self, agent):
        for _ in range(20):
            await agent.remember("seed")
        # Search no longer flushes synchronously; drain pending writes so every
        # concurrent recall sees the seeded memories.
        agent._lance_engine.flush_pending_sync()

        results = await asyncio.gather(
            *[agent.recall("topic preference", top_k=5) for _ in range(40)]
        )
        assert all(r.total > 0 for r in results)

    @pytest.mark.asyncio
    async def test_recalls_interleaved_with_writes(self, agent):
        await agent.remember("seed")

        async def writer():
            for _ in range(15):
                await agent.remember("seed")

        async def reader():
            ok = 0
            for _ in range(15):
                result = await agent.recall("topic preference", top_k=5)
                ok += 1 if result.total >= 0 else 0
            return ok

        outcomes = await asyncio.gather(writer(), reader(), reader(), reader())
        # No exception raised; every reader completed all its recalls.
        assert outcomes[1] == outcomes[2] == outcomes[3] == 15
