from __future__ import annotations

import hashlib
import math
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.db.models import Base, VECTOR_DIMENSIONS
from clara.extraction.extractor import ExtractedFact
from clara.memory.belief import BeliefMemory, SourceType
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine, normalize_embedding_dimensions
from clara.retrieval.engine import RetrievalEngine
from clara.update.engine import MemoryUpdateEngine


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
async def session() -> AsyncSession:
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
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def embedder() -> EmbeddingEngine:
    return EmbeddingEngine(_FakeBackend())


class TestMemoryCache:
    @pytest.mark.asyncio
    async def test_invalidate_user_clears_user_and_global_namespaces(self):
        cache = MemoryCache("memory://")
        user_hash = cache.make_query_hash("rust", top_k=5)
        global_hash = cache.make_query_hash("all", top_k=5)

        await cache.set("alice", user_hash, [])
        await cache.set(None, global_hash, [])
        await cache.invalidate("alice")

        assert await cache.get("alice", user_hash) is None
        assert await cache.get(None, global_hash) is None


class TestRetrievalCacheIntegration:
    @pytest.mark.asyncio
    async def test_search_uses_cache_after_first_miss(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        beliefs = BeliefMemory(session)
        memory = await beliefs.store(
            subject="user",
            relation="uses",
            object_="Rust",
            domain="systems",
            source=SourceType.user_direct,
            raw_text="I use Rust.",
            user_id="alice",
            embedding=normalize_embedding_dimensions(
                embedder.embed("user uses Rust (systems)"),
                target_dimensions=VECTOR_DIMENSIONS,
            ),
        )
        await session.commit()

        cache = MemoryCache("memory://")
        retriever = RetrievalEngine(session, embedder, cache=cache)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            new=AsyncMock(return_value=[(memory, 0.95)]),
        ) as fetch:
            first = await retriever.search("What does the user use?", user_id="alice")

        assert first.total == 1
        assert fetch.await_count == 1

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            new=AsyncMock(side_effect=AssertionError("cache hit should bypass candidate fetch")),
        ):
            second = await retriever.search("What does the user use?", user_id="alice")

        assert second.total == 1
        assert second.beliefs[0].memory.memory_id == memory.memory_id

    @pytest.mark.asyncio
    async def test_update_engine_invalidates_cache(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        cache = MemoryCache("memory://")
        retriever = RetrievalEngine(session, embedder, cache=cache)
        updater = MemoryUpdateEngine(session, embedder, retriever, cache=cache)
        query_hash = cache.make_query_hash("What does the user use?", top_k=8)
        await cache.set("alice", query_hash, [])

        fact = ExtractedFact(
            subject="user",
            relation="uses",
            object="Rust",
            domain="systems",
            source_type="user_direct",
            confidence=0.9,
            is_negation=False,
            raw_text="I use Rust.",
        )

        with patch.object(RetrievalEngine, "_fetch_candidates", new=AsyncMock(return_value=[])):
            await updater.process(fact, user_id="alice")

        assert await cache.get("alice", query_hash) is None
