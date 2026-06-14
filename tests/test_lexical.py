"""Tests for clara.retrieval.lexical — LexicalRetriever (no embeddings)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clara.db.models import Base, MemoryType
from clara.memory.belief import BeliefMemory, SourceType
from clara.memory.event import EventStore
from clara.retrieval.lexical import LexicalRetriever, tokenize


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Disable the LanceDB commit sync — this is a no-backend code path.
    factory = async_sessionmaker(
        engine, expire_on_commit=False, info={"_clara_disable_lance": True}
    )
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def _seed(session):
    beliefs = BeliefMemory(session)
    await beliefs.store(
        subject="user", relation="prefers", object_="Rust",
        domain="languages", source=SourceType.user_direct,
    )
    await beliefs.store(
        subject="user", relation="uses", object_="PostgreSQL",
        domain="databases", source=SourceType.user_direct,
    )
    await EventStore(session).create(
        subject="user", event_type="deployed",
        description="Deployed the api service to production",
    )
    await session.commit()


class TestTokenize:
    def test_drops_stopwords_and_short_tokens(self):
        assert tokenize("the user prefers a Rust toolchain") == [
            "user", "prefers", "rust", "toolchain"
        ]

    def test_empty(self):
        assert tokenize("   ") == []


class TestLexicalSearch:
    @pytest.mark.asyncio
    async def test_keyword_match_ranks_relevant_first(self, session):
        await _seed(session)
        result = await LexicalRetriever(session).search("rust", top_k=5)
        assert result.total >= 1
        top = result.all[0]
        assert top.memory.content["object"] == "Rust"
        assert top.similarity > 0.0

    @pytest.mark.asyncio
    async def test_non_matching_query_returns_nothing(self, session):
        await _seed(session)
        result = await LexicalRetriever(session).search("kubernetes helm chart")
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_type_filter(self, session):
        await _seed(session)
        result = await LexicalRetriever(session).search(
            "deployed api", memory_types=[MemoryType.event]
        )
        assert result.total == 1
        assert result.events
        assert not result.beliefs

    @pytest.mark.asyncio
    async def test_empty_query_falls_back_to_recency(self, session):
        await _seed(session)
        result = await LexicalRetriever(session).search("", top_k=10)
        # All three seeded memories come back, ranked by recency/confidence.
        assert result.total == 3

    @pytest.mark.asyncio
    async def test_empty_type_list_short_circuits(self, session):
        await _seed(session)
        result = await LexicalRetriever(session).search("rust", memory_types=[])
        assert result.total == 0
