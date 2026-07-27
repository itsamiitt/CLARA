from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.db.models import VECTOR_DIMENSIONS, Base
from clara.extraction.extractor import (
    ENV_OPENAI_KEY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    ExtractedFact,
)
from clara.memory.belief import BeliefMemory, SourceType
from clara.reasoning import ContextAssembler, ReasoningEngine
from clara.retrieval.embeddings import EmbeddingEngine, normalize_embedding_dimensions
from clara.retrieval.engine import LanceRetrievalEngine, RetrievalResult, ScoredMemory

FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        mag = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / mag for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _ResponseExtractor:
    async def extract(self, text: str) -> list[ExtractedFact]:
        if "Rust" not in text:
            return []
        return [
            ExtractedFact(
                subject="assistant",
                relation="mentioned",
                object="Rust",
                domain=None,
                source_type="system",
                confidence=0.8,
                is_negation=False,
                raw_text=text,
            )
        ]


@pytest_asyncio.fixture
async def session() -> AsyncSession:

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


class TestContextAssembler:
    @pytest.mark.asyncio
    async def test_build_formats_retrieval_result(self):
        retriever = AsyncMock()
        mem = AsyncMock()
        mem.content = {
            "subject": "user",
            "relation": "uses",
            "object": "Rust",
            "domain": "systems",
        }
        mem.confidence = 0.9
        mem.created_at = datetime(2026, 3, 11, tzinfo=timezone.utc)
        mem.memory_type = "belief"
        scored = ScoredMemory(
            memory=mem,
            score=0.9,
            similarity=0.95,
            confidence=0.9,
            recency_score=1.0,
            usage_frequency=0.0,
        )
        retriever.search.return_value = RetrievalResult(beliefs=[scored])

        assembler = ContextAssembler(retriever)
        context = await assembler.build("systems", user_id="alice", top_k=4)

        assert "=== MEMORY CONTEXT ===" in context
        assert "user uses Rust" in context
        assert retriever.search.await_args.kwargs["user_id"] == "alice"


class TestReasoningEngine:
    @pytest.mark.asyncio
    async def test_respond_builds_context_and_stores_response_facts(self, session: AsyncSession):
        embedder = EmbeddingEngine(_FakeBackend())
        beliefs = BeliefMemory(session)
        await beliefs.store(
            subject="user",
            relation="uses",
            object_="Rust",
            domain="systems",
            source=SourceType.user_direct,
            raw_text="I use Rust for systems work.",
            user_id="alice",
            embedding=normalize_embedding_dimensions(
                embedder.embed("user uses Rust (systems)"),
                target_dimensions=VECTOR_DIMENSIONS,
            ),
        )
        await session.commit()
        # Search no longer flushes synchronously; drain the pending vector write
        # so the just-stored belief is searchable during respond().
        LanceRetrievalEngine.get_default().flush_pending_sync()

        async def _respond(system_prompt: str, query: str, model: str) -> str:
            return "The user uses Rust for systems work."

        engine = ReasoningEngine(
            session,
            embedder,
            _ResponseExtractor(),  # type: ignore[arg-type]
            llm_provider="openai",
            response_generator=_respond,
        )

        response = await engine.respond(
            "What language does the user use?",
            user_id="alice",
            top_k=4,
        )

        assert "Rust" in response.text
        assert "=== MEMORY CONTEXT ===" in response.memory_context
        assert response.memories_used
        assert len(response.facts_stored) == 1
        assert response.facts_stored[0].memory_id is not None

    @pytest.mark.asyncio
    async def test_generate_response_awaits_provider_call(self, session: AsyncSession):
        embedder = EmbeddingEngine(_FakeBackend())
        engine = ReasoningEngine(
            session,
            embedder,
            _ResponseExtractor(),  # type: ignore[arg-type]
            llm_provider="openai",
        )
        engine._call_openai = AsyncMock(return_value="Async reply")  # type: ignore[method-assign]

        response = await engine._generate_response("What does the user use?", "context block")

        assert response == "Async reply"
        engine._call_openai.assert_awaited_once()
        system_prompt, query = engine._call_openai.await_args.args
        assert query == "What does the user use?"
        assert "context block" in system_prompt

    @pytest.mark.asyncio
    async def test_call_openai_uses_async_client(self, session: AsyncSession):
        embedder = EmbeddingEngine(_FakeBackend())
        engine = ReasoningEngine(
            session,
            embedder,
            _ResponseExtractor(),  # type: ignore[arg-type]
            llm_provider="openai",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Async answer"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client
        mock_openai_module.OpenAI.side_effect = AssertionError("sync client should not be used")

        with (
            patch.dict("os.environ", {ENV_OPENAI_KEY: "sk-test"}),
            patch("clara.reasoning.engine._openai", mock_openai_module),
        ):
            response = await engine._call_openai("system prompt", "hello")

        assert response == "Async answer"
        mock_openai_module.AsyncOpenAI.assert_called_once_with(
            api_key="sk-test", timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
        )
        mock_client.chat.completions.create.assert_awaited_once()
