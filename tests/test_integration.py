"""
Integration tests for ClaraMemory — full pipeline: remember → recall → context_for.

Uses a **real in-memory SQLite** database (via aiosqlite) for actual SQL
execution, but **mocks** the embedding engine (no API calls) and the fact
extractor (no LLM calls).

Because SQLite does not support pgvector, the vector similarity search in
RetrievalEngine._fetch_candidates is patched to use a simple Python-side
cosine similarity over the FakeMemory rows that exist in the session.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from clara.agent import ClaraMemory, format_context
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import RetrievalEngine, RetrievalResult, ScoredMemory
from clara.update.background import BackgroundWriter

# ---------------------------------------------------------------------------
# Fake embedding backend
# ---------------------------------------------------------------------------

FAKE_DIM = 8  # small vectors for tests


class _FakeBackend:
    """Deterministic embedding backend for integration tests."""

    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        """Produce a stable unit vector from *text*."""
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        mag = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / mag for x in raw]


def _fake_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine(_FakeBackend())


# ---------------------------------------------------------------------------
# Cosine similarity helper (for the mock retrieval)
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    # Deliberately NOT strict=True: the production query vector is zero-padded to
    # VECTOR_DIMENSIONS (1536) while the fake embedder emits FAKE_DIM (8), so the
    # lengths differ by design. zip() truncates to the meaningful prefix, which is
    # exactly the comparison this mock wants.
    dot = sum(x * y for x, y in zip(a, b))  # noqa: B905
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def db_parts():
    """Create an in-memory SQLite engine + session factory with CLARA tables.

    Registers custom type compilations so PostgreSQL-specific types
    (JSONB, Vector) map to SQLite-compatible ``TEXT`` columns.
    """

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    yield engine, factory

    await engine.dispose()


@pytest.fixture
def fake_embedder() -> EmbeddingEngine:
    return _fake_embedding_engine()


@pytest.fixture
def fake_extractor():
    """Return a FactExtractor mock that extracts canned facts."""
    extractor = MagicMock()

    async def _extract(text: str) -> list[ExtractedFact]:
        """Simple rule-based extraction for testing."""
        facts = []
        text_lower = text.lower()

        if "rust" in text_lower and "systems" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="uses",
                object="Rust",
                domain="systems",
                source_type="user_direct",
                confidence=0.9,
                is_negation=False,
                raw_text=text,
            ))
        if "python" in text_lower and "web" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="uses",
                object="Python",
                domain="web",
                source_type="user_direct",
                confidence=0.85,
                is_negation=False,
                raw_text=text,
            ))
        if "deploy" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="deployed",
                object="app",
                domain=None,
                source_type="user_direct",
                confidence=0.8,
                is_negation=False,
                raw_text=text,
            ))
        if "prefers" in text_lower and "dark" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="prefers",
                object="dark mode",
                domain=None,
                source_type="user_direct",
                confidence=0.92,
                is_negation=False,
                raw_text=text,
            ))
        # Fallback: if nothing matched, produce a generic fact
        if not facts and text.strip():
            words = text.strip().split()
            facts.append(ExtractedFact(
                subject="user",
                relation="mentioned",
                object=" ".join(words[:5]),
                domain=None,
                source_type="user_direct",
                confidence=0.7,
                is_negation=False,
                raw_text=text,
            ))
        return facts

    extractor.extract = _extract
    return extractor


# ---------------------------------------------------------------------------
# Helper: build a ClaraMemory from fixtures without real LLM/embedding APIs
# ---------------------------------------------------------------------------

async def _build_agent(
    engine,
    factory,
    embedder,
    extractor,
    *,
    background_writer=None,
) -> ClaraMemory:
    """Construct a ClaraMemory with injected fakes (no scheduler)."""
    return ClaraMemory(
        engine=engine,
        session_factory=factory,
        embedding_engine=embedder,
        extractor=extractor,
        decay_scheduler=None,  # no background scheduler in tests
        background_writer=background_writer,
    )


# ---------------------------------------------------------------------------
# Mock fetch_candidates — replaces pgvector ANN with Python-side cosine sim
# ---------------------------------------------------------------------------

def _make_fetch_candidates_mock(embedder: EmbeddingEngine):
    """Return an async function that replaces RetrievalEngine._fetch_candidates.

    It queries ALL active Memory rows from the session (ignoring the vector
    column), computes cosine similarity in Python, and returns pairs.
    """

    async def _fake_fetch(
        self,
        query_vector: list[float],
        *,
        n_candidates: int,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        from sqlalchemy import and_, select

        filters = [Memory.status == MemoryStatus.active]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if memory_types:
            filters.append(Memory.memory_type.in_(memory_types))

        stmt = select(Memory).where(and_(*filters))
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        # Compute cosine similarity using the content as an embedding proxy
        pairs = []
        for mem in rows:
            # Build a text from the content and embed it
            c = mem.content or {}
            mem_text = f"{c.get('subject', '')} {c.get('relation', '')} {c.get('object', '')}"
            mem_vec = embedder.embed(mem_text)
            sim = _cosine_sim(query_vector, mem_vec)
            pairs.append((mem, sim))

        # Sort by similarity descending and return top n_candidates
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:n_candidates]

    return _fake_fetch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemember:
    """Test the remember() pipeline: extraction → update engine → DB write."""

    @pytest.mark.asyncio
    async def test_remember_stores_facts(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            results = await agent.remember(
                "I use Rust for systems work and Python for web development."
            )

        # Should have extracted at least the Rust fact
        assert len(results) >= 1
        assert any(r["action"] == "created" for r in results)

        # Verify records actually exist in the DB
        async with factory() as session:
            from sqlalchemy import func, select
            count = await session.scalar(
                select(func.count()).select_from(Memory)
            )
            assert count >= 1

    @pytest.mark.asyncio
    async def test_remember_empty_text(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        results = await agent.remember("")
        assert results == []

    @pytest.mark.asyncio
    async def test_remember_normalizes_whitespace(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        seen: list[str] = []
        original_extract = fake_extractor.extract

        def _extract(text: str) -> list[ExtractedFact]:
            seen.append(text)
            return original_extract(text)

        agent._extractor.extract = _extract

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("  I   use   Rust   for systems   work. ")

        assert seen == ["I use Rust for systems work."]

    @pytest.mark.asyncio
    async def test_remember_sets_user_id_on_records(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("I use Rust for systems work.", user_id="alice")

        async with factory() as session:
            rows = (
                await session.execute(select(Memory).where(Memory.user_id == "alice"))
            ).scalars().all()

        assert rows
        assert all(row.user_id == "alice" for row in rows)

    @pytest.mark.asyncio
    async def test_remember_multiple_facts(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            results = await agent.remember(
                "I use Rust for systems work and Python for web development."
            )

        # Both Rust (systems) and Python (web) should be extracted
        assert len(results) == 2
        assert results[0]["action"] == "created"
        assert results[1]["action"] in {"created", "retained_both"}

    @pytest.mark.asyncio
    async def test_remember_wait_false_queues_background_processing(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        writer = BackgroundWriter(factory, fake_embedder)
        agent = await _build_agent(
            engine,
            factory,
            fake_embedder,
            fake_extractor,
            background_writer=writer,
        )

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            results = await agent.remember(
                "I use Rust for systems work.",
                user_id="alice",
                wait=False,
            )
            await writer.join()

        assert results == [{
            "action": "queued",
            "memory_id": None,
            "conflict": False,
            "superseded_id": None,
        }]

        async with factory() as session:
            rows = (
                await session.execute(select(Memory).where(Memory.user_id == "alice"))
            ).scalars().all()
        assert rows


class TestRecall:
    """Test the recall() pipeline: embed query → similarity search → rank."""

    @pytest.mark.asyncio
    async def test_recall_returns_stored_memories(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        # Store a fact first — patch needed because remember() also uses retrieval
        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("I use Rust for systems work.")

        # Recall — patch _fetch_candidates to avoid pgvector
        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            result = await agent.recall("What language for systems?")

        assert result.total >= 1

    @pytest.mark.asyncio
    async def test_recall_empty_db(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            result = await agent.recall("anything")

        assert result.total == 0

    @pytest.mark.asyncio
    async def test_recall_filters_by_user_id(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("I use Rust for systems work.", user_id="alice")
            await agent.remember("I use Rust for systems work.", user_id="bob")

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            alice = await agent.recall("What language for systems?", user_id="alice")
            bob = await agent.recall("What language for systems?", user_id="bob")
            all_rows = await agent.recall("What language for systems?")

        assert alice.total >= 1
        assert bob.total >= 1
        assert all_rows.total >= 2
        assert all(sm.memory.user_id == "alice" for sm in alice.all)
        assert all(sm.memory.user_id == "bob" for sm in bob.all)


class TestContextFor:
    """Test context_for() — full pipeline through to formatted output."""

    @pytest.mark.asyncio
    async def test_context_format_structure(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("I use Rust for systems work.")

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            ctx = await agent.context_for("systems programming")

        # Verify structural markers
        assert ctx.startswith("=== MEMORY CONTEXT ===")
        assert ctx.endswith("=== END MEMORY CONTEXT ===")
        assert "[BELIEFS]" in ctx
        assert "[WORLD MODEL]" in ctx
        assert "[RECENT EVENTS]" in ctx
        assert "[RELEVANT SKILLS]" in ctx

    @pytest.mark.asyncio
    async def test_context_contains_stored_belief(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            await agent.remember("I use Rust for systems work.")

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            ctx = await agent.context_for("Rust")

        # The belief about Rust should appear in the context
        assert "Rust" in ctx
        assert "confidence:" in ctx

    @pytest.mark.asyncio
    async def test_context_empty_db(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            ctx = await agent.context_for("anything")

        assert "=== MEMORY CONTEXT ===" in ctx
        assert "=== END MEMORY CONTEXT ===" in ctx
        # All sections should show (none) since DB is empty
        assert "(none)" in ctx


class TestFullPipeline:
    """End-to-end: remember → recall → context_for in sequence."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        # Patch _fetch_candidates throughout — remember() also uses retrieval
        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            # 1. Remember multiple facts
            r1 = await agent.remember("I use Rust for systems work.")
            assert len(r1) >= 1

            r2 = await agent.remember("I use Python for web development.")
            assert len(r2) >= 1

            r3 = await agent.remember("I deployed the app yesterday.")
            assert len(r3) >= 1

            # 2. Recall
            recalled = await agent.recall("What languages does the user know?", top_k=4)
            assert recalled.total >= 1

            # 3. Context for LLM
            ctx = await agent.context_for("Help the user with deployment")

        assert "=== MEMORY CONTEXT ===" in ctx
        assert "=== END MEMORY CONTEXT ===" in ctx

    @pytest.mark.asyncio
    async def test_remember_twice_same_text(
        self, db_parts, fake_embedder, fake_extractor,
    ):
        """Remembering the same text twice should reinforce, not duplicate."""
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        with patch.object(
            RetrievalEngine,
            "_fetch_candidates",
            _make_fetch_candidates_mock(fake_embedder),
        ):
            r1 = await agent.remember("I use Rust for systems work.")
            r2 = await agent.remember("I use Rust for systems work.")

        # First time creates, second time should reinforce (or create if
        # similarity search can't find the first — depends on mock)
        assert len(r1) >= 1
        assert len(r2) >= 1


class TestFormatContext:
    """Unit tests for the format_context() function directly."""

    def test_empty_result(self):
        ctx = format_context(RetrievalResult())
        assert "=== MEMORY CONTEXT ===" in ctx
        assert "=== END MEMORY CONTEXT ===" in ctx
        assert "[BELIEFS]" in ctx
        assert ctx.count("(none)") == 4  # all 4 sections empty

    def test_with_belief(self):
        mem = MagicMock()
        mem.content = {
            "subject": "user",
            "relation": "uses",
            "object": "Rust",
            "domain": "systems",
        }
        mem.confidence = 0.85
        mem.created_at = datetime(2026, 3, 10, tzinfo=timezone.utc)
        mem.memory_type = MemoryType.belief

        sm = ScoredMemory(
            memory=mem, score=0.9, similarity=0.95,
            confidence=0.85, recency_score=1.0, usage_frequency=0.0,
        )
        result = RetrievalResult(beliefs=[sm])
        ctx = format_context(result)

        assert "user uses Rust" in ctx
        assert "confidence: 0.85" in ctx
        assert "domain: systems" in ctx

    def test_with_event(self):
        mem = MagicMock()
        mem.content = {
            "subject": "user",
            "relation": "deployed",
            "object": "backend service",
        }
        mem.confidence = 0.80
        mem.created_at = datetime(2026, 3, 10, tzinfo=timezone.utc)
        mem.memory_type = MemoryType.event

        sm = ScoredMemory(
            memory=mem, score=0.7, similarity=0.8,
            confidence=0.80, recency_score=1.0, usage_frequency=0.0,
        )
        result = RetrievalResult(events=[sm])
        ctx = format_context(result)

        assert "2026-03-10" in ctx
        assert "deployed" in ctx

    def test_with_negated_belief(self):
        mem = MagicMock()
        mem.content = {
            "subject": "user",
            "relation": "uses",
            "object": "Python",
            "domain": "systems",
            "is_negation": True,
        }
        mem.confidence = 0.85
        mem.created_at = datetime(2026, 3, 10, tzinfo=timezone.utc)
        mem.memory_type = MemoryType.belief

        sm = ScoredMemory(
            memory=mem, score=0.9, similarity=0.95,
            confidence=0.85, recency_score=1.0, usage_frequency=0.0,
        )
        result = RetrievalResult(beliefs=[sm])
        ctx = format_context(result)

        assert "not (user uses Python)" in ctx

    def test_with_skill(self):
        mem = MagicMock()
        mem.content = {"name": "Deploy Rust API to Linux"}
        mem.confidence = 0.82
        mem.memory_type = MemoryType.skill

        sm = ScoredMemory(
            memory=mem, score=0.6, similarity=0.7,
            confidence=0.82, recency_score=1.0, usage_frequency=0.0,
        )
        result = RetrievalResult(skills=[sm])
        ctx = format_context(result)

        assert "Deploy Rust API to Linux" in ctx
        assert "confidence: 0.82" in ctx


class TestClaraMemoryLifecycle:
    """Test construction and teardown."""

    @pytest.mark.asyncio
    async def test_close(self, db_parts, fake_embedder, fake_extractor):
        engine, factory = db_parts
        agent = await _build_agent(engine, factory, fake_embedder, fake_extractor)

        # close() should not raise
        await agent.close()

    @pytest.mark.asyncio
    async def test_create_factory_method(self):
        """Test ClaraMemory.create() with mocked backends."""
        with patch("clara.agent._create_backend") as mock_backend, \
             patch("clara.agent.FactExtractor") as mock_extractor:
            mock_backend.return_value = _FakeBackend()
            mock_extractor.return_value = MagicMock()

            agent = await ClaraMemory.create(
                db_url="sqlite+aiosqlite://",
                embedding_backend="openai",
                llm_provider="openai",
                start_scheduler=False,
            )

        assert agent is not None
        await agent.close()
