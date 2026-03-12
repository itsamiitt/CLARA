from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ENV_ANTHROPIC_KEY
from clara.reflection import ReflectionEngine
from clara.reflection.pipeline import PatternCandidate, detect_patterns
from clara.retrieval.embeddings import EmbeddingEngine
from clara.scheduler.decay import DecayScheduler
from clara.update.engine import ActionTaken, UpdateResult


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


async def _seed_memory(
    session: AsyncSession,
    *,
    memory_type: MemoryType,
    subject: str,
    relation: str,
    object_: str,
    user_id: str | None,
    days_ago: int = 0,
    confidence: float = 0.8,
) -> Memory:
    now = datetime.now(timezone.utc) - timedelta(days=days_ago)
    record = Memory(
        memory_type=memory_type,
        user_id=user_id,
        content={
            "subject": subject,
            "relation": relation,
            "object": object_,
        },
        embedding=None,
        confidence=confidence,
        status=MemoryStatus.active,
        decay_rate=0.0 if memory_type == MemoryType.event else 0.02,
        created_at=now,
        updated_at=now,
        metadata_={},
    )
    session.add(record)
    await session.flush()
    return record


class TestDetectPatterns:
    def test_detects_recurring_subjects_events_and_skills(self):
        now = datetime.now(timezone.utc)
        memories = [
            Memory(
                memory_type=MemoryType.event,
                user_id="alice",
                content={"subject": "user", "relation": "deployed", "object": "service"},
                embedding=None,
                confidence=0.8,
                status=MemoryStatus.active,
                decay_rate=0.0,
                created_at=now,
                updated_at=now,
                metadata_={},
            )
            for _ in range(3)
        ]
        memories.extend([
            Memory(
                memory_type=MemoryType.skill,
                user_id="alice",
                content={"subject": "user", "relation": "knows", "object": "Kubernetes"},
                embedding=None,
                confidence=0.8,
                status=MemoryStatus.active,
                decay_rate=0.01,
                created_at=now,
                updated_at=now,
                metadata_={},
            )
            for _ in range(3)
        ])

        patterns = detect_patterns(memories)
        keys = {(pattern.pattern_type, pattern.relation, pattern.object) for pattern in patterns}

        assert ("recurring_entity", "appears_frequently_in", "recent_activity") in keys
        assert ("repeated_event_type", "repeatedly_deployed", "service") in keys
        assert ("skill_generalization", "is_consistently_skilled_at", "Kubernetes") in keys


class TestReflectionEngine:
    @pytest.mark.asyncio
    async def test_run_stores_agent_reflection_beliefs(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        for days_ago in (1, 2, 3):
            await _seed_memory(
                session,
                memory_type=MemoryType.event,
                subject="user",
                relation="deployed",
                object_="service",
                user_id="alice",
                days_ago=days_ago,
            )

        async def _insight(system_prompt: str, prompt: str, model: str) -> str:
            assert "repeatedly_deployed" in prompt or "appears_frequently_in" in prompt
            return "Generated reflection insight."

        engine = ReflectionEngine(
            session,
            embedder,
            llm_provider="openai",
            insight_generator=_insight,
        )

        results = await engine.run(user_id="alice")

        assert len(results) == 2
        assert all(result.action_taken == ActionTaken.created for result in results)

        rows = (
            await session.execute(
                select(Memory)
                .where(Memory.memory_type == MemoryType.belief)
                .where(Memory.user_id == "alice")
            )
        ).scalars().all()

        assert len(rows) == 2
        assert all(row.confidence == pytest.approx(0.5, abs=1e-6) for row in rows)
        assert all(
            (row.metadata_ or {}).get("evidence", [{}])[0].get("source") == "agent_reflection"
            for row in rows
        )
        assert all(
            (row.metadata_ or {}).get("evidence", [{}])[0].get("text") == "Generated reflection insight."
            for row in rows
        )

    @pytest.mark.asyncio
    async def test_run_is_tenant_scoped(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        for user_id in ("alice", "bob"):
            for days_ago in (1, 2, 3):
                await _seed_memory(
                    session,
                    memory_type=MemoryType.event,
                    subject="user",
                    relation="deployed",
                    object_="service",
                    user_id=user_id,
                    days_ago=days_ago,
                )

        engine = ReflectionEngine(
            session,
            embedder,
            llm_provider="openai",
            insight_generator=AsyncMock(return_value="Tenant reflection."),
        )

        await engine.run(user_id="alice")

        reflected = (
            await session.execute(
                select(Memory).where(Memory.memory_type == MemoryType.belief)
            )
        ).scalars().all()
        reflected = [
            row for row in reflected
            if row.content.get("relation") in {"appears_frequently_in", "repeatedly_deployed"}
        ]

        assert reflected
        assert all(row.user_id == "alice" for row in reflected)

    @pytest.mark.asyncio
    async def test_second_run_reinforces_existing_reflection_beliefs(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        for days_ago in (1, 2, 3):
            await _seed_memory(
                session,
                memory_type=MemoryType.event,
                subject="user",
                relation="deployed",
                object_="service",
                user_id="alice",
                days_ago=days_ago,
            )

        engine = ReflectionEngine(
            session,
            embedder,
            llm_provider="openai",
            insight_generator=AsyncMock(return_value="Repeated deployment pattern."),
        )

        first = await engine.run(user_id="alice")
        second = await engine.run(user_id="alice")

        assert all(result.action_taken == ActionTaken.created for result in first)
        assert all(result.action_taken == ActionTaken.reinforced for result in second)

    @pytest.mark.asyncio
    async def test_generate_insight_awaits_provider_call(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        engine = ReflectionEngine(
            session,
            embedder,
            llm_provider="anthropic",
        )
        engine._call_anthropic = AsyncMock(return_value="Async insight")  # type: ignore[method-assign]
        pattern = PatternCandidate(
            subject="user",
            relation="repeatedly_deployed",
            object="service",
            pattern_type="repeated_event_type",
            count=3,
        )

        insight = await engine._generate_insight(pattern, [])

        assert insight == "Async insight"
        engine._call_anthropic.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_anthropic_uses_async_client(
        self,
        session: AsyncSession,
        embedder: EmbeddingEngine,
    ):
        engine = ReflectionEngine(
            session,
            embedder,
            llm_provider="anthropic",
        )
        pattern = PatternCandidate(
            subject="user",
            relation="repeatedly_deployed",
            object="service",
            pattern_type="repeated_event_type",
            count=3,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Async reflection")]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.AsyncAnthropic.return_value = mock_client
        mock_anthropic_module.Anthropic.side_effect = AssertionError(
            "sync client should not be used"
        )

        with patch.dict("os.environ", {ENV_ANTHROPIC_KEY: "test-key"}):
            with patch("clara.reflection.pipeline._anthropic", mock_anthropic_module):
                insight = await engine._call_anthropic("prompt", pattern)

        assert insight == "Async reflection"
        mock_anthropic_module.AsyncAnthropic.assert_called_once_with(api_key="test-key")
        mock_client.messages.create.assert_awaited_once()


class TestReflectionScheduler:
    def test_start_registers_reflection_job_when_configured(self):
        scheduler = DecayScheduler(
            MagicMock(),
            embedding_engine=EmbeddingEngine(_FakeBackend()),
            llm_provider="openai",
        )

        with patch.object(scheduler._scheduler, "add_job") as add_job, \
             patch.object(scheduler._scheduler, "start"):
            scheduler.start()

        job_ids = {call.kwargs["id"] for call in add_job.call_args_list}
        assert {"daily_decay", "weekly_pruning", "daily_reflection"} <= job_ids

    @pytest.mark.asyncio
    async def test_run_daily_reflection_processes_each_tenant_once(self):
        session = AsyncMock()
        begin_ctx = AsyncMock()
        begin_ctx.__aenter__ = AsyncMock(return_value=None)
        begin_ctx.__aexit__ = AsyncMock(return_value=False)
        session.begin = MagicMock(return_value=begin_ctx)

        result = MagicMock()
        result.scalars.return_value.all.return_value = ["alice", "bob", None]
        session.execute = AsyncMock(return_value=result)

        factory = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        factory.return_value = ctx

        scheduler = DecayScheduler(
            factory,
            embedding_engine=EmbeddingEngine(_FakeBackend()),
            llm_provider="openai",
        )

        with patch("clara.scheduler.decay.ReflectionEngine") as reflection_cls:
            reflection = reflection_cls.return_value
            reflection.run = AsyncMock(side_effect=[
                [UpdateResult(ActionTaken.created, None, False)],
                [
                    UpdateResult(ActionTaken.created, None, False),
                    UpdateResult(ActionTaken.reinforced, None, False),
                ],
            ])

            summary = await scheduler.run_daily_reflection()

        assert summary == {"users_processed": 2, "insights_generated": 3}
        assert [call.kwargs["user_id"] for call in reflection.run.await_args_list] == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_run_daily_reflection_uses_legacy_partition_only_when_needed(self):
        session = AsyncMock()
        begin_ctx = AsyncMock()
        begin_ctx.__aenter__ = AsyncMock(return_value=None)
        begin_ctx.__aexit__ = AsyncMock(return_value=False)
        session.begin = MagicMock(return_value=begin_ctx)

        result = MagicMock()
        result.scalars.return_value.all.return_value = [None, None]
        session.execute = AsyncMock(return_value=result)

        factory = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        factory.return_value = ctx

        scheduler = DecayScheduler(
            factory,
            embedding_engine=EmbeddingEngine(_FakeBackend()),
            llm_provider="openai",
        )

        with patch("clara.scheduler.decay.ReflectionEngine") as reflection_cls:
            reflection = reflection_cls.return_value
            reflection.run = AsyncMock(return_value=[
                UpdateResult(ActionTaken.created, None, False),
            ])

            summary = await scheduler.run_daily_reflection()

        assert summary == {"users_processed": 1, "insights_generated": 1}
        assert reflection.run.await_args.kwargs["user_id"] is None
