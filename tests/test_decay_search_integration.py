"""
Regression tests: status changes from the decay scheduler must propagate to the
vector store so archived/deprecated memories drop out of ``recall()``.

This locks in the SQLite -> LanceDB sync path (old bug #10). It drives the REAL
DecayScheduler and the REAL retrieval engine over a file-backed SQLite store and
an isolated LanceDB table (see tests/conftest.py), using a deterministic fake
embedding backend so no network/model is needed.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from clara.agent import ClaraMemory, _make_engine
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.embeddings import EmbeddingEngine
from clara.scheduler.decay import DecayScheduler

FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        magnitude = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / magnitude for v in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _FixedExtractor:
    """Returns a preset list of facts regardless of input text."""

    def __init__(self, facts: list[ExtractedFact]) -> None:
        self._facts = facts

    async def extract(self, text: str) -> list[ExtractedFact]:
        return list(self._facts)


def _fact(subject, relation, object_, *, confidence=0.9) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        relation=relation,
        object=object_,
        domain=None,
        source_type="user_direct",
        confidence=confidence,
        is_negation=False,
        raw_text="seed",
    )


@pytest_asyncio.fixture
async def agent_with_scheduler(tmp_path):
    db_path = (tmp_path / "clara.db").as_posix()
    engine = _make_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    embedder = EmbeddingEngine(_FakeBackend())
    facts = [
        _fact("user", "prefers", "darkmode"),   # -> belief
        _fact("user", "knows", "kubernetes"),   # -> skill
    ]
    agent = ClaraMemory(
        engine=engine,
        session_factory=factory,
        embedding_engine=embedder,
        extractor=_FixedExtractor(facts),  # type: ignore[arg-type]
        decay_scheduler=None,
    )
    scheduler = DecayScheduler(factory, embedding_engine=embedder)  # not started

    await agent.remember("seed")
    try:
        yield agent, scheduler, factory
    finally:
        await agent.close()
        await engine.dispose()


async def _recall_objects(agent: ClaraMemory, query: str) -> set[str]:
    # Search no longer flushes synchronously, so drain pending vector writes
    # (e.g. decay/prune status changes) before asserting on recall results.
    agent._lance_engine.flush_pending_sync()
    result = await agent.recall(query, top_k=10)
    return {sm.memory.content.get("object") for sm in result.all}


async def _age_memory(factory, memory_type: MemoryType, days: int) -> None:
    """Push a memory's timestamps into the past so decay/pruning acts on it."""
    old = datetime.now(timezone.utc) - timedelta(days=days)
    async with factory() as session:
        async with session.begin():
            row = (
                await session.execute(
                    select(Memory).where(Memory.memory_type == memory_type)
                )
            ).scalars().first()
            row.created_at = old
            row.updated_at = old


class TestDecayPropagatesToSearch:
    @pytest.mark.asyncio
    async def test_seeded_memories_are_recallable(self, agent_with_scheduler):
        agent, _scheduler, _factory = agent_with_scheduler
        assert "darkmode" in await _recall_objects(agent, "darkmode preference")
        assert "kubernetes" in await _recall_objects(agent, "kubernetes skill")

    @pytest.mark.asyncio
    async def test_daily_decay_archive_excludes_from_recall(self, agent_with_scheduler):
        agent, scheduler, factory = agent_with_scheduler

        # Make the belief ancient so exponential decay drops it below threshold.
        await _age_memory(factory, MemoryType.belief, days=2000)
        summary = await scheduler.run_daily_decay()
        assert summary["archived"] >= 1

        # SQLite says archived...
        async with factory() as session:
            belief = (
                await session.execute(
                    select(Memory).where(Memory.memory_type == MemoryType.belief)
                )
            ).scalars().first()
            assert belief.status == MemoryStatus.archived

        # ...and the vector search no longer returns it.
        assert "darkmode" not in await _recall_objects(agent, "darkmode preference")

    @pytest.mark.asyncio
    async def test_weekly_prune_deprecate_excludes_from_recall(self, agent_with_scheduler):
        agent, scheduler, factory = agent_with_scheduler

        await _age_memory(factory, MemoryType.skill, days=120)
        summary = await scheduler.run_weekly_pruning()
        assert summary["skills_deprecated"] >= 1

        async with factory() as session:
            skill = (
                await session.execute(
                    select(Memory).where(Memory.memory_type == MemoryType.skill)
                )
            ).scalars().first()
            assert skill.status == MemoryStatus.deprecated

        assert "kubernetes" not in await _recall_objects(agent, "kubernetes skill")
