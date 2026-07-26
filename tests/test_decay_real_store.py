"""Real-store decay tests: exact confidence math, batching, and cursor.

The bulk of ``tests/test_decay.py`` drives a hand-written mock session that
re-implements the decay WHERE/UPDATE in Python — so a regression in the real
SQL would pass green. These tests instead run the REAL ``DecayScheduler`` over a
file-backed SQLite store and assert exact post-decay confidence values, that
archival flips status, and that batched keyset pagination visits every row
(the dashed-vs-dashless UUID cursor bug re-fetched a batch's last row forever).
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from clara.agent import _make_engine
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.scheduler.decay import (
    ARCHIVAL_THRESHOLD,
    DecayScheduler,
    compute_decayed_confidence,
)


@pytest_asyncio.fixture
async def factory(tmp_path):
    db_path = (tmp_path / "clara.db").as_posix()
    engine = _make_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield sf
    finally:
        await engine.dispose()


async def _insert_belief(
    factory,
    *,
    confidence: float,
    decay_rate: float,
    age_days: float,
    subject: str = "user",
) -> uuid.UUID:
    mid = uuid.uuid4()
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    async with factory() as session, session.begin():
        session.add(
            Memory(
                memory_id=mid,
                memory_type=MemoryType.belief,
                user_id=None,
                content={"subject": subject, "relation": "prefers", "object": "x"},
                confidence=confidence,
                status=MemoryStatus.active,
                decay_rate=decay_rate,
                created_at=ts,
                updated_at=ts,
                metadata_={},
            )
        )
    return mid


async def _get(factory, mid: uuid.UUID) -> Memory:
    async with factory() as session:
        return (
            await session.execute(select(Memory).where(Memory.memory_id == mid))
        ).scalars().one()


@pytest.mark.asyncio
async def test_daily_decay_applies_exact_exponential(factory):
    # confidence 0.9, rate 0.02, 10 days => 0.9 * e^(-0.2)
    mid = await _insert_belief(factory, confidence=0.9, decay_rate=0.02, age_days=10)
    scheduler = DecayScheduler(factory)

    summary = await scheduler.run_daily_decay()

    assert summary["decayed"] >= 1
    row = await _get(factory, mid)
    expected = compute_decayed_confidence(0.9, 0.02, 10.0)
    # The real SQL must produce the same value the pure helper documents.
    assert row.confidence == pytest.approx(expected, abs=1e-6)
    assert row.confidence == pytest.approx(0.9 * math.exp(-0.2), abs=1e-6)
    assert row.status == MemoryStatus.active


@pytest.mark.asyncio
async def test_daily_decay_archives_below_threshold(factory):
    # Very old + volatile => confidence falls under ARCHIVAL_THRESHOLD.
    mid = await _insert_belief(factory, confidence=0.5, decay_rate=0.02, age_days=2000)
    scheduler = DecayScheduler(factory)

    summary = await scheduler.run_daily_decay()

    assert summary["archived"] >= 1
    row = await _get(factory, mid)
    assert row.confidence < ARCHIVAL_THRESHOLD
    assert row.status == MemoryStatus.archived


@pytest.mark.asyncio
async def test_zero_decay_rate_is_untouched(factory):
    mid = await _insert_belief(factory, confidence=0.7, decay_rate=0.0, age_days=365)
    scheduler = DecayScheduler(factory)

    await scheduler.run_daily_decay()

    row = await _get(factory, mid)
    assert row.confidence == pytest.approx(0.7, abs=1e-9)


@pytest.mark.asyncio
async def test_batched_decay_visits_every_row(factory):
    # More rows than one batch so the keyset cursor must advance correctly.
    # The dashed/dashless UUID cursor bug re-selected a batch's max row and
    # stalled; if any row is skipped, its confidence stays at the seed value.
    scheduler = DecayScheduler(factory)
    scheduler.BATCH_SIZE = 10
    ids = [
        await _insert_belief(
            factory, confidence=0.9, decay_rate=0.02, age_days=30, subject=f"s{i}"
        )
        for i in range(25)
    ]

    summary = await scheduler.run_daily_decay()

    assert summary["decayed"] == 25
    expected = compute_decayed_confidence(0.9, 0.02, 30.0)
    for mid in ids:
        row = await _get(factory, mid)
        assert row.confidence == pytest.approx(expected, abs=1e-6), (
            "a row was skipped by keyset pagination"
        )


@pytest.mark.asyncio
async def test_decay_does_not_bump_updated_at(factory):
    # Pure decay is bookkeeping, not a content change: updated_at must be
    # preserved or recency ranking would flatten store-wide every night.
    mid = await _insert_belief(factory, confidence=0.9, decay_rate=0.02, age_days=10)
    before = (await _get(factory, mid)).updated_at
    scheduler = DecayScheduler(factory)

    await scheduler.run_daily_decay()

    after = (await _get(factory, mid)).updated_at
    assert after == before
