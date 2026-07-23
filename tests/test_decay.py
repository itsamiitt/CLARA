"""
Tests for clara.scheduler.decay — DecayScheduler with mocked DB + scheduler.

All tests mock the SQLAlchemy async session and APScheduler timer.
No real database or scheduler threads are started.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.retrieval.engine import LanceRetrievalEngine
from clara.scheduler.decay import (
    ARCHIVAL_THRESHOLD,
    EVENT_STALE_DAYS,
    SKILL_UNUSED_DAYS,
    DecayScheduler,
    compute_decayed_confidence,
    should_archive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeMemory:
    """Lightweight stand-in for the Memory ORM model."""

    def __init__(
        self,
        *,
        memory_type: MemoryType = MemoryType.belief,
        confidence: float = 0.80,
        decay_rate: float = 0.02,
        status: MemoryStatus = MemoryStatus.active,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        memory_id: uuid.UUID | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> None:
        self.memory_id = memory_id or uuid.uuid4()
        self.memory_type = memory_type
        self.confidence = confidence
        self.decay_rate = decay_rate
        self.status = status
        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.metadata_ = metadata_ if metadata_ is not None else {}
        self.content: dict[str, Any] = {}


def _make_session_factory(
    records: list[FakeMemory],
    *,
    event_cutoff_days: int = EVENT_STALE_DAYS,
    skill_cutoff_days: int = SKILL_UNUSED_DAYS,
) -> AsyncMock:
    """Build a mock ``async_sessionmaker`` that yields *records* on queries.

    The mock routes calls based on the filters in each ``select`` statement so
    the daily-decay and weekly-pruning code-paths get the right rows.

    Date-based filtering (created_at / updated_at cutoffs) is applied here in
    the mock so that the production code's WHERE clauses are faithfully
    simulated.
    """
    now = datetime.now(timezone.utc)
    session = AsyncMock()

    _call_count = {"n": 0}

    def _apply_one(param_row: dict) -> int:
        """Apply one UPDATE parameter row (single-row or executemany) to
        the fake records; returns the number of records touched."""
        target_ids: set[uuid.UUID] = set()
        for key, value in param_row.items():
            if "mid" not in key and not key.startswith("memory_id"):
                continue
            if isinstance(value, (list, tuple)):
                target_ids.update(value)
            elif value is not None:
                target_ids.add(value)
        touched = 0
        for r in records:
            if r.memory_id not in target_ids:
                continue
            touched += 1
            for conf_key in ("b_conf", "confidence"):
                if param_row.get(conf_key) is not None:
                    r.confidence = param_row[conf_key]
            for meta_key in ("b_meta", "metadata"):
                if param_row.get(meta_key) is not None:
                    r.metadata_ = param_row[meta_key]
            if param_row.get("status") is not None:
                r.status = param_row["status"]
            if param_row.get("updated_at") is not None:
                r.updated_at = param_row["updated_at"]
        return touched

    def _apply_update(stmt, extra_params=None) -> MagicMock:
        """Apply a Core UPDATE (decay/prune batches) to the fake records.

        The production code stopped mutating ORM instances for pure decay —
        it emits ``update(Memory)...`` (executemany for decay batches) so
        the ``onupdate`` hook is bypassed — and the mock mirrors that by
        applying the bound params here.
        """
        touched = 0
        if isinstance(extra_params, (list, tuple)):
            # Static VALUES entries (e.g. status='archived', updated_at=now)
            # live in the statement; per-row b_* params ride the list.
            try:
                static = dict(stmt.compile().params)
            except Exception:
                static = {}
            for row in extra_params:
                touched += _apply_one({**static, **dict(row)})
        else:
            merged = dict(stmt.compile().params)
            if isinstance(extra_params, dict):
                merged.update(extra_params)
            touched = _apply_one(merged)
        result = MagicMock()
        result.rowcount = touched
        return result

    async def _execute(stmt, params=None):
        """Inspect the compiled SQL to decide which subset to return."""
        from sqlalchemy.sql.dml import Update as _Update

        if isinstance(stmt, _Update):
            return _apply_update(stmt, params)

        scalars_mock = MagicMock()

        # Build a rough clause string for routing.
        # We inspect the WHERE portion only, because column names like
        # ``memories.memory_type`` always appear in the SELECT list.
        try:
            full = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        except Exception:
            full = str(stmt)

        where = full.split("WHERE", 1)[1] if "WHERE" in full else full

        branch = None
        matching = []
        for r in records:
            # --- daily decay: active + decay_rate > 0 ---
            if "decay_rate >" in where and "'event'" not in where and "'skill'" not in where:
                branch = "decay"
                if (
                    r.status == MemoryStatus.active
                    and r.decay_rate > 0.0
                ):
                    matching.append(r)

            # --- weekly: stale events ---
            # The unlinked check is pushed down to SQL (json_extract on
            # related_beliefs) — mirror it here.
            elif "'event'" in where:
                branch = "event"
                event_cutoff = now - timedelta(days=event_cutoff_days)
                if (
                    r.memory_type == MemoryType.event
                    and r.status == MemoryStatus.active
                    and r.created_at < event_cutoff
                    and not (r.metadata_ or {}).get("related_beliefs")
                ):
                    matching.append(r)

            # --- weekly: unused skills ---
            # Production selects ALL active skills and applies the
            # last-used cutoff in Python (updated_at is unusable — the
            # daily decay job resets it every night via onupdate).
            elif "'skill'" in where:
                branch = "skill"
                if (
                    r.memory_type == MemoryType.skill
                    and r.status == MemoryStatus.active
                ):
                    matching.append(r)

        scalars_mock.all.return_value = matching
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        if branch == "decay":
            # The decay pass reads Core column rows, not ORM instances.
            from types import SimpleNamespace

            result_mock.all.return_value = [
                SimpleNamespace(
                    memory_id=r.memory_id,
                    memory_type=r.memory_type.value,
                    confidence=r.confidence,
                    decay_rate=r.decay_rate,
                    updated_at=r.updated_at,
                    metadata=r.metadata_,
                )
                for r in matching
            ]
        else:
            # The stale-event pass selects bare memory_id rows.
            result_mock.all.return_value = [(r.memory_id,) for r in matching]
        return result_mock

    session.execute = AsyncMock(side_effect=_execute)

    # Support ``async with session.begin(): ...``
    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    # The factory is an async context manager itself
    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx

    return factory


# ---------------------------------------------------------------------------
# Pure-function tests: compute_decayed_confidence
# ---------------------------------------------------------------------------

class TestComputeDecayedConfidence:
    def test_zero_decay_rate_returns_original(self):
        assert compute_decayed_confidence(0.9, 0.0, 100) == 0.9

    def test_zero_days_returns_original(self):
        assert compute_decayed_confidence(0.8, 0.02, 0) == 0.8

    def test_belief_stable_decay(self):
        """Decay rate 0.005 over 30 days → e^(−0.15) ≈ 0.8607."""
        result = compute_decayed_confidence(1.0, 0.005, 30)
        expected = math.exp(-0.005 * 30)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_belief_volatile_decay(self):
        """Decay rate 0.02 over 30 days → e^(−0.6) ≈ 0.5488."""
        result = compute_decayed_confidence(1.0, 0.02, 30)
        expected = math.exp(-0.02 * 30)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_skill_decay(self):
        """Decay rate 0.01 over 60 days → e^(−0.6) ≈ 0.5488."""
        result = compute_decayed_confidence(1.0, 0.01, 60)
        expected = math.exp(-0.01 * 60)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_world_model_decay(self):
        """Decay rate 0.03 over 30 days → e^(−0.9) ≈ 0.4066."""
        result = compute_decayed_confidence(1.0, 0.03, 30)
        expected = math.exp(-0.03 * 30)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_decays_below_threshold(self):
        """With high decay and long elapsed time, confidence drops below 0.15."""
        result = compute_decayed_confidence(0.5, 0.03, 200)
        assert result < ARCHIVAL_THRESHOLD

    def test_clamped_to_zero(self):
        """Result is never negative even with extreme parameters."""
        result = compute_decayed_confidence(0.01, 1.0, 1000)
        assert result >= 0.0

    def test_clamped_to_one(self):
        """Result never exceeds 1.0 (should not happen with decay, but guard)."""
        result = compute_decayed_confidence(1.0, 0.001, 1)
        assert result <= 1.0

    def test_partial_confidence(self):
        """Starting at 0.6 with volatile decay over 50 days."""
        result = compute_decayed_confidence(0.6, 0.02, 50)
        expected = 0.6 * math.exp(-0.02 * 50)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_event_decay_rate_zero(self):
        """Events use decay_rate=0.0, so confidence stays unchanged."""
        assert compute_decayed_confidence(0.75, 0.0, 365) == 0.75


# ---------------------------------------------------------------------------
# Pure-function tests: should_archive
# ---------------------------------------------------------------------------

class TestShouldArchive:
    def test_below_threshold(self):
        assert should_archive(0.10) is True

    def test_at_threshold(self):
        assert should_archive(0.15) is False

    def test_above_threshold(self):
        assert should_archive(0.50) is False

    def test_zero(self):
        assert should_archive(0.0) is True

    def test_barely_below(self):
        assert should_archive(0.1499) is True


# ---------------------------------------------------------------------------
# DecayScheduler — lifecycle / APScheduler integration
# ---------------------------------------------------------------------------

class TestDecaySchedulerLifecycle:
    def test_start_registers_jobs(self):
        factory = MagicMock()
        scheduler = DecayScheduler(factory)

        with patch.object(scheduler._scheduler, "add_job") as add_job, \
             patch.object(scheduler._scheduler, "start"):
            scheduler.start()

        assert add_job.call_count == 2
        job_ids = {call.kwargs["id"] for call in add_job.call_args_list}
        assert "daily_decay" in job_ids
        assert "weekly_pruning" in job_ids

    def test_shutdown_delegates(self):
        factory = MagicMock()
        scheduler = DecayScheduler(factory)

        with patch.object(scheduler._scheduler, "shutdown") as mock_shutdown:
            scheduler.shutdown(wait=False)

        mock_shutdown.assert_called_once_with(wait=False)


# ---------------------------------------------------------------------------
# Daily decay job
# ---------------------------------------------------------------------------

class TestRunDailyDecay:
    @pytest.mark.asyncio
    async def test_decays_active_records(self):
        """Active record with decay_rate>0 should have its confidence reduced."""
        ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
        record = FakeMemory(
            confidence=0.80,
            decay_rate=0.02,
            updated_at=ten_days_ago,
        )
        factory = _make_session_factory([record])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        expected = compute_decayed_confidence(0.80, 0.02, 10)
        assert record.confidence == pytest.approx(expected, abs=1e-4)
        assert record.status == MemoryStatus.active
        assert summary["decayed"] == 1
        assert summary["archived"] == 0

    @pytest.mark.asyncio
    async def test_archival_threshold_is_configurable(self):
        """Regression: CLARA_ARCHIVAL_THRESHOLD used to be parsed and then
        ignored — the scheduler ctor param must actually gate archival."""
        record = FakeMemory(
            confidence=0.40,
            decay_rate=0.001,
            updated_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        factory = _make_session_factory([record])
        scheduler = DecayScheduler(factory, archival_threshold=0.5)

        summary = await scheduler.run_daily_decay()

        assert record.status == MemoryStatus.archived
        assert summary["archived"] == 1

    @pytest.mark.asyncio
    async def test_archives_when_below_threshold(self):
        """Record that decays below 0.15 should be archived."""
        long_ago = datetime.now(timezone.utc) - timedelta(days=200)
        record = FakeMemory(
            memory_type=MemoryType.belief,
            confidence=0.20,
            decay_rate=0.03,
            updated_at=long_ago,
        )
        factory = _make_session_factory([record])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        assert record.status == MemoryStatus.archived
        assert summary["archived"] == 1
        assert summary["decayed"] == 0

    @pytest.mark.asyncio
    async def test_skills_below_threshold_remain_active_for_pruning(self):
        """Low-confidence skills should decay but stay active until weekly pruning."""
        long_ago = datetime.now(timezone.utc) - timedelta(days=200)
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            confidence=0.20,
            decay_rate=0.03,
            updated_at=long_ago,
        )
        factory = _make_session_factory([skill])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        assert skill.confidence < ARCHIVAL_THRESHOLD
        assert skill.status == MemoryStatus.active
        assert summary["archived"] == 0
        assert summary["decayed"] == 1

    @pytest.mark.asyncio
    async def test_skips_zero_decay_rate(self):
        """Events (decay_rate=0) should not be included in the query at all."""
        record = FakeMemory(
            memory_type=MemoryType.event,
            confidence=0.90,
            decay_rate=0.0,
            updated_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        factory = _make_session_factory([record])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        # Event should have been filtered out by the WHERE clause mock
        assert record.confidence == 0.90
        assert summary["decayed"] == 0
        assert summary["archived"] == 0

    @pytest.mark.asyncio
    async def test_multiple_records(self):
        """Process a mix of records: some decay, some archive."""
        now = datetime.now(timezone.utc)
        records = [
            # Will decay but stay active
            FakeMemory(
                confidence=0.90,
                decay_rate=0.005,
                updated_at=now - timedelta(days=5),
            ),
            # Will be archived (low confidence + high decay + old)
            FakeMemory(
                confidence=0.16,
                decay_rate=0.03,
                updated_at=now - timedelta(days=30),
            ),
            # Event — skipped
            FakeMemory(
                memory_type=MemoryType.event,
                decay_rate=0.0,
                confidence=0.80,
                updated_at=now - timedelta(days=200),
            ),
        ]
        factory = _make_session_factory(records)
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        assert summary["decayed"] == 1
        assert summary["archived"] == 1

    @pytest.mark.asyncio
    async def test_recently_updated_not_decayed(self):
        """A record updated just now should have ~0 days elapsed → no change."""
        record = FakeMemory(
            confidence=0.80,
            decay_rate=0.02,
            updated_at=datetime.now(timezone.utc),
        )
        factory = _make_session_factory([record])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_daily_decay()

        # days_elapsed ≈ 0 so confidence should stay the same (or the loop skips it)
        assert record.confidence == pytest.approx(0.80, abs=0.01)


# ---------------------------------------------------------------------------
# Weekly pruning job
# ---------------------------------------------------------------------------

class TestRunWeeklyPruning:
    @pytest.mark.asyncio
    async def test_archives_stale_events_no_linked_beliefs(self):
        """Events older than 90 days with no related_beliefs → archived."""
        old = datetime.now(timezone.utc) - timedelta(days=EVENT_STALE_DAYS + 10)
        event = FakeMemory(
            memory_type=MemoryType.event,
            decay_rate=0.0,
            created_at=old,
            updated_at=old,
            metadata_={"related_beliefs": []},
        )
        factory = _make_session_factory([event])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert event.status == MemoryStatus.archived
        assert summary["events_archived"] == 1

    @pytest.mark.asyncio
    async def test_keeps_stale_events_with_linked_beliefs(self):
        """Events older than 90 days but WITH related_beliefs → kept."""
        old = datetime.now(timezone.utc) - timedelta(days=EVENT_STALE_DAYS + 10)
        event = FakeMemory(
            memory_type=MemoryType.event,
            decay_rate=0.0,
            created_at=old,
            updated_at=old,
            metadata_={"related_beliefs": ["bel_001"]},
        )
        factory = _make_session_factory([event])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert event.status == MemoryStatus.active
        assert summary["events_archived"] == 0

    @pytest.mark.asyncio
    async def test_keeps_recent_events(self):
        """Events younger than 90 days → kept regardless of linked beliefs."""
        recent = datetime.now(timezone.utc) - timedelta(days=30)
        event = FakeMemory(
            memory_type=MemoryType.event,
            decay_rate=0.0,
            created_at=recent,
            updated_at=recent,
            metadata_={"related_beliefs": []},
        )
        factory = _make_session_factory([event])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        # The WHERE clause filters out recent events, so this should not match
        assert summary["events_archived"] == 0

    @pytest.mark.asyncio
    async def test_deprecates_unused_skills(self):
        """Skills not *used* in 60+ days → deprecated (keyed on
        metadata.last_used, not updated_at — the daily decay job bumps
        updated_at every night via the ORM onupdate hook)."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=SKILL_UNUSED_DAYS + 5)
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            decay_rate=0.01,
            updated_at=now,  # freshly touched by last night's decay pass
            metadata_={"last_used": old.isoformat()},
        )
        factory = _make_session_factory([skill])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert skill.status == MemoryStatus.deprecated
        assert summary["skills_deprecated"] == 1

    @pytest.mark.asyncio
    async def test_recently_used_skill_survives_pruning(self):
        """Regression (immortal-skills inverse): a skill with a recent
        metadata.last_used stays active regardless of created_at age."""
        now = datetime.now(timezone.utc)
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            decay_rate=0.01,
            created_at=now - timedelta(days=365),
            updated_at=now - timedelta(days=365),
            metadata_={"last_used": (now - timedelta(days=3)).isoformat()},
        )
        factory = _make_session_factory([skill])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert skill.status == MemoryStatus.active
        assert summary["skills_deprecated"] == 0

    @pytest.mark.asyncio
    async def test_keeps_recently_used_skills(self):
        """Skills updated within 60 days → kept active."""
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            decay_rate=0.01,
            updated_at=recent,
        )
        factory = _make_session_factory([skill])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert skill.status == MemoryStatus.active
        assert summary["skills_deprecated"] == 0


@pytest_asyncio.fixture
async def pruning_db():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


class TestDecayLanceSync:
    @pytest.mark.asyncio
    async def test_weekly_pruning_enqueues_status_changes_to_bound_lance_engine(self, pruning_db):
        lance = MagicMock(spec=LanceRetrievalEngine)
        factory = async_sessionmaker(
            pruning_db,
            expire_on_commit=False,
            info={"_lance_engine": lance},
        )
        old = datetime.now(timezone.utc) - timedelta(days=SKILL_UNUSED_DAYS + 5)

        async with factory() as session:
            session.add(
                Memory(
                    memory_type=MemoryType.skill,
                    user_id="alice",
                    content={"subject": "user", "relation": "knows", "object": "Docker"},
                    embedding=[0.1, 0.2, 0.3],
                    confidence=0.5,
                    status=MemoryStatus.active,
                    decay_rate=0.01,
                    created_at=old,
                    updated_at=old,
                    metadata_={},
                )
            )
            await session.commit()

        lance.reset_mock()
        scheduler = DecayScheduler(factory)
        summary = await scheduler.run_weekly_pruning()

        assert summary["skills_deprecated"] == 1
        lance.enqueue_records.assert_called()
        snapshots = lance.enqueue_records.call_args.args[0]
        assert len(snapshots) == 1
        assert snapshots[0].status == MemoryStatus.deprecated.value

    @pytest.mark.asyncio
    async def test_combined_pruning(self):
        """Both stale events and unused skills are processed in a single run."""
        now = datetime.now(timezone.utc)
        event = FakeMemory(
            memory_type=MemoryType.event,
            decay_rate=0.0,
            created_at=now - timedelta(days=120),
            updated_at=now - timedelta(days=120),
            metadata_={"related_beliefs": []},
        )
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            decay_rate=0.01,
            created_at=now - timedelta(days=90),  # never used since creation
            updated_at=now - timedelta(days=90),
        )
        factory = _make_session_factory([event, skill])
        scheduler = DecayScheduler(factory)

        summary = await scheduler.run_weekly_pruning()

        assert event.status == MemoryStatus.archived
        assert skill.status == MemoryStatus.deprecated
        assert summary["events_archived"] == 1
        assert summary["skills_deprecated"] == 1

    @pytest.mark.asyncio
    async def test_old_low_confidence_skill_deprecates_after_daily_decay(self):
        """Skills should survive daily decay and be deprecated by weekly pruning."""
        old = datetime.now(timezone.utc) - timedelta(days=SKILL_UNUSED_DAYS + 30)
        skill = FakeMemory(
            memory_type=MemoryType.skill,
            confidence=0.20,
            decay_rate=0.03,
            created_at=old,  # never exercised since creation
            updated_at=old,
        )
        factory = _make_session_factory([skill])
        scheduler = DecayScheduler(factory)

        daily_summary = await scheduler.run_daily_decay()
        weekly_summary = await scheduler.run_weekly_pruning()

        assert daily_summary == {"decayed": 1, "archived": 0}
        assert skill.status == MemoryStatus.deprecated
        assert weekly_summary["skills_deprecated"] == 1


# ---------------------------------------------------------------------------
# Decay math — integration-level spot checks
# ---------------------------------------------------------------------------

class TestDecayMathIntegration:
    """Verify specific scenarios against hand-computed values."""

    def test_stable_belief_90_days(self):
        """Stable belief (0.005) after 90 days: 0.8 × e^(−0.45) ≈ 0.5101."""
        result = compute_decayed_confidence(0.8, 0.005, 90)
        assert result == pytest.approx(0.8 * math.exp(-0.45), abs=1e-4)

    def test_volatile_belief_reaches_archive(self):
        """Volatile belief (0.02): find how long until 0.5 → 0.15.
        0.5 × e^(−0.02t) = 0.15  →  t = −ln(0.3)/0.02 ≈ 60.2 days.
        """
        days = -math.log(0.15 / 0.5) / 0.02
        result = compute_decayed_confidence(0.5, 0.02, days)
        assert result == pytest.approx(0.15, abs=1e-4)

    def test_world_model_fast_decay(self):
        """World model (0.03) after 60 days: 0.9 × e^(−1.8) ≈ 0.1487 → archived."""
        result = compute_decayed_confidence(0.9, 0.03, 60)
        expected = 0.9 * math.exp(-1.8)
        assert result == pytest.approx(expected, abs=1e-4)
        assert should_archive(result) is True

    def test_skill_moderate_decay(self):
        """Skill (0.01) after 30 days: 0.82 × e^(−0.3) ≈ 0.6074."""
        result = compute_decayed_confidence(0.82, 0.01, 30)
        expected = 0.82 * math.exp(-0.3)
        assert result == pytest.approx(expected, abs=1e-4)
        assert should_archive(result) is False
