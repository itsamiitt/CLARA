"""
CLARA — Decay Scheduler

Runs two scheduled jobs using APScheduler:

1. **Daily confidence decay** (every day at 02:00 UTC)
   Applies exponential decay to every active memory record:

       confidence_t = confidence_0 × e^(−decay_rate × days_since_updated)

   Decay rates (stored per-record in ``Memory.decay_rate``):
     - belief_stable   → 0.005
     - belief_volatile → 0.02
     - event           → 0.0   (events never decay)
     - skill           → 0.01
     - world_model     → 0.03

   Records whose confidence drops below the archival threshold (0.15)
   are set to ``status = "archived"``, except skills, which remain active
   until the weekly pruning job can mark stale ones as ``deprecated``.

2. **Weekly pruning** (every Sunday at 02:30 UTC)
   - Archive events older than 90 days that have no linked beliefs.
   - Deprecate skills unused for more than 60 days.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.reflection import ReflectionEngine
from clara.retrieval.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVAL_THRESHOLD: float = 0.15

# Event archival: events older than this many days with no linked beliefs
EVENT_STALE_DAYS: int = 90

# Skill deprecation: skills unused for longer than this many days
SKILL_UNUSED_DAYS: int = 60


# ---------------------------------------------------------------------------
# Pure helpers (stateless, easy to test)
# ---------------------------------------------------------------------------

def _ensure_aware(dt: datetime) -> datetime:
    """Normalize naive datetimes to UTC for SQLite-backed test environments."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _decay_anchor(record: Memory) -> datetime:
    """Use the last decay timestamp when present, else fall back to updated_at."""
    meta = dict(record.metadata_) if record.metadata_ else {}
    last_decay_raw = meta.get("last_decay_at")
    if isinstance(last_decay_raw, str):
        try:
            return _ensure_aware(datetime.fromisoformat(last_decay_raw))
        except ValueError:
            pass
    return _ensure_aware(record.updated_at)

def _skill_last_used(record: Memory) -> datetime:
    """Last real usage of a skill: ``metadata.last_used`` (stamped by
    SkillStore.record_outcome) when parseable, else ``created_at``.

    Deliberately ignores ``updated_at``, which the daily decay job resets
    every night via the ORM ``onupdate`` hook.
    """
    meta = dict(record.metadata_) if record.metadata_ else {}
    last_used_raw = meta.get("last_used")
    if isinstance(last_used_raw, str):
        try:
            return _ensure_aware(datetime.fromisoformat(last_used_raw))
        except ValueError:
            pass
    return _ensure_aware(record.created_at)


def compute_decayed_confidence(
    confidence_0: float,
    decay_rate: float,
    days_elapsed: float,
) -> float:
    """Return ``confidence_0 × e^(−decay_rate × days_elapsed)``.

    If *decay_rate* is zero the original confidence is returned unchanged.
    The result is clamped to ``[0.0, 1.0]``.
    """
    if decay_rate == 0.0:
        return confidence_0
    decayed = confidence_0 * math.exp(-decay_rate * days_elapsed)
    return max(0.0, min(1.0, decayed))


def should_archive(confidence: float) -> bool:
    """Return ``True`` when the confidence has fallen below the archival threshold."""
    return confidence < ARCHIVAL_THRESHOLD


# ---------------------------------------------------------------------------
# DecayScheduler
# ---------------------------------------------------------------------------

class DecayScheduler:
    """Manages the daily decay and weekly pruning jobs.

    Usage::

        scheduler = DecayScheduler(session_factory)
        scheduler.start()     # registers + starts APScheduler
        ...
        scheduler.shutdown()  # graceful teardown

    The ``session_factory`` must be an
    :class:`~sqlalchemy.ext.asyncio.async_sessionmaker` so that each job
    execution receives its own session (and therefore its own transaction).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_engine: EmbeddingEngine | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        reflection_generator=None,
        *,
        archival_threshold: float = ARCHIVAL_THRESHOLD,
        event_stale_days: int = EVENT_STALE_DAYS,
        skill_unused_days: int = SKILL_UNUSED_DAYS,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_engine = embedding_engine
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._reflection_generator = reflection_generator
        self._archival_threshold = archival_threshold
        self._event_stale_days = event_stale_days
        self._skill_unused_days = skill_unused_days
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register jobs and start the underlying APScheduler."""
        self._scheduler.add_job(
            self.run_daily_decay,
            trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
            id="daily_decay",
            name="Daily confidence decay",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self.run_weekly_pruning,
            trigger=CronTrigger(day_of_week="sun", hour=2, minute=30, timezone="UTC"),
            id="weekly_pruning",
            name="Weekly memory pruning",
            replace_existing=True,
        )
        if self._embedding_engine is not None and (
            self._llm_provider is not None or self._reflection_generator is not None
        ):
            self._scheduler.add_job(
                self.run_daily_reflection,
                trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
                id="daily_reflection",
                name="Daily reflection & insight generation",
                replace_existing=True,
            )
        self._scheduler.start()
        logger.info("DecayScheduler started (daily decay @ 02:00 UTC, weekly prune @ Sun 02:30 UTC)")

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying APScheduler."""
        self._scheduler.shutdown(wait=wait)
        logger.info("DecayScheduler shut down")

    # ------------------------------------------------------------------
    # Daily decay
    # ------------------------------------------------------------------

    async def run_daily_decay(self) -> dict[str, int]:
        """Apply exponential confidence decay to all active records.

        Returns a summary dict: ``{"decayed": N, "archived": M}``.
        """
        now = datetime.now(timezone.utc)
        decayed_count = 0
        archived_count = 0

        async with self._session_factory() as session:
            async with session.begin():
                # Fetch all active memories that actually decay
                stmt = (
                    select(Memory)
                    .where(Memory.status == MemoryStatus.active)
                    .where(Memory.decay_rate > 0.0)
                )
                result = await session.execute(stmt)
                records: Sequence[Memory] = result.scalars().all()

                for record in records:
                    decay_anchor = _decay_anchor(record)
                    days_elapsed = (now - decay_anchor).total_seconds() / 86_400.0
                    if days_elapsed <= 0:
                        continue

                    old_confidence = record.confidence
                    new_confidence = compute_decayed_confidence(
                        old_confidence,
                        record.decay_rate,
                        days_elapsed,
                    )
                    meta = dict(record.metadata_) if record.metadata_ else {}
                    meta["last_decay_at"] = now.isoformat()
                    record.metadata_ = meta

                    if (
                        new_confidence < self._archival_threshold
                        and record.memory_type != MemoryType.skill
                    ):
                        record.status = MemoryStatus.archived
                        record.confidence = new_confidence
                        record.updated_at = now
                        archived_count += 1
                        logger.info(
                            "Archived memory %s (type=%s, confidence=%.4f -> %.4f)",
                            record.memory_id,
                            record.memory_type.value,
                            old_confidence,
                            new_confidence,
                        )
                    else:
                        record.confidence = new_confidence
                        decayed_count += 1

        summary = {"decayed": decayed_count, "archived": archived_count}
        logger.info("Daily decay complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Weekly pruning
    # ------------------------------------------------------------------

    async def run_weekly_pruning(self) -> dict[str, int]:
        """Run the weekly pruning pass.

        1. Archive events older than 90 days with no linked beliefs.
        2. Deprecate skills unused for more than 60 days.

        Returns a summary dict:
        ``{"events_archived": N, "skills_deprecated": M}``.
        """
        now = datetime.now(timezone.utc)
        events_archived = 0
        skills_deprecated = 0

        async with self._session_factory() as session:
            async with session.begin():
                # --- 1. Stale events ----------------------------------
                event_cutoff = now - timedelta(days=self._event_stale_days)
                stmt = (
                    select(Memory)
                    .where(Memory.memory_type == MemoryType.event)
                    .where(Memory.status == MemoryStatus.active)
                    .where(Memory.created_at < event_cutoff)
                )
                result = await session.execute(stmt)
                stale_events: Sequence[Memory] = result.scalars().all()

                for event in stale_events:
                    linked = (event.metadata_ or {}).get("related_beliefs", [])
                    if not linked:
                        event.status = MemoryStatus.archived
                        event.updated_at = now
                        events_archived += 1
                        logger.info(
                            "Pruned stale event %s (created_at=%s, no linked beliefs)",
                            event.memory_id,
                            event.created_at.isoformat(),
                        )

                # --- 2. Unused skills ---------------------------------
                # NOTE: the cutoff must NOT use ``updated_at`` — the daily
                # decay job dirties every skill row each night, which fires
                # the ORM ``onupdate`` hook and resets ``updated_at``, so an
                # ``updated_at``-based predicate can never match and skills
                # become immortal. Real usage lives in ``metadata.last_used``
                # (stamped by SkillStore.record_outcome), falling back to
                # ``created_at`` for skills that were never exercised.
                skill_cutoff = now - timedelta(days=self._skill_unused_days)
                stmt = (
                    select(Memory)
                    .where(Memory.memory_type == MemoryType.skill)
                    .where(Memory.status == MemoryStatus.active)
                )
                result = await session.execute(stmt)
                active_skills: Sequence[Memory] = result.scalars().all()

                for skill in active_skills:
                    last_used = _skill_last_used(skill)
                    if last_used >= skill_cutoff:
                        continue
                    skill.status = MemoryStatus.deprecated
                    skill.updated_at = now
                    skills_deprecated += 1
                    logger.info(
                        "Deprecated unused skill %s (last used=%s)",
                        skill.memory_id,
                        last_used.isoformat(),
                    )

        summary = {
            "events_archived": events_archived,
            "skills_deprecated": skills_deprecated,
        }
        logger.info("Weekly pruning complete: %s", summary)
        return summary

    async def run_daily_reflection(self) -> dict[str, int]:
        """Generate tenant-scoped reflection insights from recent memories."""
        if self._embedding_engine is None:
            return {"users_processed": 0, "insights_generated": 0}

        users_processed = 0
        insights_generated = 0

        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(select(Memory.user_id).distinct())
                user_ids = list(result.scalars().all())
                concrete_user_ids = sorted({user_id for user_id in user_ids if user_id is not None})
                has_legacy_rows = any(user_id is None for user_id in user_ids)

                reflection = ReflectionEngine(
                    session,
                    self._embedding_engine,
                    llm_provider=self._llm_provider or "openai",
                    llm_model=self._llm_model,
                    insight_generator=self._reflection_generator,
                )

                for user_id in concrete_user_ids:
                    results = await reflection.run(user_id=user_id)
                    users_processed += 1
                    insights_generated += len(results)

                if not concrete_user_ids and has_legacy_rows:
                    results = await reflection.run(user_id=None)
                    users_processed += 1
                    insights_generated += len(results)

        summary = {
            "users_processed": users_processed,
            "insights_generated": insights_generated,
        }
        logger.info("Daily reflection complete: %s", summary)
        return summary
