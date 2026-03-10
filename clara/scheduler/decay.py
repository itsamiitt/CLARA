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
   are set to ``status = "archived"``.

2. **Weekly pruning** (every Sunday at 02:30 UTC)
   - Archive events older than 90 days that have no linked beliefs.
   - Deprecate skills unused for more than 60 days.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clara.db.models import Memory, MemoryStatus, MemoryType

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

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
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

                    if should_archive(new_confidence):
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
                event_cutoff = now - timedelta(days=EVENT_STALE_DAYS)
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
                skill_cutoff = now - timedelta(days=SKILL_UNUSED_DAYS)
                stmt = (
                    select(Memory)
                    .where(Memory.memory_type == MemoryType.skill)
                    .where(Memory.status == MemoryStatus.active)
                    .where(Memory.updated_at < skill_cutoff)
                )
                result = await session.execute(stmt)
                unused_skills: Sequence[Memory] = result.scalars().all()

                for skill in unused_skills:
                    skill.status = MemoryStatus.deprecated
                    skill.updated_at = now
                    skills_deprecated += 1
                    logger.info(
                        "Deprecated unused skill %s (last updated=%s)",
                        skill.memory_id,
                        skill.updated_at.isoformat(),
                    )

        summary = {
            "events_archived": events_archived,
            "skills_deprecated": skills_deprecated,
        }
        logger.info("Weekly pruning complete: %s", summary)
        return summary
