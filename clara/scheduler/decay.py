"""
CLARA — Decay Scheduler

Runs three scheduled jobs using APScheduler:

1. **Daily confidence decay** (every day at 02:00 UTC)
   Applies exponential decay to every active memory record:

       confidence_t = confidence_0 × e^(−decay_rate × days_since_updated)

   Decay rates (stored per-record in ``Memory.decay_rate``; keep in sync with
   the per-store constants that write them — clara.memory.belief/skill/
   world_model and clara.update.engine):
     - belief_stable   → 0.005
     - belief_volatile → 0.02
     - event           → 0.0    (events never decay)
     - skill           → 0.02   (SKILL_DECAY_RATE)
     - world_model     → 0.005  (WORLD_MODEL_DECAY_RATE)

   Records whose confidence drops below the archival threshold (0.15)
   are set to ``status = "archived"``, except skills, which remain active
   until the weekly pruning job can mark stale ones as ``deprecated``.

2. **Weekly pruning** (every Sunday at 02:30 UTC)
   - Archive events older than 90 days that have no linked beliefs.
   - Deprecate skills unused for more than 60 days.

3. **Daily reflection** (every day at 03:00 UTC)
   Generate tenant-scoped insight beliefs from recent memories.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from typing import cast as type_cast

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import String, Table, bindparam, cast, select
from sqlalchemy import text as sa_text
from sqlalchemy import update as sa_update
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


def _id_cursor(memory_id: object) -> str:
    """Keyset cursor string matching ``cast(memory_id, String)`` on SQLite.

    SQLite stores ``Uuid`` as 32-char dashless hex, so the cursor must be the
    dashless form for the ``>`` comparison to advance correctly. Accepts a
    ``uuid.UUID`` (``.hex``) or a string the driver already returned.
    """
    hex_attr = getattr(memory_id, "hex", None)
    return hex_attr if isinstance(hex_attr, str) else str(memory_id).replace("-", "")


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
    """Last real usage of a skill: the most recent of ``metadata.last_used``
    (stamped by SkillStore.record_outcome) and ``metadata.last_accessed``
    (stamped when the skill is a search hit), else ``created_at``.

    A skill surfaced in context every day but never explicitly outcome-scored
    was previously deprecated after 60 days because only ``last_used`` was
    consulted while access recording writes ``last_accessed``. Deliberately
    ignores ``updated_at``, which the daily decay job resets every night via
    the ORM ``onupdate`` hook.
    """
    meta = dict(record.metadata_) if record.metadata_ else {}
    candidates: list[datetime] = []
    for key in ("last_used", "last_accessed"):
        raw = meta.get(key)
        if isinstance(raw, str):
            with contextlib.suppress(ValueError):
                candidates.append(_ensure_aware(datetime.fromisoformat(raw)))
    if candidates:
        return max(candidates)
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
        reflection_generator: Any = None,
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
        logger.info(
            "DecayScheduler started (daily decay @ 02:00 UTC, weekly prune @ Sun 02:30 UTC)"
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying APScheduler."""
        self._scheduler.shutdown(wait=wait)
        logger.info("DecayScheduler shut down")

    # ------------------------------------------------------------------
    # Daily decay
    # ------------------------------------------------------------------

    # Rows mutated per transaction. Bounds write-lock hold time so a decay
    # pass over a 100k-row store never starves concurrent sessions; a crash
    # mid-run is safe because ``metadata.last_decay_at`` anchors per-row.
    BATCH_SIZE = 500

    async def run_daily_decay(self) -> dict[str, int]:
        """Apply exponential confidence decay to all active records.

        Batched keyset pagination, one transaction per batch. Pure-decay
        writes self-assign ``updated_at`` so the ORM ``onupdate`` does NOT
        fire — decay is bookkeeping, not a content change, and letting it
        bump ``updated_at`` flattened recency ranking store-wide every night.
        Archival transitions are real state changes and do bump it.

        Returns a summary dict: ``{"decayed": N, "archived": M}``.
        """
        now = datetime.now(timezone.utc)
        decayed_count = 0
        archived_count = 0
        last_id = ""
        table = type_cast(Table, Memory.__table__)

        while True:
            async with self._session_factory() as session, session.begin():
                # Core column select — no ORM instances, no identity map, no
                # unit-of-work flush: a decay pass reads plain values and
                # writes two executemany statements per batch.
                stmt = (
                    select(
                        table.c.memory_id,
                        table.c.memory_type,
                        table.c.confidence,
                        table.c.decay_rate,
                        table.c.updated_at,
                        table.c.metadata,
                    )
                    .where(table.c.status == "active")
                    .where(table.c.decay_rate > 0.0)
                    .where(cast(table.c.memory_id, String) > last_id)
                    .order_by(cast(table.c.memory_id, String))
                    .limit(self.BATCH_SIZE)
                )
                rows = (await session.execute(stmt)).all()
                if not rows:
                    break
                # Cursor must match what `cast(memory_id, String)` produces —
                # SQLite stores Uuid as 32-char dashless hex, so the dashed
                # str(uuid) form used previously sorted below every hex digit
                # and re-fetched the prior batch's max row forever (the weekly
                # skill loop already uses .hex; keep them consistent).
                last_id = max(_id_cursor(row.memory_id) for row in rows)

                pending: list[dict[str, object]] = []
                archive_pending: list[dict[str, object]] = []
                for row in rows:
                    meta = dict(row.metadata) if isinstance(row.metadata, dict) else {}
                    anchor_raw = meta.get("last_decay_at")
                    anchor = None
                    if isinstance(anchor_raw, str):
                        try:
                            anchor = _ensure_aware(datetime.fromisoformat(anchor_raw))
                        except ValueError:
                            anchor = None
                    if anchor is None:
                        anchor = _ensure_aware(row.updated_at)
                    days_elapsed = (now - anchor).total_seconds() / 86_400.0
                    if days_elapsed <= 0:
                        continue

                    new_confidence = compute_decayed_confidence(
                        float(row.confidence), float(row.decay_rate), days_elapsed
                    )
                    meta["last_decay_at"] = now.isoformat()
                    entry = {
                        "b_mid": row.memory_id,
                        "b_conf": new_confidence,
                        "b_meta": meta,
                    }
                    is_skill = str(row.memory_type) in ("skill", "MemoryType.skill")
                    if new_confidence < self._archival_threshold and not is_skill:
                        archive_pending.append(entry)
                        archived_count += 1
                    else:
                        pending.append(entry)
                        decayed_count += 1

                # Pure decay: self-assigned updated_at — bookkeeping must not
                # look like a content change or recency ranking flattens.
                if pending:
                    await session.execute(
                        sa_update(table)
                        .where(table.c.memory_id == bindparam("b_mid"))
                        .values(
                            {
                                "confidence": bindparam("b_conf"),
                                "metadata": bindparam("b_meta"),
                                "updated_at": table.c.updated_at,
                            }
                        ),
                        pending,
                    )
                # Archival: a real state change — updated_at bumps.
                if archive_pending:
                    await session.execute(
                        sa_update(table)
                        .where(table.c.memory_id == bindparam("b_mid"))
                        .values(
                            {
                                "confidence": bindparam("b_conf"),
                                "metadata": bindparam("b_meta"),
                                "status": "archived",
                                "updated_at": now,
                            }
                        ),
                        archive_pending,
                    )
                    logger.info(
                        "Archived %d memories below threshold %.2f",
                        len(archive_pending),
                        self._archival_threshold,
                    )
                if len(rows) < self.BATCH_SIZE:
                    break

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

        # --- 1. Stale events (pushed down + batched) ------------------
        # ``related_beliefs`` absent or an empty list means unlinked; the
        # json_extract pushdown keeps linked events from ever being loaded.
        event_cutoff = now - timedelta(days=self._event_stale_days)
        unlinked = sa_text(
            "(json_extract(metadata, '$.related_beliefs') IS NULL "
            "OR json_extract(metadata, '$.related_beliefs') = '[]')"
        )
        while True:
            async with self._session_factory() as session, session.begin():
                ids = [
                    row[0]
                    for row in (
                        await session.execute(
                            select(Memory.memory_id)
                            .where(Memory.memory_type == MemoryType.event)
                            .where(Memory.status == MemoryStatus.active)
                            .where(Memory.created_at < event_cutoff)
                            .where(unlinked)
                            .limit(self.BATCH_SIZE)
                        )
                    ).all()
                ]
                if not ids:
                    break
                await session.execute(
                    sa_update(Memory)
                    .where(Memory.memory_id.in_(ids))
                    .values(status=MemoryStatus.archived, updated_at=now)
                    .execution_options(synchronize_session=False)
                )
                events_archived += len(ids)
                logger.info("Pruned %d stale unlinked events", len(ids))
                if len(ids) < self.BATCH_SIZE:
                    break

        # --- 2. Unused skills (batched) -------------------------------
        # NOTE: the cutoff must NOT use ``updated_at`` — real usage lives in
        # ``metadata.last_used`` (stamped by SkillStore.record_outcome and by
        # search-hit access recording), falling back to ``created_at`` for
        # skills that were never exercised.
        skill_cutoff = now - timedelta(days=self._skill_unused_days)
        last_id = ""
        while True:
            async with self._session_factory() as session, session.begin():
                stmt = (
                    select(Memory)
                    .where(Memory.memory_type == MemoryType.skill)
                    .where(Memory.status == MemoryStatus.active)
                    .where(cast(Memory.memory_id, String) > last_id)
                    .order_by(cast(Memory.memory_id, String))
                    .limit(self.BATCH_SIZE)
                )
                active_skills: Sequence[Memory] = (
                    (await session.execute(stmt)).scalars().all()
                )
                if not active_skills:
                    break
                last_id = max(_id_cursor(s.memory_id) for s in active_skills)

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
                if len(active_skills) < self.BATCH_SIZE:
                    break

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
            # No outer transaction: ReflectionEngine.run() opens a write
            # transaction per insight so the per-user LLM calls are not made
            # while a transaction is held. The user-list read below releases
            # its own snapshot before run() is called.
            result = await session.execute(select(Memory.user_id).distinct())
            user_ids = list(result.scalars().all())
            await session.rollback()
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
