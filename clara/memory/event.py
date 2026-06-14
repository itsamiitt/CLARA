"""
CLARA — Event Memory Store

Manages lifecycle-tracked event memories: creation, state transitions,
timeline queries, and entity-scoped retrieval.

Events never decay (``decay_rate = 0.0``) and follow a state machine:

    created → in_progress → completed | failed | abandoned

The ``EventStore`` is session-bound and flush-only — the caller manages
transaction boundaries.

Usage::

    async with async_session() as session:
        events = EventStore(session)
        record = await events.create(
            subject="user",
            event_type="deployed",
            description="Deployed v2.1 of the API to production",
            user_id="alice",
        )
        await events.update_outcome(record.memory_id, EventStatus.completed)
        await session.commit()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clara.core.enums import EventStatus, validate_event_transition
from clara.core.exceptions import InvalidTransitionError, MemoryNotFoundError
from clara.db.models import Memory, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)

# Events never decay
EVENT_DECAY_RATE = 0.0


class EventStore:
    """Async interface for creating, transitioning, and querying event memories.

    Each instance is bound to an ``AsyncSession``.  The caller is responsible
    for committing or rolling back the session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        subject: str,
        event_type: str,
        description: str = "",
        domain: str | None = None,
        user_id: str | None = None,
        confidence: float = 1.0,
        source_type: str = "user_direct",
        raw_text: str | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Create a new event memory with status ``created``.

        Args:
            subject:      Entity the event is about (e.g. ``"user"``).
            event_type:   Verb/action (e.g. ``"deployed"``, ``"migrated"``).
            description:  Human-readable description of the event.
            domain:       Optional domain tag for scoping.
            user_id:      Tenant partition key.
            confidence:   Initial confidence (events default to 1.0).
            source_type:  Evidence source classification string.
            raw_text:     Original text that produced this event.
            embedding:    Precomputed embedding vector.

        Returns:
            The newly created :class:`Memory` row (not yet committed).
        """
        now = datetime.now(timezone.utc)

        content: dict[str, Any] = {
            "subject": subject,
            "relation": event_type,
            "object": description,
            "event_type": event_type,
            "event_status": EventStatus.created.value,
        }
        if domain:
            content["domain"] = domain

        metadata: dict[str, Any] = {
            "source_type": source_type,
            "raw_text": raw_text,
            "event_history": [
                {
                    "status": EventStatus.created.value,
                    "timestamp": now.isoformat(),
                }
            ],
        }

        record = Memory(
            memory_type=MemoryType.event,
            user_id=user_id,
            content=content,
            embedding=embedding,
            confidence=confidence,
            status=MemoryStatus.active,
            decay_rate=EVENT_DECAY_RATE,
            created_at=now,
            updated_at=now,
            metadata_=metadata,
        )
        self._session.add(record)
        await self._session.flush()

        logger.debug("Created event %s: %s %s", record.memory_id, subject, event_type)
        return record

    # ------------------------------------------------------------------
    # update_outcome — lifecycle state machine
    # ------------------------------------------------------------------

    async def update_outcome(
        self,
        event_id: uuid.UUID,
        new_status: EventStatus,
        *,
        outcome_detail: str | None = None,
    ) -> Memory:
        """Transition an event to a new lifecycle state.

        Args:
            event_id:        UUID of the event record.
            new_status:      Target :class:`EventStatus`.
            outcome_detail:  Optional description of the outcome.

        Returns:
            The updated :class:`Memory` row.

        Raises:
            MemoryNotFoundError:    If the event doesn't exist.
            InvalidTransitionError: If the transition is not allowed.
        """
        record = await self._get_event(event_id)

        content = dict(record.content) if isinstance(record.content, dict) else {}
        current_status_str = content.get("event_status", EventStatus.created.value)

        try:
            current_status = EventStatus(current_status_str)
        except ValueError:
            logger.warning(
                "Event %s has unrecognized event_status %r; treating it as "
                "%r for transition validation.",
                event_id, current_status_str, EventStatus.created.value,
            )
            current_status = EventStatus.created

        if not validate_event_transition(current_status, new_status):
            raise InvalidTransitionError(
                current_status.value,
                new_status.value,
                f"Event {event_id}: cannot transition from "
                f"{current_status.value!r} to {new_status.value!r}.",
            )

        now = datetime.now(timezone.utc)

        # Update content
        content["event_status"] = new_status.value
        if outcome_detail:
            content["outcome_detail"] = outcome_detail
        record.content = content

        # Append to event history
        meta = dict(record.metadata_) if record.metadata_ else {}
        history = list(meta.get("event_history", []))
        history_entry: dict[str, Any] = {
            "status": new_status.value,
            "timestamp": now.isoformat(),
        }
        if outcome_detail:
            history_entry["detail"] = outcome_detail
        history.append(history_entry)
        meta["event_history"] = history
        record.metadata_ = meta

        record.updated_at = now
        await self._session.flush()

        logger.debug(
            "Event %s transitioned: %s → %s",
            event_id, current_status.value, new_status.value,
        )
        return record

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get(self, event_id: uuid.UUID) -> Memory | None:
        """Retrieve a single event by UUID. Returns ``None`` if not found."""
        return await self._get_event_or_none(event_id)

    async def get_timeline(
        self,
        *,
        user_id: str | None = None,
        subject: str | None = None,
        limit: int = 50,
    ) -> Sequence[Memory]:
        """Return active events ordered by creation time (newest first).

        Args:
            user_id:  Filter to a specific tenant.
            subject:  Filter to events about a specific entity.
            limit:    Maximum number of records to return.

        Returns:
            A list of :class:`Memory` rows.
        """
        filters = [
            Memory.memory_type == MemoryType.event,
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if subject is not None:
            filters.append(Memory.content["subject"].as_string() == subject)

        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_entity(
        self,
        entity: str,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[Memory]:
        """Return events where subject or object matches *entity*."""
        filters = [
            Memory.memory_type == MemoryType.event,
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        # Match on subject field
        filters.append(Memory.content["subject"].as_string() == entity)

        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_event_or_none(self, event_id: uuid.UUID) -> Memory | None:
        stmt = select(Memory).where(
            and_(
                Memory.memory_id == event_id,
                Memory.memory_type == MemoryType.event,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_event(self, event_id: uuid.UUID) -> Memory:
        record = await self._get_event_or_none(event_id)
        if record is None:
            raise MemoryNotFoundError(str(event_id))
        return record
