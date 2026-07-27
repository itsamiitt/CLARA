"""
CLARA — Core Enums

Canonical re-exports of all CLARA enums in one place, plus new enums
for memory types that don't yet have dedicated modules.

Existing modules (``db.models``, ``memory.belief``) remain the source
of truth for their enums — this module simply re-exports them so that
**new code** has a single, stable import path:

    from clara.core.enums import MemoryType, MemoryStatus, SourceType, EventStatus
"""

from __future__ import annotations

import enum

# Re-export from db.models — the single-table schema enums
from clara.db.models import MemoryStatus, MemoryType

# Re-export from memory.belief — evidence source classification
from clara.memory.belief import SourceType

# ---------------------------------------------------------------------------
# New enums for pending milestones
# ---------------------------------------------------------------------------

class EventStatus(str, enum.Enum):
    """Lifecycle status of an event memory record.

    Events progress through these states:

        created → in_progress → completed | failed | abandoned

    Only ``created → in_progress`` and ``in_progress → *`` transitions
    are valid.
    """

    created = "created"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    abandoned = "abandoned"


# Valid transitions map: current_status → set of allowed next statuses
EVENT_TRANSITIONS: dict[EventStatus, frozenset[EventStatus]] = {
    EventStatus.created: frozenset({
        EventStatus.in_progress,
        EventStatus.completed,
        EventStatus.failed,
        EventStatus.abandoned,
    }),
    EventStatus.in_progress: frozenset({
        EventStatus.completed,
        EventStatus.failed,
        EventStatus.abandoned,
    }),
    EventStatus.completed: frozenset(),
    EventStatus.failed: frozenset(),
    EventStatus.abandoned: frozenset(),
}


def validate_event_transition(
    current: EventStatus,
    target: EventStatus,
) -> bool:
    """Return ``True`` if transitioning from *current* → *target* is valid."""
    return target in EVENT_TRANSITIONS.get(current, frozenset())


class SkillOutcome(str, enum.Enum):
    """Outcome of a skill execution attempt."""

    success = "success"
    failure = "failure"


# ---------------------------------------------------------------------------
# Convenience re-exports for ``from clara.core.enums import *``
# ---------------------------------------------------------------------------

__all__ = [
    "MemoryType",
    "MemoryStatus",
    "SourceType",
    "EventStatus",
    "EVENT_TRANSITIONS",
    "validate_event_transition",
    "SkillOutcome",
]
