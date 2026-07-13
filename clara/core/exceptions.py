"""
CLARA — Core Exceptions

Custom exception hierarchy for CLARA.  All CLARA-specific exceptions
inherit from :class:`ClaraError` so callers can catch the full family
with a single ``except ClaraError``.
"""

from __future__ import annotations


class ClaraError(Exception):
    """Base exception for all CLARA errors."""


# ---------------------------------------------------------------------------
# Memory store errors
# ---------------------------------------------------------------------------

class MemoryNotFoundError(ClaraError):
    """Raised when a memory record lookup returns no result.

    Attributes:
        memory_id: The UUID that was searched for (as string).
    """

    def __init__(self, memory_id: str, message: str | None = None) -> None:
        self.memory_id = memory_id
        super().__init__(message or f"Memory record {memory_id!r} not found.")


class InvalidTransitionError(ClaraError):
    """Raised when an event lifecycle transition is invalid.

    Attributes:
        current_status:  The current status of the event.
        target_status:   The attempted target status.
    """

    def __init__(
        self,
        current_status: str,
        target_status: str,
        message: str | None = None,
    ) -> None:
        self.current_status = current_status
        self.target_status = target_status
        super().__init__(
            message
            or f"Cannot transition from {current_status!r} to {target_status!r}."
        )
