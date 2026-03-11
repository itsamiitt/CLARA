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


# ---------------------------------------------------------------------------
# Conflict errors
# ---------------------------------------------------------------------------

class ConflictError(ClaraError):
    """Raised when a memory conflict cannot be automatically resolved.

    Attributes:
        conflict_type:   Type of conflict detected.
        existing_id:     UUID of the conflicting existing record.
        candidate_text:  Human-readable description of the new candidate.
    """

    def __init__(
        self,
        conflict_type: str,
        existing_id: str | None = None,
        candidate_text: str | None = None,
        message: str | None = None,
    ) -> None:
        self.conflict_type = conflict_type
        self.existing_id = existing_id
        self.candidate_text = candidate_text
        super().__init__(
            message
            or f"Unresolvable {conflict_type!r} conflict with memory {existing_id!r}."
        )


# ---------------------------------------------------------------------------
# Multi-tenant errors
# ---------------------------------------------------------------------------

class TenantViolationError(ClaraError):
    """Raised when a cross-tenant data access is attempted.

    Attributes:
        requesting_user:  The user_id making the request.
        owning_user:      The user_id that owns the resource.
    """

    def __init__(
        self,
        requesting_user: str,
        owning_user: str,
        message: str | None = None,
    ) -> None:
        self.requesting_user = requesting_user
        self.owning_user = owning_user
        super().__init__(
            message
            or (
                f"Tenant violation: user {requesting_user!r} attempted to "
                f"access resources owned by user {owning_user!r}."
            )
        )


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(ClaraError):
    """Raised when CLARA is misconfigured (missing API keys, bad URLs, etc.)."""
