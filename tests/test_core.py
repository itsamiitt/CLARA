"""
Tests for clara.core — enums, schemas, and exceptions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from clara.core.enums import (
    EventStatus,
    EVENT_TRANSITIONS,
    MemoryStatus,
    MemoryType,
    SkillOutcome,
    SourceType,
    validate_event_transition,
)
from clara.core.exceptions import (
    ClaraError,
    ConfigurationError,
    ConflictError,
    InvalidTransitionError,
    MemoryNotFoundError,
    TenantViolationError,
)
from clara.core.schemas import (
    ExtractionCandidate,
    InteractionRecord,
    MemorySummary,
)


# ===================================================================
# Enums
# ===================================================================

class TestMemoryTypeReExport:
    """MemoryType should be accessible from clara.core.enums."""

    def test_has_all_values(self):
        assert MemoryType.belief.value == "belief"
        assert MemoryType.event.value == "event"
        assert MemoryType.skill.value == "skill"
        assert MemoryType.world_model.value == "world_model"

    def test_is_same_class_as_db_models(self):
        from clara.db.models import MemoryType as OriginalMemoryType
        assert MemoryType is OriginalMemoryType


class TestMemoryStatusReExport:
    def test_has_all_values(self):
        assert MemoryStatus.active.value == "active"
        assert MemoryStatus.superseded.value == "superseded"
        assert MemoryStatus.deprecated.value == "deprecated"
        assert MemoryStatus.archived.value == "archived"


class TestSourceTypeReExport:
    def test_has_all_values(self):
        assert SourceType.user_direct.value == "user_direct"
        assert SourceType.user_indirect.value == "user_indirect"
        assert SourceType.tool_api.value == "tool_api"
        assert SourceType.system.value == "system"
        assert SourceType.agent_inference.value == "agent_inference"
        assert SourceType.agent_reflection.value == "agent_reflection"


class TestEventStatus:
    def test_has_all_values(self):
        assert EventStatus.created.value == "created"
        assert EventStatus.in_progress.value == "in_progress"
        assert EventStatus.completed.value == "completed"
        assert EventStatus.failed.value == "failed"
        assert EventStatus.abandoned.value == "abandoned"

    def test_is_string_enum(self):
        assert isinstance(EventStatus.created, str)
        assert EventStatus.created == "created"


class TestEventTransitions:
    """Verify that the EVENT_TRANSITIONS map enforces valid state machine rules."""

    def test_created_can_go_to_in_progress(self):
        assert validate_event_transition(EventStatus.created, EventStatus.in_progress) is True

    def test_created_can_go_to_completed(self):
        assert validate_event_transition(EventStatus.created, EventStatus.completed) is True

    def test_created_can_go_to_failed(self):
        assert validate_event_transition(EventStatus.created, EventStatus.failed) is True

    def test_created_can_go_to_abandoned(self):
        assert validate_event_transition(EventStatus.created, EventStatus.abandoned) is True

    def test_in_progress_can_go_to_completed(self):
        assert validate_event_transition(EventStatus.in_progress, EventStatus.completed) is True

    def test_in_progress_can_go_to_failed(self):
        assert validate_event_transition(EventStatus.in_progress, EventStatus.failed) is True

    def test_in_progress_can_go_to_abandoned(self):
        assert validate_event_transition(EventStatus.in_progress, EventStatus.abandoned) is True

    def test_completed_is_terminal(self):
        assert validate_event_transition(EventStatus.completed, EventStatus.created) is False
        assert validate_event_transition(EventStatus.completed, EventStatus.in_progress) is False
        assert validate_event_transition(EventStatus.completed, EventStatus.failed) is False

    def test_failed_is_terminal(self):
        assert validate_event_transition(EventStatus.failed, EventStatus.created) is False
        assert validate_event_transition(EventStatus.failed, EventStatus.in_progress) is False

    def test_abandoned_is_terminal(self):
        assert validate_event_transition(EventStatus.abandoned, EventStatus.created) is False


class TestSkillOutcome:
    def test_has_both_values(self):
        assert SkillOutcome.success.value == "success"
        assert SkillOutcome.failure.value == "failure"


# ===================================================================
# Schemas
# ===================================================================

class TestInteractionRecord:
    def test_create_minimal(self):
        record = InteractionRecord(
            interaction_id=uuid.uuid4(),
            raw_text="Hello world",
            source=SourceType.user_direct,
        )
        assert record.raw_text == "Hello world"
        assert record.source == SourceType.user_direct
        assert record.session_id is None
        assert record.user_id is None
        assert isinstance(record.timestamp, datetime)
        assert record.metadata == {}

    def test_create_full(self):
        ts = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
        record = InteractionRecord(
            interaction_id=uuid.uuid4(),
            raw_text="I use Rust for systems work",
            source=SourceType.user_direct,
            session_id="session-123",
            user_id="user-001",
            timestamp=ts,
            metadata={"channel": "chat"},
        )
        assert record.session_id == "session-123"
        assert record.user_id == "user-001"
        assert record.timestamp == ts
        assert record.metadata == {"channel": "chat"}

    def test_is_frozen(self):
        record = InteractionRecord(
            interaction_id=uuid.uuid4(),
            raw_text="test",
            source=SourceType.user_direct,
        )
        with pytest.raises(AttributeError):
            record.raw_text = "modified"  # type: ignore[misc]


class TestExtractionCandidate:
    def test_create_belief_candidate(self):
        candidate = ExtractionCandidate(
            candidate_type="belief",
            subject="user",
            relation="uses",
            object="Rust",
            confidence=0.9,
            raw_evidence="I use Rust for systems work",
            domain="systems",
        )
        assert candidate.candidate_type == "belief"
        assert candidate.subject == "user"
        assert candidate.domain == "systems"
        assert candidate.is_negation is False
        assert candidate.negates is None

    def test_create_negation_candidate(self):
        candidate = ExtractionCandidate(
            candidate_type="belief",
            subject="user",
            relation="uses",
            object="Python",
            confidence=0.8,
            raw_evidence="I stopped using Python",
            is_negation=True,
            negates="user uses Python",
        )
        assert candidate.is_negation is True
        assert candidate.negates == "user uses Python"

    def test_create_event_candidate(self):
        candidate = ExtractionCandidate(
            candidate_type="event",
            subject="user",
            relation="deployed",
            object="API",
            confidence=0.85,
            raw_evidence="I deployed the API yesterday",
            event_type="deployed",
        )
        assert candidate.candidate_type == "event"
        assert candidate.event_type == "deployed"


class TestMemorySummary:
    def test_create_summary(self):
        summary = MemorySummary(
            memory_id="abc-123",
            memory_type="belief",
            content={"subject": "user", "relation": "uses", "object": "Rust"},
            confidence=0.85,
            status="active",
            user_id="user-001",
            created_at="2026-03-11T12:00:00Z",
        )
        assert summary.memory_id == "abc-123"
        assert summary.memory_type == "belief"
        assert summary.user_id == "user-001"

    def test_optional_fields_default_to_none(self):
        summary = MemorySummary(
            memory_id="abc-123",
            memory_type="belief",
            content={},
            confidence=0.5,
            status="active",
        )
        assert summary.user_id is None
        assert summary.created_at is None
        assert summary.updated_at is None


# ===================================================================
# Exceptions
# ===================================================================

class TestExceptionHierarchy:
    def test_all_inherit_from_clara_error(self):
        assert issubclass(MemoryNotFoundError, ClaraError)
        assert issubclass(InvalidTransitionError, ClaraError)
        assert issubclass(ConflictError, ClaraError)
        assert issubclass(TenantViolationError, ClaraError)
        assert issubclass(ConfigurationError, ClaraError)

    def test_clara_error_inherits_from_exception(self):
        assert issubclass(ClaraError, Exception)


class TestMemoryNotFoundError:
    def test_default_message(self):
        err = MemoryNotFoundError("abc-123")
        assert "abc-123" in str(err)
        assert err.memory_id == "abc-123"

    def test_custom_message(self):
        err = MemoryNotFoundError("abc-123", message="Custom msg")
        assert str(err) == "Custom msg"


class TestInvalidTransitionError:
    def test_default_message(self):
        err = InvalidTransitionError("completed", "in_progress")
        assert "completed" in str(err)
        assert "in_progress" in str(err)
        assert err.current_status == "completed"
        assert err.target_status == "in_progress"


class TestConflictError:
    def test_default_message(self):
        err = ConflictError("direct_replacement", existing_id="mem-001")
        assert "direct_replacement" in str(err)
        assert err.conflict_type == "direct_replacement"
        assert err.existing_id == "mem-001"


class TestTenantViolationError:
    def test_default_message(self):
        err = TenantViolationError("user-A", "user-B")
        assert "user-A" in str(err)
        assert "user-B" in str(err)
        assert err.requesting_user == "user-A"
        assert err.owning_user == "user-B"


class TestConfigurationError:
    def test_can_raise_with_message(self):
        with pytest.raises(ConfigurationError, match="Missing API key"):
            raise ConfigurationError("Missing API key")
