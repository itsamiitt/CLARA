"""
Tests for clara.core — enums, schemas, and exceptions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from clara.core.enums import (
    EventStatus,
    MemoryStatus,
    MemoryType,
    SkillOutcome,
    SourceType,
    validate_event_transition,
)
from clara.core.exceptions import (
    ClaraError,
    InvalidTransitionError,
    MemoryNotFoundError,
)
from clara.interaction.layer import InteractionRecord

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
# Schemas — the live InteractionRecord (clara/interaction/layer.py)
# ===================================================================

class TestInteractionRecord:
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
            session_id=None,
            user_id=None,
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )
        with pytest.raises(AttributeError):
            record.raw_text = "modified"  # type: ignore[misc]


# ===================================================================
# Exceptions
# ===================================================================

class TestExceptionHierarchy:
    def test_all_inherit_from_clara_error(self):
        assert issubclass(MemoryNotFoundError, ClaraError)
        assert issubclass(InvalidTransitionError, ClaraError)

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


