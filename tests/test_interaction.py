from __future__ import annotations

from clara.interaction import InteractionLayer
from clara.memory.belief import SourceType


class TestInteractionLayer:
    def test_normalizes_whitespace(self):
        layer = InteractionLayer()

        record = layer.receive("  hello   from   clara  ")

        assert record.raw_text == "hello from clara"

    def test_assigns_uuid_and_timestamp(self):
        layer = InteractionLayer()

        record = layer.receive("hello")

        assert record.interaction_id is not None
        assert record.timestamp.tzinfo is not None

    def test_accepts_source_enum(self):
        layer = InteractionLayer()

        record = layer.receive("hello", source=SourceType.tool_api)

        assert record.source == SourceType.tool_api

    def test_accepts_source_string(self):
        layer = InteractionLayer()

        record = layer.receive("hello", source="system")

        assert record.source == SourceType.system

    def test_sets_session_and_user(self):
        layer = InteractionLayer()

        record = layer.receive(
            "hello",
            session_id="sess-1",
            user_id="alice",
        )

        assert record.session_id == "sess-1"
        assert record.user_id == "alice"

    def test_sets_default_confidence_floor(self):
        layer = InteractionLayer()

        record = layer.receive("hello")

        assert record.metadata["confidence_floor"] == 0.4

    def test_empty_input_raises(self):
        layer = InteractionLayer()

        try:
            layer.receive("   ")
        except ValueError as exc:
            assert "cannot be empty" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError for empty interaction input")

    def test_invalid_source_raises(self):
        layer = InteractionLayer()

        try:
            layer.receive("hello", source="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid source")
