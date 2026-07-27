"""
Saves carrying tool-call framing are rejected, not stored.

Observed in a real store: a belief's evidence text ended with

    ...(leads, call_sessions).</parameter>\n<parameter name="domain">security

The tool call was malformed, the XML framing was glued into the description,
and the domain field was silently swallowed. CLARA stored it verbatim, so the
store held a mangled fact and no signal that a field had gone missing.

Rejecting is recoverable — the model re-sends plain text. Storing is not:
nothing downstream can tell mangled from meant. The rare legitimate memory
*about* this markup loses to the observed failure mode, and the error says
exactly what to do instead.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

# Verbatim from the affected store, shortened only in the middle.
REAL_LEAK = (
    "Fix: drive the warn-log clean, expand FORCE RLS onto high-risk "
    'workspace_id tables (leads, call_sessions).</parameter>\n'
    '<parameter name="domain">security'
)


def call(name, args):
    from clara.integrations.mcp_server import build_server

    server = build_server()
    result = asyncio.run(server.call_tool(name, args))
    return result[1] if isinstance(result, tuple) else result


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))


class TestTheObservedCorruption:
    def test_the_exact_leaked_string_is_rejected(self) -> None:
        with pytest.raises(Exception) as caught:
            call("memory_save", {
                "mem_type": "belief",
                "subject": "ClosoCRM tenant isolation",
                "relation": "is enforced only by",
                "object": "the application layer",
                "description": REAL_LEAK,
            })
        message = str(caught.value)
        assert "tool-call markup" in message
        assert "re-send" in message.lower()

    def test_nothing_was_stored(self, tmp_path) -> None:
        with pytest.raises(Exception, match="tool-call markup"):
            call("memory_save", {
                "mem_type": "belief", "subject": "s", "relation": "r",
                "object": "o", "description": REAL_LEAK,
            })
        found = call("memory_search", {"query": "tenant"})
        assert found["total"] == 0


class TestWhereMarkupCanHide:
    @pytest.mark.parametrize("field, value", [
        ("subject", "x</parameter>y"),
        ("object", '<invoke name="Bash">'),
        ("description", "text <function_calls> text"),
        ("tags", ["fine", "bad</invoke>"]),
        ("properties", {"nested": {"deep": "a<parameter name='x'>b"}}),
    ])
    def test_every_field_is_scanned(self, field, value) -> None:
        args = {"mem_type": "world_model", "entity_type": "svc", "name": "api",
                "subject": "s", "relation": "r", "object": "o"}
        args[field] = value
        with pytest.raises(Exception) as caught:
            call("memory_save", args)
        assert "tool-call markup" in str(caught.value)

    def test_save_many_names_the_item_and_field(self) -> None:
        items = [
            {"mem_type": "belief", "subject": "a", "relation": "r", "object": "o"},
            {"mem_type": "belief", "subject": "b", "relation": "r",
             "object": "o", "description": REAL_LEAK},
        ]
        with pytest.raises(Exception) as caught:
            call("memory_save_many", {"items": items})
        message = str(caught.value)
        assert "item 1" in message
        assert "description" in message

    def test_update_tags_are_scanned(self) -> None:
        saved = call("memory_save", {"mem_type": "belief", "subject": "s",
                                     "relation": "r", "object": "o"})
        with pytest.raises(Exception) as caught:
            call("memory_update", {"memory_id": saved["memory_id"],
                                   "tags": ["ok", "</parameter>"]})
        assert "tool-call markup" in str(caught.value)


class TestLegitimateContentStillSaves:
    def test_ordinary_technical_text_passes(self) -> None:
        payload = call("memory_save", {
            "mem_type": "belief",
            "subject": "api",
            "relation": "validates input with",
            "object": "pydantic models and XML schemas",
            "description": "uses <root> and <config> elements, parameters "
                           "documented per endpoint",
        })
        assert payload["action"] == "saved"

    def test_angle_brackets_alone_are_not_markup(self) -> None:
        payload = call("memory_save", {
            "mem_type": "belief", "subject": "team",
            "relation": "compares versions with", "object": "a < b and b > a",
        })
        assert payload["action"] == "saved"
