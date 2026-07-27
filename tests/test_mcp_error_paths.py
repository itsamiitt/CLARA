"""
What the MCP tools do when the input is wrong.

The caller here is a model. An error it cannot act on is nearly as bad as a
crash: "badly formed hexadecimal UUID string" leaked straight through
memory_update and says nothing about what was passed or what to do instead,
while memory_forget answered the same situation with a clear sentence.

Every case below was run against the real server before being written down.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")


def call(name, args):
    from clara.integrations.mcp_server import build_server

    server = build_server()
    result = asyncio.run(server.call_tool(name, args))
    return result[1] if isinstance(result, tuple) else result


def error_from(exc_info) -> str:
    return str(exc_info.value)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARA_HOME", str(tmp_path / "clara"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    (tmp_path / "claude").mkdir(parents=True)
    return tmp_path


class TestMemoryIdErrors:
    @pytest.mark.parametrize("bad", ["not-a-real-id", "", "12345"])
    def test_update_rejects_a_malformed_id_readably(self, store, bad) -> None:
        with pytest.raises(Exception) as caught:
            call("memory_update", {"memory_id": bad, "confidence": 0.5})
        message = error_from(caught)
        assert "not found" in message
        # The uuid module's own wording is not an answer anyone can act on.
        assert "hexadecimal" not in message

    def test_forget_rejects_a_malformed_id_readably(self, store) -> None:
        with pytest.raises(Exception) as caught:
            call("memory_forget", {"memory_id": "nonsense"})
        message = error_from(caught)
        assert "not found" in message
        assert "hexadecimal" not in message

    def test_well_formed_but_absent_id_still_says_not_found(self, store) -> None:
        with pytest.raises(Exception) as caught:
            call("memory_forget",
                 {"memory_id": "00000000-0000-0000-0000-000000000000"})
        assert "not found" in error_from(caught)


class TestConfidenceRange:
    """The tools document confidence (0..1) and used to accept anything.

    5.0 was stored as 1.0 and reported as saved, so the caller was told its
    value was accepted when it had been replaced.
    """

    @pytest.mark.parametrize("bad", [5.0, -1.0, 1.5])
    def test_out_of_range_is_rejected(self, store, bad) -> None:
        with pytest.raises(Exception) as caught:
            call("memory_save", {"mem_type": "belief", "subject": "a",
                                 "relation": "b", "object": "c",
                                 "confidence": bad})
        message = error_from(caught)
        assert "0.0" in message and "1.0" in message

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
    def test_the_boundaries_are_valid(self, store, good) -> None:
        payload = call("memory_save", {"mem_type": "belief", "subject": "a",
                                       "relation": "likes", "object": "c",
                                       "confidence": good})
        assert payload["action"] == "saved"

    def test_update_validates_too(self, store) -> None:
        saved = call("memory_save", {"mem_type": "belief", "subject": "s",
                                     "relation": "r", "object": "o"})
        with pytest.raises(Exception) as caught:
            call("memory_update",
                 {"memory_id": saved["memory_id"], "confidence": 9.0})
        assert "1.0" in error_from(caught)

    def test_omitting_confidence_is_still_fine(self, store) -> None:
        payload = call("memory_save", {"mem_type": "belief", "subject": "x",
                                       "relation": "uses", "object": "y"})
        assert payload["action"] == "saved"


class TestStatuslineAgainstRealSettingsFiles:
    def test_corrupt_settings_is_refused_and_left_alone(self, store) -> None:
        settings = store / "claude" / "settings.json"
        original = '{"statusLine": {broken json, "hooks": }'
        settings.write_text(original, encoding="utf-8")

        payload = call("statusline_install", {})
        assert payload["ok"] is False
        assert "not valid JSON" in payload["error"]
        # The file is the user's. Refusing must not mean rewriting it.
        assert settings.read_text(encoding="utf-8") == original

    def test_existing_user_keys_survive_installation(self, store) -> None:
        settings = store / "claude" / "settings.json"
        settings.write_text(json.dumps({
            "model": "opus",
            "env": {"MY_KEY": "keep-me"},
            "permissions": {"allow": ["Bash(ls:*)"]},
        }), encoding="utf-8")

        payload = call("statusline_install", {})
        assert payload["ok"] is True
        after = json.loads(settings.read_text(encoding="utf-8"))
        assert after["model"] == "opus"
        assert after["env"]["MY_KEY"] == "keep-me"
        assert after["permissions"]["allow"] == ["Bash(ls:*)"]
        assert "statusLine" in after

    def test_status_reports_corruption_rather_than_claiming_unconfigured(
        self, store
    ) -> None:
        settings = store / "claude" / "settings.json"
        settings.write_text("{ nope", encoding="utf-8")
        payload = call("statusline_status", {})
        assert payload["configured"] is False
        # "not configured" and "cannot be read" are different facts.
        assert "error" in payload


class TestCodeToolArguments:
    def test_invalid_direction_is_named(self, tmp_path, monkeypatch) -> None:
        from clara.db.migrations import open_db
        from clara.index import indexer
        from clara.repoid import repo_id
        from clara.store import resolve_store

        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "clara"))
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "a.ts").write_text('import "react";', encoding="utf-8")
        resolution = resolve_store(str(repo), create=True)
        conn = open_db(str(resolution.db_path))
        indexer.index_repo(conn, repo_id(str(repo)), repo)
        conn.commit()
        conn.close()

        with pytest.raises(Exception) as caught:
            call("code_deps", {"target": "src/a.ts", "direction": "sideways",
                               "repo": str(repo)})
        assert "forward" in error_from(caught)

    def test_unknown_target_is_not_an_error(self, tmp_path, monkeypatch) -> None:
        # A module that does not exist has no dependencies; that is an answer,
        # not a failure.
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "clara"))
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        payload = call("code_deps", {"target": "nope.ts", "repo": str(fresh)})
        assert payload["indexed"] is False
