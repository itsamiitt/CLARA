"""
Uninstall must take the status line with it — and only its own.

The status line is the one thing CLARA writes outside its own data directory.
A real plugin install registers
"<CLAUDE_PLUGIN_DATA>/current/Scripts/clara.exe statusline" (verified against a
real plugin venv), and uninstall deletes that directory. Leaving the entry
behind meant Claude Code ran a deleted binary every session, and the only way
to stop it was hand-editing settings.json — the exact thing installing a plugin
is supposed to spare someone.

The risk in fixing that is overreach, so the other half matters just as much:
a status line the user configured themselves must survive untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from clara.cli import _cmd_uninstall


def uninstall(purge=False):
    return asyncio.run(_cmd_uninstall(argparse.Namespace(purge_memories=purge)))


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "home").mkdir()
    (tmp_path / "cfg").mkdir()
    return tmp_path


def settings(home):
    return home / "cfg" / "settings.json"


def write_settings(home, payload):
    settings(home).write_text(json.dumps(payload), encoding="utf-8")


class TestClarasOwnStatusLine:
    def test_it_is_removed(self, home) -> None:
        from clara import statusline_setup

        assert statusline_setup.install()["ok"] is True
        assert "statusLine" in json.loads(settings(home).read_text("utf-8"))

        assert uninstall() == 0
        after = json.loads(settings(home).read_text(encoding="utf-8"))
        assert "statusLine" not in after

    def test_removal_is_reported(self, home, capsys) -> None:
        from clara import statusline_setup

        statusline_setup.install()
        uninstall()
        out = capsys.readouterr().out
        # Deleting something outside the data directory without saying so
        # would be worse than leaving it.
        assert "statusLine" in out


class TestEverythingElseSurvives:
    def test_a_users_own_status_line_is_left_alone(self, home) -> None:
        write_settings(home, {"statusLine": {"type": "command",
                                             "command": "my-own-prompt.sh"}})
        assert uninstall() == 0
        after = json.loads(settings(home).read_text(encoding="utf-8"))
        assert after["statusLine"]["command"] == "my-own-prompt.sh"

    def test_unrelated_settings_keys_are_left_alone(self, home) -> None:
        from clara import statusline_setup

        write_settings(home, {"model": "opus", "env": {"KEEP": "yes"},
                              "permissions": {"allow": ["Bash(ls:*)"]}})
        statusline_setup.install()
        uninstall()
        after = json.loads(settings(home).read_text(encoding="utf-8"))
        assert after["model"] == "opus"
        assert after["env"] == {"KEEP": "yes"}
        assert after["permissions"] == {"allow": ["Bash(ls:*)"]}

    def test_no_settings_file_is_not_an_error(self, home) -> None:
        assert not settings(home).exists()
        assert uninstall() == 0

    def test_corrupt_settings_does_not_fail_the_uninstall(self, home) -> None:
        # The runtime still has to come off even if the settings file cannot
        # be parsed; the warning goes to stderr and the exit code stays 0.
        original = "{ not json"
        settings(home).write_text(original, encoding="utf-8")
        assert uninstall() == 0
        assert settings(home).read_text(encoding="utf-8") == original


class TestMemoriesAreStillSafe:
    def test_default_keeps_the_store(self, home) -> None:
        store = home / "home" / "clara.db"
        store.write_bytes(b"not really a database, but a file that must remain")
        assert uninstall() == 0
        assert store.exists()

    def test_purge_removes_it_only_when_asked(self, home) -> None:
        store = home / "home" / "clara.db"
        store.write_bytes(b"x")
        assert uninstall(purge=True) == 0
        assert not store.exists()
