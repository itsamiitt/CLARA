"""Tests for `clara statusline --install/--uninstall`.

This command edits a file the user owns (~/.claude/settings.json), so the
behaviours that matter are: never lose unrelated keys, never silently replace
someone else's statusLine, and always leave valid JSON behind.

A plugin cannot ship a `statusLine` (Claude Code accepts only
`subagentStatusLine` from plugin settings), which is why this writer exists.
"""

from __future__ import annotations

import argparse
import json

import pytest

from clara.cli import _cmd_statusline_install, _statusline_command


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _args(**kwargs) -> argparse.Namespace:
    base = {
        "install": True,
        "uninstall": False,
        "refresh_interval": 5,
        "force": False,
        "no_stdin": True,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def _settings(home) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))


class TestInstall:
    def test_creates_settings_when_absent(self, fake_home):
        assert _cmd_statusline_install(_args()) == 0
        block = _settings(fake_home)["statusLine"]
        assert block["type"] == "command"
        assert "statusline" in block["command"]
        assert block["refreshInterval"] == 5

    def test_preserves_unrelated_keys(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        path.write_text(json.dumps({"theme": "dark", "model": "opus"}), encoding="utf-8")

        assert _cmd_statusline_install(_args()) == 0

        data = _settings(fake_home)
        assert data["theme"] == "dark"
        assert data["model"] == "opus"
        assert "statusLine" in data

    def test_refresh_interval_is_honoured_and_floored(self, fake_home):
        assert _cmd_statusline_install(_args(refresh_interval=30)) == 0
        assert _settings(fake_home)["statusLine"]["refreshInterval"] == 30
        # Claude Code documents refreshInterval >= 1; never emit 0.
        assert _cmd_statusline_install(_args(refresh_interval=0)) == 0
        assert _settings(fake_home)["statusLine"]["refreshInterval"] == 1

    def test_backup_written_when_file_existed(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

        _cmd_statusline_install(_args())

        backup = fake_home / ".claude" / "settings.json.clara-bak"
        assert backup.is_file()
        assert json.loads(backup.read_text(encoding="utf-8")) == {"theme": "dark"}

    def test_is_idempotent(self, fake_home):
        _cmd_statusline_install(_args())
        first = _settings(fake_home)
        _cmd_statusline_install(_args())
        assert _settings(fake_home) == first

    def test_no_temp_file_left_behind(self, fake_home):
        _cmd_statusline_install(_args())
        assert list((fake_home / ".claude").glob("*clara-tmp*")) == []


class TestRefusesToClobber:
    def test_foreign_statusline_requires_force(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        foreign = {"statusLine": {"type": "command", "command": "/usr/bin/mybar"}}
        path.write_text(json.dumps(foreign), encoding="utf-8")

        assert _cmd_statusline_install(_args()) == 2
        # Untouched.
        assert _settings(fake_home)["statusLine"]["command"] == "/usr/bin/mybar"

    def test_force_replaces_foreign_statusline(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        foreign = {"statusLine": {"type": "command", "command": "/usr/bin/mybar"}}
        path.write_text(json.dumps(foreign), encoding="utf-8")

        assert _cmd_statusline_install(_args(force=True)) == 0
        assert "statusline" in _settings(fake_home)["statusLine"]["command"]

    def test_malformed_settings_is_reported_not_overwritten(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        path.write_text("{ this is not json", encoding="utf-8")

        assert _cmd_statusline_install(_args()) == 2
        # The user's file must survive verbatim so they can repair it.
        assert path.read_text(encoding="utf-8") == "{ this is not json"


class TestUninstall:
    def test_removes_only_clara_entry(self, fake_home):
        _cmd_statusline_install(_args())
        assert _cmd_statusline_install(_args(install=False, uninstall=True)) == 0
        assert "statusLine" not in _settings(fake_home)

    def test_keeps_other_keys(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        _cmd_statusline_install(_args())

        _cmd_statusline_install(_args(install=False, uninstall=True))

        assert _settings(fake_home)["theme"] == "dark"

    def test_leaves_foreign_statusline_alone(self, fake_home):
        path = fake_home / ".claude" / "settings.json"
        foreign = {"statusLine": {"type": "command", "command": "/usr/bin/mybar"}}
        path.write_text(json.dumps(foreign), encoding="utf-8")

        assert _cmd_statusline_install(_args(install=False, uninstall=True)) == 0
        assert _settings(fake_home)["statusLine"]["command"] == "/usr/bin/mybar"


class TestCommandResolution:
    def test_command_is_quoted_and_absolute(self):
        command = _statusline_command()
        assert command.startswith('"')
        assert command.endswith("statusline")
