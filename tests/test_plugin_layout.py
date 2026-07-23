"""Structural tests for the Claude Code plugin wiring (manifest, hooks, scripts)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SCRIPTS = [
    "scripts/bootstrap.sh",
    "scripts/clara-mcp-launch.sh",
    "scripts/session-start.sh",
]


def _working_bash() -> str | None:
    """A bash that actually runs scripts — the Windows System32 WSL stub
    answers ``bash -c`` with an install prompt, not by executing."""
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return candidate
    return None


class TestManifests:
    def test_plugin_json(self):
        data = json.loads((_ROOT / ".claude-plugin" / "plugin.json").read_text("utf-8"))
        assert data["name"] == "clara"
        assert data["displayName"] == "CLARA Memory"
        # The milestone spec wanted commit-SHA versioning (no version field),
        # but `claude plugin validate --strict` fails without one — see the
        # implementation log for the recorded deviation.
        assert data["version"]
        command = data["mcpServers"]["memory"]["command"]
        assert command == "${CLAUDE_PLUGIN_ROOT}/scripts/clara-mcp-launch.sh"
        assert data["hooks"] == "./hooks/hooks.json"

    def test_marketplace_json(self):
        data = json.loads(
            (_ROOT / ".claude-plugin" / "marketplace.json").read_text("utf-8")
        )
        assert data["name"] == "clara-marketplace"
        (entry,) = data["plugins"]
        assert entry["name"] == "clara"
        assert entry["source"] == "./"

    def test_hooks_json(self):
        data = json.loads((_ROOT / "hooks" / "hooks.json").read_text("utf-8"))
        (session_start,) = data["hooks"]["SessionStart"]
        assert session_start["matcher"] == "startup|resume|compact"
        (hook,) = session_start["hooks"]
        assert hook["type"] == "command"
        assert hook["command"].endswith("/scripts/session-start.sh")
        assert "${CLAUDE_PLUGIN_ROOT}" in hook["command"]

    def test_manifest_in_excludes_plugin_dirs(self):
        text = (_ROOT / "MANIFEST.in").read_text("utf-8")
        for dirname in (".claude-plugin", "hooks", "scripts", "skills", "commands"):
            assert f"prune {dirname}" in text


class TestScripts:
    def test_scripts_exist_and_are_executable_in_git(self):
        proc = subprocess.run(
            ["git", "ls-files", "-s", "scripts"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        modes = {
            line.split()[3]: line.split()[0] for line in proc.stdout.splitlines()
        }
        for script in _SCRIPTS:
            assert modes.get(script) == "100755", f"{script} must be committed 100755"

    @pytest.mark.parametrize("script", _SCRIPTS)
    def test_scripts_parse_as_posix_sh(self, script):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash (Windows WSL stub does not count)")
        proc = subprocess.run(
            [bash, "--posix", "-n", str(_ROOT / script)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


class TestSkillAndCommands:
    def test_skill_frontmatter(self):
        text = (_ROOT / "skills" / "using-clara-memory" / "SKILL.md").read_text("utf-8")
        assert text.startswith("---\n")
        assert "name: using-clara-memory" in text
        assert "description:" in text

    @pytest.mark.parametrize("command", ["remember", "recall", "memories", "forget"])
    def test_command_frontmatter(self, command):
        text = (_ROOT / "commands" / f"{command}.md").read_text("utf-8")
        assert text.startswith("---\n")
        assert "description:" in text
        assert "argument-hint:" in text
        assert "$ARGUMENTS" in text
