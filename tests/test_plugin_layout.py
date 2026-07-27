"""Structural tests for the Claude Code plugin wiring (manifest, hooks, scripts)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SCRIPTS = [
    "scripts/bootstrap.sh",
    "scripts/clara-mcp-launch.sh",
    "scripts/session-start.sh",
    "scripts/session-stop.sh",
    "scripts/read-annotate.sh",
]
_CMD_DISPATCHERS = [
    "scripts/session-start.cmd",
    "scripts/session-stop.cmd",
    "scripts/read-annotate.cmd",
]
_PS1_BODIES = [
    "scripts/win/bootstrap.ps1",
    "scripts/win/session-start.ps1",
    "scripts/win/session-stop.ps1",
    "scripts/win/read-annotate.ps1",
    "scripts/win/clara-mcp-launch.ps1",
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
        # `claude plugin validate --strict` requires a version, and the
        # marketplace auto-update flow keys on it (bump = release).
        assert data["version"]
        # The MCP entry must be a real executable path, NOT a shell script:
        # stdio servers are spawned without a shell on native Windows. The
        # extensionless shim resolves to clara-mcp.exe there (libuv appends
        # .exe) and is a symlink to the venv binary on POSIX.
        command = data["mcpServers"]["memory"]["command"]
        assert command == "${CLAUDE_PLUGIN_DATA}/shim/clara-mcp"
        assert data["hooks"] == "./hooks/hooks.json"

    def test_plugin_version_matches_package(self):
        plugin = json.loads((_ROOT / ".claude-plugin" / "plugin.json").read_text("utf-8"))
        pyproject = (_ROOT / "pyproject.toml").read_text("utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        assert match, "pyproject.toml must declare a version"
        assert plugin["version"] == match.group(1), (
            "plugin.json and pyproject.toml versions must move together"
        )

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
        # clear + fork included: without them the memory block vanishes after
        # /clear and never reaches forked subagent sessions.
        assert session_start["matcher"] == "startup|resume|clear|compact|fork"
        (hook,) = session_start["hooks"]
        assert hook["type"] == "command"
        assert hook["command"].endswith("/scripts/session-start.cmd")
        assert "${CLAUDE_PLUGIN_ROOT}" in hook["command"]
        # Every hook dispatches through a polyglot .cmd (sh on POSIX,
        # PowerShell body on native Windows).
        for event in ("PostToolUse", "Stop"):
            for entry in data["hooks"][event]:
                for h in entry["hooks"]:
                    assert h["command"].endswith(".cmd"), (
                        f"{event} hook must use a .cmd dispatcher"
                    )

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

    def test_python_floor_agrees_everywhere(self):
        """bootstrap.sh, bootstrap.ps1 and pyproject requires-python must
        gate on the same Python floor (3.10)."""
        pyproject = (_ROOT / "pyproject.toml").read_text("utf-8")
        assert 'requires-python = ">=3.10"' in pyproject
        sh = (_ROOT / "scripts" / "bootstrap.sh").read_text("utf-8")
        assert "sys.version_info >= (3, 10)" in sh
        ps1 = (_ROOT / "scripts" / "win" / "bootstrap.ps1").read_text("utf-8")
        assert "sys.version_info >= (3, 10)" in ps1

    def test_sh_bootstrap_finds_the_private_interpreter(self):
        """bootstrap.sh must try $DATA_DIR/pytools before telling the user to
        go install Python.

        bootstrap.ps1 provisions a private CPython there, and this script still
        runs on those same machines: the .cmd dispatchers exec sh, and Git Bash
        users land here directly. Verified before the fix -- with a portable
        interpreter present in the plugin's own data directory and nothing on
        PATH, bootstrap.sh exited 1 with "install Python 3.10+ from
        python.org". The fallback must stay ahead of that message.
        """
        sh = (_ROOT / "scripts" / "bootstrap.sh").read_text("utf-8")
        assert "pytools/python-" in sh, "no private-interpreter fallback"
        fallback_at = sh.index("pytools/python-")
        give_up_at = sh.index("no Python >= 3.10 on PATH (tried")
        assert fallback_at < give_up_at, (
            "the pytools fallback must be attempted before the give-up message"
        )


class TestWindowsDispatch:
    @pytest.mark.parametrize("dispatcher", _CMD_DISPATCHERS)
    def test_cmd_polyglot_shape(self, dispatcher):
        """Line 1 must be the sh branch (`:; exec sh ... #`), the rest the
        batch branch. The trailing '#' comments out a CR under sh."""
        text = (_ROOT / dispatcher).read_text("utf-8")
        lines = text.splitlines()
        assert lines[0].startswith(":; exec sh ")
        assert lines[0].rstrip().endswith("#")
        target = dispatcher.rsplit("/", 1)[1].replace(".cmd", "")
        assert f"{target}.sh" in lines[0]
        assert any(line.strip().lower() == "@echo off" for line in lines)
        assert any("powershell" in line and f"{target}.ps1" in line for line in lines)

    @pytest.mark.parametrize("body", _PS1_BODIES)
    def test_ps1_exists_and_parses(self, body):
        path = _ROOT / body
        assert path.is_file(), f"{body} missing"
        if sys.platform != "win32":
            pytest.skip("PowerShell parse check runs on Windows")
        probe = (
            "$errs = $null; "
            "[System.Management.Automation.PSParser]::Tokenize("
            "(Get-Content -Raw '" + str(path) + "'), [ref]$errs) | Out-Null; "
            "if ($errs.Count -gt 0) { exit 1 }"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"{body} has PowerShell parse errors"

    @pytest.mark.parametrize("dispatcher", _CMD_DISPATCHERS)
    def test_cmd_sh_branch_parses(self, dispatcher):
        """The sh branch of the polyglot (line 1) must be valid sh."""
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        line1 = (_ROOT / dispatcher).read_text("utf-8").splitlines()[0]
        proc = subprocess.run(
            [bash, "--posix", "-n", "-c", line1],
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


class TestSourceFreshness:
    """The installed package must not silently fall behind the checkout.

    Regression: the venv is keyed on a hash of pyproject.toml, which describes
    dependencies. Editing clara/*.py leaves that key unchanged, so bootstrap's
    fast path concluded "venv is current" and kept serving a stale copy —
    users never received code-only updates. Verified in the field: a plugin
    install was missing three modules and a shipped performance fix.
    """

    def _bootstrap(self) -> str:
        return (Path(__file__).parents[1] / "scripts" / "bootstrap.sh").read_text(
            encoding="utf-8"
        )

    def test_bootstrap_tracks_source_separately_from_dependencies(self):
        script = self._bootstrap()
        assert "hash_sources()" in script
        assert "ensure_current_sources()" in script
        # The fast path is the one that used to skip the package entirely.
        assert "ensure_current_sources \"$VENV\" \"$DATA_DIR\"" in script

    def test_helpers_are_defined_before_the_worker_uses_them(self):
        # sh resolves functions at call time; the worker block runs and exits
        # before the foreground section, so a definition below it never loads.
        script = self._bootstrap()
        assert script.index("hash_sources()") < script.index("# Detached worker")

    def test_source_hash_changes_when_a_module_changes(self, tmp_path):
        # Exercise the same hashing rule bootstrap uses, over a fake package.
        def source_hash(root: Path) -> str:
            digest = hashlib.sha256()
            for path in sorted(root.rglob("*.py")):
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
            return digest.hexdigest()[:12]

        pkg = tmp_path / "clara"
        (pkg / "sub").mkdir(parents=True)
        (pkg / "a.py").write_text("x = 1", encoding="utf-8")
        (pkg / "sub" / "b.py").write_text("y = 2", encoding="utf-8")

        before = source_hash(pkg)
        assert source_hash(pkg) == before, "hash must be stable for unchanged input"

        (pkg / "sub" / "b.py").write_text("y = 3", encoding="utf-8")
        assert source_hash(pkg) != before, "a changed module must change the hash"

        # Renames matter too: the path is folded into the digest.
        (pkg / "sub" / "b.py").rename(pkg / "sub" / "c.py")
        assert source_hash(pkg) != before


class TestReadHookGating:
    """PostToolUse(Read) fires on every file read, so its cost is multiplied.

    Starting PowerShell costs ~500 ms on native Windows (measured 412-1127 ms),
    and it was paid on every Read even though the hook usually has nothing to
    say. An annotation can only come from a quarantine manifest under the CLARA
    home, so when no manifest exists anywhere the dispatcher can skip the
    interpreter entirely -- a repo-independent check that is free in cmd.exe.
    """

    def _dispatcher(self) -> str:
        return (
            Path(__file__).parents[1] / "scripts" / "read-annotate.cmd"
        ).read_text(encoding="utf-8")

    def test_polyglot_first_line_is_preserved(self):
        # The first line must still hand POSIX shells off to the .sh script;
        # everything added for cmd.exe has to live below it.
        assert self._dispatcher().splitlines()[0].startswith(":;")

    def test_gates_on_the_quarantine_manifests(self):
        script = self._dispatcher()
        assert "quarantine" in script
        assert "CLARA_HOME" in script, "must honour a relocated CLARA home"
        assert "USERPROFILE" in script, "must fall back to the default home"

    def test_drains_stdin_before_the_early_exit(self):
        # The host writes hook JSON into this process; exiting with it unread
        # can break that write, which is why the shell paths drain it too.
        script = self._dispatcher()
        early = script.split("powershell", 1)[0]
        assert "more > nul" in early

    @pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe dispatcher")
    def test_skips_dispatch_when_no_manifest_exists(self, tmp_path, monkeypatch):
        script = Path(__file__).parents[1] / "scripts" / "read-annotate.cmd"
        env = {**os.environ, "CLARA_HOME": str(tmp_path / "clara-home")}
        payload = json.dumps(
            {"tool_input": {"file_path": str(tmp_path / "some.md")}}
        )
        proc = subprocess.run(
            ["cmd", "/c", str(script)],
            input=payload, env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    @pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe dispatcher")
    def test_dispatches_when_a_manifest_exists(self, tmp_path, monkeypatch):
        # With a manifest present the dispatcher must hand off rather than
        # short-circuit; the PowerShell hook then applies its own checks.
        home = tmp_path / "clara-home"
        (home / "quarantine").mkdir(parents=True)
        (home / "quarantine" / "abc123.tsv").write_text("x", encoding="utf-8")
        script = Path(__file__).parents[1] / "scripts" / "read-annotate.cmd"
        env = {**os.environ, "CLARA_HOME": str(home)}
        payload = json.dumps(
            {"tool_input": {"file_path": str(tmp_path / "some.md")}}
        )
        proc = subprocess.run(
            ["cmd", "/c", str(script)],
            input=payload, env=env, capture_output=True, text=True, timeout=60,
        )
        # Fail-open either way; what matters is that it did not error out.
        assert proc.returncode == 0


class TestStopHookGating:
    """Stop fires once per TURN, not once per session.

    Twenty user messages start PowerShell twenty times, ~537 ms each on native
    Windows (measured 438-1248 ms), almost always to conclude there is nothing
    to say. The nudge can only come from a proposals sidecar under the CLARA
    home, so the dispatcher checks for that first -- repo-independent, and free
    in cmd.exe.
    """

    def _dispatcher(self) -> str:
        return (
            Path(__file__).parents[1] / "scripts" / "session-stop.cmd"
        ).read_text(encoding="utf-8")

    def test_polyglot_first_line_is_preserved(self):
        assert self._dispatcher().splitlines()[0].startswith(":;")

    def test_gates_on_the_proposals_directory(self):
        script = self._dispatcher()
        assert "proposals" in script
        assert "CLARA_HOME" in script, "must honour a relocated CLARA home"
        assert "USERPROFILE" in script, "must fall back to the default home"

    @pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe dispatcher")
    def test_skips_dispatch_without_proposals(self, tmp_path):
        script = Path(__file__).parents[1] / "scripts" / "session-stop.cmd"
        env = {**os.environ, "CLARA_HOME": str(tmp_path / "clara-home")}
        proc = subprocess.run(
            ["cmd", "/c", str(script)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    @pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe dispatcher")
    def test_dispatches_when_proposals_exist(self, tmp_path):
        home = tmp_path / "clara-home"
        (home / "proposals").mkdir(parents=True)
        script = Path(__file__).parents[1] / "scripts" / "session-stop.cmd"
        env = {**os.environ, "CLARA_HOME": str(home)}
        proc = subprocess.run(
            ["cmd", "/c", str(script)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0


class TestSlashCommandsAreDiscoverable:
    """Every shipped slash command must be documented, and vice versa.

    `/clara:statusline` shipped for a while without appearing in the README
    table, so the only way to find it was to list the plugin directory. A
    command nobody knows about is a command nobody runs.
    """

    def _shipped(self) -> set[str]:
        return {p.stem for p in (_ROOT / "commands").glob("*.md")}

    def _documented(self) -> set[str]:
        import re

        readme = (_ROOT / "README.md").read_text("utf-8")
        return set(re.findall(r"`/clara:([a-z-]+)", readme))

    def test_every_shipped_command_is_documented(self):
        missing = self._shipped() - self._documented()
        assert not missing, f"shipped but undocumented: {sorted(missing)}"

    def test_every_documented_command_exists(self):
        phantom = self._documented() - self._shipped()
        assert not phantom, (
            f"documented but not shipped: {sorted(phantom)} — the README "
            "promises a command the plugin does not provide"
        )

    def test_commands_that_shell_out_explain_where_the_cli_lives(self):
        """A plugin-only install never puts `clara` on PATH.

        The bootstrap keeps the CLI in the plugin's private venv. Commands that
        tell Claude to run `clara ...` with Bash must also say where to find it,
        or they fail with "command not found" for exactly the users who
        installed the recommended way.
        """
        offenders = []
        for path in (_ROOT / "commands").glob("*.md"):
            text = path.read_text("utf-8")
            shells_out = "`clara " in text
            explains = "shim/clara" in text
            if shells_out and not explains:
                offenders.append(path.name)
        assert not offenders, (
            "these slash commands invoke the clara CLI without saying where it "
            f"lives for a plugin install: {sorted(offenders)}"
        )


class TestHooksUseClaudeProjectDir:
    """Hooks must use the project dir Claude Code hands them, not their cwd.

    Claude Code's changelog: "Hooks: Added CLAUDE_PROJECT_DIR env var for hook
    commands", and later "MCP stdio servers now receive CLAUDE_PROJECT_DIR in
    their environment, matching hooks". CLARA walked up from $PWD instead.
    Verified before the fix: with the hook's cwd outside the repo and
    CLAUDE_PROJECT_DIR pointing at it, the quarantine annotation was silently
    dropped.

    Relativisation had the same flaw: it measured the read file against $PWD,
    so a read while cwd was a *subdirectory* produced the wrong relative path.
    """

    def _hook_source(self, name: str) -> str:
        return (_ROOT / "scripts" / name).read_text("utf-8")

    @pytest.mark.parametrize("hook", ["read-annotate.sh", "session-stop.sh"])
    def test_sh_hooks_prefer_the_project_dir(self, hook):
        source = self._hook_source(hook)
        assert "CLAUDE_PROJECT_DIR" in source, f"{hook} still guesses from cwd"
        # ...and still works when the variable is absent.
        assert ":-$PWD" in source, f"{hook} must fall back to cwd"

    @pytest.mark.parametrize("hook", ["read-annotate.ps1", "session-stop.ps1"])
    def test_ps1_hooks_prefer_the_project_dir(self, hook):
        source = (_ROOT / "scripts" / "win" / hook).read_text("utf-8")
        assert "CLAUDE_PROJECT_DIR" in source, f"{hook} still guesses from cwd"
        assert "Get-Location" in source, f"{hook} must fall back to cwd"

    def test_relativisation_uses_the_discovered_root(self):
        """Not $PWD — a read from a subdirectory would otherwise mis-relativise."""
        source = self._hook_source("read-annotate.sh")
        assert 'CLARA_HK_ROOT_B="$root"' in source
        assert 'CLARA_HK_ROOT_B="$PWD"' not in source

    def test_backslash_normalisation_is_portable(self):
        """A lone backslash as a tr operand is undefined in POSIX; BSD tr on
        macOS need not accept it, and GNU tr warns."""
        backslash = chr(92)
        portable = "tr " + chr(39) + backslash * 2 + chr(39)   # tr '\'
        lone = "tr " + chr(39) + backslash + chr(39) + " "     # tr '' followed by a space
        for hook in ("read-annotate.sh", "session-stop.sh"):
            source = self._hook_source(hook)
            assert portable in source, f"{hook} does not escape the tr operand"
            assert lone not in source, f"{hook} uses a lone backslash operand"


class TestMcpServerUsesClaudeProjectDir:
    def test_anchor_falls_back_to_project_dir_before_cwd(self):
        import inspect

        from clara.integrations import mcp_server

        source = inspect.getsource(mcp_server._session_anchor)
        assert "CLAUDE_PROJECT_DIR" in source
        # It must rank below the signals that track a mid-session cd.
        assert source.index("list_roots") < source.index("CLAUDE_PROJECT_DIR")
        assert source.index("session-cwd") < source.index("CLAUDE_PROJECT_DIR")
        assert source.index("CLAUDE_PROJECT_DIR") < source.index("os.getcwd()")
