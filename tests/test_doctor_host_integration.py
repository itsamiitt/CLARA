"""
doctor must check the host wiring — installed is not enabled.

Diagnosed on a real machine: clara@clara-marketplace sat in
installed_plugins.json while no settings file enabled it, so no hook fired
and no memory tool loaded — and doctor reported every store check ok, because
nothing looked at the host. A second blindness on the same machine: the
project was pinned to an old plugin cache whose hooks.json had no
UserPromptSubmit, so per-prompt recall silently did not exist.

The host is faked through CLAUDE_CONFIG_DIR (the same seam
tests/test_claude_config_dir.py uses).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

MARKETPLACE_KEY = "clara@clara-marketplace"


def pkg_version(cwd: Path) -> str:
    """clara-memory's version as the doctor subprocess will see it.

    With an editable install the answer is cwd-dependent: the repo root's
    egg-info can carry a different version than the site-packages metadata,
    so the expected pin must be read from the same cwd doctor runs in.
    """
    done = subprocess.run(
        [sys.executable, "-c",
         "from importlib.metadata import version; print(version('clara-memory'))"],
        capture_output=True, text=True, encoding="utf-8", cwd=cwd,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()

ALL_HOOKS = ("SessionStart", "PostToolUse", "UserPromptSubmit", "Stop")


def doctor(tmp_path: Path, config_dir: Path, cwd: Path | None = None):
    env = {
        **os.environ,
        "CLARA_HOME": str(tmp_path / "clara-home"),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }
    env.pop("CLAUDE_PLUGIN_DATA", None)
    done = subprocess.run(
        [sys.executable, "-m", "clara.cli", "doctor"],
        capture_output=True, text=True, encoding="utf-8",
        env=env, cwd=cwd or tmp_path,
    )
    assert done.returncode in (0, 1), done.stderr[-300:]
    return done


def fake_host(
    tmp_path: Path,
    *,
    enabled: bool,
    pin_version: str | None = None,
    hook_events: tuple[str, ...] = ALL_HOOKS,
    project: Path | None = None,
) -> Path:
    """A Claude Code config dir with clara installed for *project*."""
    config = tmp_path / "claude-config"
    (config / "plugins").mkdir(parents=True)
    if pin_version is None:
        pin_version = pkg_version(tmp_path)

    cache = config / "plugins" / "cache" / "clara-marketplace" / "clara" / pin_version
    (cache / "hooks").mkdir(parents=True)
    hooks = {event: [] for event in hook_events}
    (cache / "hooks" / "hooks.json").write_text(
        json.dumps({"hooks": hooks}), encoding="utf-8"
    )

    entry = {
        "scope": "project",
        "projectPath": str(project or tmp_path),
        "installPath": str(cache),
        "version": pin_version,
    }
    (config / "plugins" / "installed_plugins.json").write_text(
        json.dumps({"version": 2, "plugins": {MARKETPLACE_KEY: [entry]}}),
        encoding="utf-8",
    )

    settings: dict = {}
    if enabled:
        settings["enabledPlugins"] = {MARKETPLACE_KEY: True}
    (config / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return config


class TestEnablement:
    def test_installed_and_enabled_is_ok(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=True)
        out = doctor(tmp_path, config).stdout
        assert f"plugin enabled: {MARKETPLACE_KEY}" in out
        assert "not enabled" not in out

    def test_installed_but_not_enabled_warns_and_degrades(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=False)
        done = doctor(tmp_path, config)
        assert "installed but not enabled" in done.stdout
        assert "what to do:" in done.stdout
        assert done.returncode == 1

    def test_project_level_enablement_counts(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=False)
        dot_claude = tmp_path / ".claude"
        dot_claude.mkdir()
        (dot_claude / "settings.json").write_text(
            json.dumps({"enabledPlugins": {MARKETPLACE_KEY: True}}),
            encoding="utf-8",
        )
        out = doctor(tmp_path, config).stdout
        assert "not enabled" not in out


class TestVersionPin:
    def test_matching_pin_is_silent(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=True)
        out = doctor(tmp_path, config).stdout
        assert "pinned to plugin cache" not in out

    def test_stale_pin_warns(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=True, pin_version="0.0.1")
        done = doctor(tmp_path, config)
        assert "pinned to plugin cache 0.0.1" in done.stdout
        assert "/plugin update clara" in done.stdout
        assert done.returncode == 1


class TestHookRegistration:
    def test_missing_prompt_recall_hook_is_named(self, tmp_path) -> None:
        # The 0.2.0 cache's exact shape: no UserPromptSubmit registration.
        config = fake_host(
            tmp_path, enabled=True,
            hook_events=("SessionStart", "PostToolUse", "Stop"),
        )
        out = doctor(tmp_path, config).stdout
        assert "registers no UserPromptSubmit" in out
        assert "no per-prompt recall" in out

    def test_full_registration_is_ok(self, tmp_path) -> None:
        config = fake_host(tmp_path, enabled=True)
        out = doctor(tmp_path, config).stdout
        assert "registers all hook events" in out


class TestSilence:
    def test_no_host_dir_no_noise(self, tmp_path) -> None:
        out = doctor(tmp_path, tmp_path / "does-not-exist").stdout
        assert "host:" not in out

    def test_no_clara_install_no_noise(self, tmp_path) -> None:
        config = tmp_path / "claude-config"
        (config / "plugins").mkdir(parents=True)
        (config / "plugins" / "installed_plugins.json").write_text(
            json.dumps({"version": 2, "plugins": {"other@place": []}}),
            encoding="utf-8",
        )
        out = doctor(tmp_path, config).stdout
        assert "host:" not in out
