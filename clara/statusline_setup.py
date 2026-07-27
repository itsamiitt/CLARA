"""
CLARA — register the memory counter in Claude Code's status bar.

A Claude Code plugin cannot ship a main ``statusLine``: plugin ``settings.json``
accepts only ``agent`` and ``subagentStatusLine``, there is no ``statusLine``
key in ``plugin.json``, and ``${CLAUDE_PLUGIN_ROOT}`` is not expanded inside
settings files. The counter therefore has to be written into the *user's* own
``~/.claude/settings.json``.

That file belongs to the user, so this module is deliberately conservative:
it merges instead of overwriting, backs the file up before touching it, refuses
to displace a status line it did not write unless forced, and reports malformed
JSON rather than clobbering it.

Shared by the ``clara statusline --install`` CLI flag and the
``statusline_install`` MCP tool, so both behave identically — the MCP tool is
the path that matters for users who never see a terminal, since the plugin's
``clara`` executable is intentionally not on their PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Claude Code documents refreshInterval as an integer >= 1 (seconds).
MIN_REFRESH_INTERVAL = 1
DEFAULT_REFRESH_INTERVAL = 5


def settings_path() -> Path:
    """The user settings file Claude Code reads."""
    return Path.home() / ".claude" / "settings.json"


def statusline_command() -> str:
    """Shell command Claude Code should run for the status line.

    Returns an absolute path: settings.json gets no variable expansion, and the
    plugin's virtualenv is deliberately kept off the user's PATH.
    """
    exe = Path(sys.executable)
    candidates = [
        exe.with_name("clara.exe"),
        exe.with_name("clara"),
        exe.parent / "Scripts" / "clara.exe",
        exe.parent / "bin" / "clara",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        target = candidate
        # Prefer the version-independent `current` pointer when this
        # interpreter lives in the plugin's own venv: venv directories are
        # named by a pyproject hash and are garbage-collected on upgrade, so a
        # versioned path would blank the bar at the next plugin update.
        data_dir = Path(
            os.environ.get("CLAUDE_PLUGIN_DATA")
            or (Path(os.environ.get("CLARA_HOME") or (Path.home() / ".clara")) / "plugin")
        )
        try:
            venv_root = candidate.parents[1]
            if venv_root.parent == data_dir and venv_root.name.startswith("venv-"):
                stable = data_dir / "current" / candidate.parent.name / candidate.name
                if stable.is_file():
                    target = stable
        except (IndexError, OSError):
            pass
        return f'"{target}" statusline'
    # No console script (e.g. a source checkout): drive the module through the
    # interpreter itself, which always exists.
    return f'"{exe}" -m clara.cli statusline'


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(settings, error)``; *settings* is ``None`` when unreadable."""
    if not path.is_file():
        return {}, None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    if not raw:
        return {}, None
    try:
        loaded = json.loads(raw)
    except ValueError as exc:
        return None, f"{path} is not valid JSON ({exc}) — fix or remove it, then retry"
    if not isinstance(loaded, dict):
        return None, f"{path} must contain a JSON object"
    return loaded, None


def _save(path: Path, settings: dict[str, Any]) -> str | None:
    """Write *settings* atomically, keeping a backup. Returns an error or None."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            shutil.copyfile(path, path.with_suffix(".json.clara-bak"))
        tmp = path.with_suffix(".json.clara-tmp")
        tmp.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except OSError as exc:
        return f"cannot write {path}: {exc}"
    return None


def _is_clara(block: Any) -> bool:
    return isinstance(block, dict) and "clara" in str(block.get("command", ""))


def install(
    *,
    refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
    force: bool = False,
) -> dict[str, Any]:
    """Add (or update) CLARA's status line. Never raises.

    Returns ``{"ok": bool, "action": str, ...}``; ``action`` is one of
    ``installed``, ``blocked`` (a foreign status line is present and *force*
    was not set), or ``error``.
    """
    path = settings_path()
    settings, error = _load(path)
    if settings is None:
        return {"ok": False, "action": "error", "error": error, "path": str(path)}

    existing = settings.get("statusLine")
    if isinstance(existing, dict) and not _is_clara(existing) and not force:
        return {
            "ok": False,
            "action": "blocked",
            "path": str(path),
            "existing_command": str(existing.get("command", "")),
            "hint": "a different statusLine is configured; pass force=true to replace it",
        }

    command = statusline_command()
    interval = max(MIN_REFRESH_INTERVAL, int(refresh_interval))
    settings["statusLine"] = {
        "type": "command",
        "command": command,
        "refreshInterval": interval,
    }
    error = _save(path, settings)
    if error:
        return {"ok": False, "action": "error", "error": error, "path": str(path)}
    return {
        "ok": True,
        "action": "installed",
        "path": str(path),
        "command": command,
        "refresh_interval": interval,
        "note": "start a new Claude Code session to see the counter",
    }


def uninstall() -> dict[str, Any]:
    """Remove CLARA's status line, leaving anything else untouched."""
    path = settings_path()
    settings, error = _load(path)
    if settings is None:
        return {"ok": False, "action": "error", "error": error, "path": str(path)}

    if not _is_clara(settings.get("statusLine")):
        return {
            "ok": True,
            "action": "absent",
            "path": str(path),
            "note": "no CLARA status line was configured",
        }
    settings.pop("statusLine", None)
    error = _save(path, settings)
    if error:
        return {"ok": False, "action": "error", "error": error, "path": str(path)}
    return {"ok": True, "action": "removed", "path": str(path)}


def status() -> dict[str, Any]:
    """Report whether CLARA's status line is currently configured."""
    path = settings_path()
    settings, error = _load(path)
    if settings is None:
        return {"configured": False, "error": error, "path": str(path)}
    block = settings.get("statusLine")
    return {
        "configured": _is_clara(block),
        "path": str(path),
        "current_command": str(block.get("command", "")) if isinstance(block, dict) else None,
    }
