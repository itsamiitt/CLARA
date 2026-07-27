"""
Native-memory file locations. Stdlib-only.

The auto-memory directory name encodes the project path with every character
outside ``[A-Za-z0-9-]`` replaced by ``-`` (verified against live
``~/.claude/projects/`` entries, e.g. ``C:\\Users\\Ada\\proj`` →
``C--Users-Ada-proj``; the original character case is preserved). The
project path is the git toplevel when inside a repo, else the anchor
directory — mirroring "keyed by git repo, worktrees share".

Respects ``autoMemoryDirectory`` from the Claude config dir
(``$CLAUDE_CONFIG_DIR`` or ``~/.claude``) and the
disable switches (``autoMemoryEnabled: false`` /
``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1``): when auto memory is off, export
targets nothing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from clara.store import git_toplevel

_ENCODE_RE = re.compile(r"[^A-Za-z0-9-]")

CLARA_TOPIC_FILE = "clara-memory.md"


def long_path(path: Path) -> Path:
    """Extended-length form on Windows: encoded project-dir names are long,
    and stock MAX_PATH (260) trips on deep repo paths."""
    raw = str(path)
    if os.name == "nt" and len(raw) > 240 and not raw.startswith("\\\\?\\"):
        return Path("\\\\?\\" + os.path.abspath(raw))
    return path


def encode_project_dir(project_path: str) -> str:
    return _ENCODE_RE.sub("-", project_path)


def claude_config_dir() -> Path:
    """Claude Code's config directory: ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``.

    Claude Code lets users relocate it, and its own changelog records
    "Respect CLAUDE_CONFIG_DIR everywhere" plus later fixes for spots that did
    not. CLARA hardcoded ``~/.claude``, so with the variable set every native
    file went somewhere Claude Code never reads: `clara sync` reported
    "MEMORY.md updated" while writing to a directory nothing consumes, and the
    settings lookup (autoMemoryDirectory, autoMemoryEnabled) silently read
    defaults from the wrong file.
    """
    override = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".claude"


def _claude_settings() -> dict[str, Any]:
    settings = claude_config_dir() / "settings.json"
    try:
        loaded = json.loads(settings.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def auto_memory_enabled() -> bool:
    if os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "").strip() == "1":
        return False
    return _claude_settings().get("autoMemoryEnabled", True) is not False


def project_root(anchor: str) -> str:
    return git_toplevel(anchor) or anchor


def auto_memory_dir(anchor: str) -> Path | None:
    """The auto-memory directory for *anchor*'s project, or ``None`` when
    auto memory is disabled. Does not create anything."""
    if not auto_memory_enabled():
        return None
    override = _claude_settings().get("autoMemoryDirectory")
    if isinstance(override, str) and override.strip():
        base = Path(os.path.expanduser(override))
        return base / encode_project_dir(project_root(anchor)) / "memory"
    return (
        claude_config_dir()
        / "projects"
        / encode_project_dir(project_root(anchor))
        / "memory"
    )


def memory_md_path(anchor: str) -> Path | None:
    directory = auto_memory_dir(anchor)
    return None if directory is None else directory / "MEMORY.md"


def topic_file_path(anchor: str) -> Path | None:
    directory = auto_memory_dir(anchor)
    return None if directory is None else directory / CLARA_TOPIC_FILE


def claude_md_paths(anchor: str) -> list[Path]:
    """Import sources: project CLAUDE.md variants + the user-level file."""
    root = Path(project_root(anchor))
    candidates = [
        root / "CLAUDE.md",
        root / ".claude" / "CLAUDE.md",
        root / "CLAUDE.local.md",
        claude_config_dir() / "CLAUDE.md",
    ]
    return [p for p in candidates if p.is_file()]
