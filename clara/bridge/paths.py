"""
Native-memory file locations. Stdlib-only.

The auto-memory directory name encodes the project path with every character
outside ``[A-Za-z0-9-]`` replaced by ``-`` (verified against live
``~/.claude/projects/`` entries, e.g. ``C:\\Users\\Ada\\proj`` →
``C--Users-Ada-proj``; the original character case is preserved). The
project path is the git toplevel when inside a repo, else the anchor
directory — mirroring "keyed by git repo, worktrees share".

Respects ``autoMemoryDirectory`` from ``~/.claude/settings.json`` and the
disable switches (``autoMemoryEnabled: false`` /
``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1``): when auto memory is off, export
targets nothing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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


def _claude_settings() -> dict:
    settings = Path.home() / ".claude" / "settings.json"
    try:
        return json.loads(settings.read_text(encoding="utf-8"))
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
        Path.home()
        / ".claude"
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
        Path.home() / ".claude" / "CLAUDE.md",
    ]
    return [p for p in candidates if p.is_file()]
