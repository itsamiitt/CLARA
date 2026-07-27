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
import subprocess
from pathlib import Path
from typing import Any

from clara.store import _GIT_TIMEOUT_S, git_toplevel

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


def _main_worktree_root(anchor: str) -> str | None:
    """Root of the MAIN worktree, from anywhere in the repo.

    Claude Code shares auto memory across worktrees of one repository
    ("Project configs & auto memory now shared across git worktrees of the
    same repository"), so the encoded directory name must key on the main
    worktree. `git rev-parse --show-toplevel` returns the *linked* worktree's
    own path, which gave every worktree its own auto-memory directory:
    measured, main and worktree resolved to ...-main and ...-wt, so a sync run
    inside a worktree wrote where Claude Code never looks.

    --git-common-dir points at the main repo's .git from every worktree; its
    parent is the main worktree root. stdin is closed and the call is bounded:
    an inherited stdin once made git block until timeout on every MCP call.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=anchor,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    common = proc.stdout.strip()
    if not common:
        return None
    root = Path(common).parent
    # A bare repo has no working tree to share memory for.
    return str(root) if root.is_dir() and (root / ".git").exists() else None


def project_root(anchor: str) -> str:
    return _main_worktree_root(anchor) or git_toplevel(anchor) or anchor


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
