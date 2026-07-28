"""
CLARA fastpath — change capture (PostToolUse hook on Edit/Write/Bash).

The change journal and its indexer worker have existed since the code index
shipped, but nothing fed the journal during a session: the daily maintenance
pass did a full reconciling walk and everything between passes was invisible.
This hook is the missing event source from the memory-systems plan §4.1 —
one cheap INSERT per observed change, no parsing, no indexing:

* Edit/Write/MultiEdit/NotebookEdit → enqueue the file as ``modified``.
* Bash → classify the command; a git mutation enqueues repo-level ``git``,
  a package-manager change enqueues repo-level ``manifest``.

Draining happens elsewhere (the Stop-hook flush and the daily maintenance
pass). What this hook captures is *code and repo events* — conversational
facts remain the model's job through memory_save; nothing here reads the
conversation.

Discipline, same contract as prompt_recall:

* **Silence on stdout.** PostToolUse stdout is not model-visible on exit 0,
  and this must never add noise to a turn.
* **Never blocks, fail-open everywhere.** Exit 0 on every path; a short busy
  timeout, and a locked store simply drops the event — the daily walk
  reconciles anything missed.
* **Writes only the journal** (plus a dirty-flag sidecar so the Stop
  dispatcher can gate for free), through the same resolver every other
  surface uses.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from clara.db.migrations import SchemaTooNew, check_version
from clara.index import journal
from clara.index.indexer import INDEXABLE_SUFFIXES
from clara.repoid import repo_id
from clara.store import git_toplevel, resolve_store

_BUSY_TIMEOUT_MS = 500  # an event source must never make the user wait

_FILE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

# Bash commands worth a repo-level journal entry. Deliberately coarse: the
# journal is a queue of "look here again", not a parser — false positives
# cost one no-op claim, false negatives wait for the daily walk.
_GIT_RE = re.compile(
    r"\bgit\b[^|;&]*\b(commit|merge|rebase|pull|checkout|switch|reset|"
    r"cherry-pick|revert|apply|am|mv|rm|stash)\b"
)
_MANIFEST_RE = re.compile(
    r"\b(?:pnpm|npm|yarn|bun)\b[^|;&]*\b(?:add|install|remove|uninstall|update|up)\b"
    r"|\b(?:pip3?|uv)\b[^|;&]*\b(?:install|uninstall)\b"
    r"|\bpoetry\b[^|;&]*\b(?:add|remove|update|install|lock)\b"
    r"|\bcargo\b[^|;&]*\b(?:add|remove)\b"
    r"|\bgo\s+get\b"
)


def classify_bash(command: str) -> str | None:
    """``git`` / ``manifest`` / None for one Bash command line."""
    if _MANIFEST_RE.search(command):
        return "manifest"
    if _GIT_RE.search(command):
        return "git"
    return None


def _clara_base() -> Path:
    return Path(os.environ.get("CLARA_HOME") or Path.home() / ".clara")


def mark_dirty(rid: str) -> None:
    """Touch the sidecar the Stop dispatcher gates on — free to test in cmd."""
    try:
        flag_dir = _clara_base() / "journal-dirty"
        flag_dir.mkdir(parents=True, exist_ok=True)
        (flag_dir / rid).touch()
    except OSError:
        pass  # gating is an optimisation; the daily walk still reconciles


def _relative_source_path(raw: str, root: str) -> str | None:
    """*raw* as a repo-relative posix path, or None when it is not an
    indexable file inside *root*."""
    try:
        rel = Path(raw).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return None
    posix = rel.as_posix()
    if rel.suffix not in INDEXABLE_SUFFIXES:
        return None
    return posix


def capture(payload: dict[str, object], cwd: str) -> bool:
    """Enqueue what *payload* describes. True when something was written."""
    tool = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}

    root = git_toplevel(cwd)
    if not root:
        return False

    change: str | None = None
    rel_path: str | None = None
    if tool in _FILE_TOOLS:
        raw = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not isinstance(raw, str) or not raw:
            return False
        rel_path = _relative_source_path(raw, root)
        if rel_path is None:
            return False
        change = "modified"
    elif tool == "Bash":
        command = tool_input.get("command")
        if not isinstance(command, str):
            return False
        change = classify_bash(command)
        if change is None:
            return False
    else:
        return False

    resolution = resolve_store(cwd, create=False)
    if not resolution.exists:
        return False
    rid = repo_id(cwd)

    conn = sqlite3.connect(resolution.db_path)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            check_version(conn)
        except SchemaTooNew:
            return False
        journal.enqueue(conn, rid, change=change, path=rel_path)
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    mark_dirty(rid)
    return True


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(payload, dict):
        return 0
    cwd = str(
        payload.get("cwd")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.getcwd()
    )
    try:
        capture(payload, cwd)
    except Exception as exc:  # noqa: BLE001 — a hook failure must stay invisible
        print(f"clara change-capture: skipped ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
