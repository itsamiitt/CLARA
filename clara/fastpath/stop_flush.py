"""
CLARA fastpath — Stop-hook journal flush.

change_capture enqueues during the session; without this, nothing drains the
queue until the daily maintenance pass. One bounded indexing cycle at Stop
keeps the code index hot without a daemon: claim small batches, respect a
wall-clock budget, and leave the rest for the next Stop or the daily walk.

Also reconciles commits made outside any session: a cursor sidecar remembers
the last flushed HEAD, and when HEAD moved, ``git diff --name-status`` between
the two enqueues exactly what changed — the same delta the plan's SessionStart
enqueue wanted, moved here so session start pays nothing for it.

Contract: exit 0 always, silent stdout (Stop stdout is not model-visible
feedback), writable store access through the shared resolver, and every
failure is a skipped nicety, never an error the user sees.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from clara.db.migrations import SchemaTooNew, check_version
from clara.index import journal
from clara.index.indexer import drain_journal
from clara.repoid import repo_id
from clara.store import _GIT_TIMEOUT_S, git_toplevel, resolve_store

_BUSY_TIMEOUT_MS = 3_000
_TIME_BUDGET_S = 2.0
_BATCH = 25

_STATUS_TO_CHANGE = {"A": "added", "M": "modified", "D": "removed"}


def _clara_base() -> Path:
    return Path(os.environ.get("CLARA_HOME") or Path.home() / ".clara")


def _cursor_file(rid: str, root: str) -> Path:
    # Keyed by repo_id AND checkout path: repo_id is the root-commit hash,
    # shared by every clone/worktree of one repository, and a cursor shared
    # between two checkouts at different HEADs would ping-pong diffs between
    # them. The path hash keeps one cursor per checkout.
    checkout = hashlib.sha256(
        str(Path(root).resolve()).casefold().encode("utf-8", "replace")
    ).hexdigest()[:12]
    return _clara_base() / "journal-cursor" / f"{rid}-{checkout}"


def _git_head(root: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and head else None


def enqueue_head_delta(conn: sqlite3.Connection, rid: str, root: str) -> str | None:
    """Journal what changed between the cursor's HEAD and the current one.

    Returns the new HEAD to record, or None when there is nothing to move the
    cursor to. A missing cursor writes the current HEAD without a diff — the
    first flush must not replay the repository's whole history.
    """
    head = _git_head(root)
    if head is None:
        return None
    cursor = None
    try:
        cursor = _cursor_file(rid, root).read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    if cursor is None or cursor == head:
        return head
    try:
        proc = subprocess.run(
            ["git", "-C", root, "diff", "--name-status", cursor, head],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # Transient failure (timeout, spawn error): do NOT advance the
        # cursor — the delta is still owed, and the next Stop retries it.
        return None
    if proc.returncode != 0:
        # An unknown cursor (rebase, gc) has no diff; restart from here.
        return head
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][:1]
        try:
            if status == "R" and len(parts) >= 3:
                journal.enqueue(
                    conn, rid, change="renamed", path=parts[2], old_path=parts[1]
                )
            elif status in _STATUS_TO_CHANGE:
                journal.enqueue(
                    conn, rid, change=_STATUS_TO_CHANGE[status], path=parts[1]
                )
        except (ValueError, sqlite3.Error):
            continue
    return head


_FLAG_MAX_AGE_S = 7 * 24 * 3600


def _sweep_stale_flags() -> None:
    """Drop journal-dirty flags nothing has cleared in a week.

    The dirty gate is machine-wide but each flush clears only its own repo's
    flag, so a flag whose repository is never opened again would force an
    interpreter start on every Stop in every project forever. A week is long
    past any daily maintenance walk that would have reconciled the work.
    """
    try:
        cutoff = time.time() - _FLAG_MAX_AGE_S
        for flag in (_clara_base() / "journal-dirty").iterdir():
            try:
                if flag.is_file() and flag.stat().st_mtime < cutoff:
                    flag.unlink()
            except OSError:
                continue
    except OSError:
        pass


def flush(cwd: str, session_id: str) -> None:
    root = git_toplevel(cwd)
    if not root:
        return
    resolution = resolve_store(cwd, create=False)
    if not resolution.exists:
        return
    rid = repo_id(cwd)

    conn = sqlite3.connect(resolution.db_path)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            check_version(conn)
        except SchemaTooNew:
            return
        new_head = enqueue_head_delta(conn, rid, root)

        journal.release_stale_claims(conn)
        # release_stale_claims leaves its UPDATE in sqlite3's implicit
        # transaction; commit it so a zero-pending early path cannot roll
        # the releases back on close.
        conn.commit()
        deadline = time.monotonic() + _TIME_BUDGET_S
        worker = f"stop-flush-{session_id or os.getpid()}"
        while journal.pending_count(conn, rid) > 0:
            drain_journal(conn, rid, Path(root), worker=worker, limit=_BATCH)
            if time.monotonic() >= deadline:
                break

        if new_head is not None:
            try:
                cursor = _cursor_file(rid, root)
                cursor.parent.mkdir(parents=True, exist_ok=True)
                cursor.write_text(new_head, encoding="utf-8")
            except OSError:
                pass
        if journal.pending_count(conn, rid) == 0:
            try:
                (_clara_base() / "journal-dirty" / rid).unlink()
            except OSError:
                pass
        _sweep_stale_flags()
    except sqlite3.Error:
        return
    finally:
        conn.close()


def main() -> int:
    # Stop hooks are not reliably given stdin by every host version: read it
    # if present, fall back to the environment.
    payload: dict[str, object] = {}
    try:
        if not sys.stdin.isatty():
            payload = json.loads(sys.stdin.read() or "{}")
            if not isinstance(payload, dict):
                payload = {}
    except (OSError, ValueError):
        payload = {}
    cwd = str(
        payload.get("cwd")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.getcwd()
    )
    session_id = str(
        payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or ""
    )
    try:
        flush(cwd, session_id)
    except Exception as exc:  # noqa: BLE001 — a hook failure must stay invisible
        print(f"clara stop-flush: skipped ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
