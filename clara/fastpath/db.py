"""
CLARA fastpath — store resolution and raw-SQL access.

Stdlib-only (see the package docstring's hard rule). Mirrors the runtime's
store-location logic without importing it:

- project store: ``<repo root>/.clara/clara.db`` when the file exists,
- else global store: ``$CLARA_DB_PATH``, or ``$CLARA_HOME/clara.db``,
  or ``~/.clara/clara.db``.

Never creates files or directories — a missing store means "no context".
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from clara.db.migrations import SchemaTooNew, ensure_schema
from clara.repoid import repo_id

_GIT_TIMEOUT_S = 5.0
_BUSY_TIMEOUT_MS = 3_000  # session start must not hang on a locked store

# Candidate cap before re-ranking in Python; matches
# clara.retrieval.lexical.DEFAULT_CANDIDATE_LIMIT.
CANDIDATE_LIMIT = 1000


def _git_toplevel(cwd: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def global_db_path() -> Path:
    """The global store path ($CLARA_DB_PATH / $CLARA_HOME / ~/.clara).

    The doc ledger always lives here (keyed by repo_id) so worktrees and
    clones of one repository share a single ledger.
    """
    explicit = os.environ.get("CLARA_DB_PATH")
    if explicit:
        return Path(explicit)
    base = Path(os.environ.get("CLARA_HOME") or (Path.home() / ".clara"))
    return base / "clara.db"


def resolve_store(cwd: str) -> tuple[Path | None, str]:
    """Return ``(db_path or None, repo_id)`` for the session working dir."""
    rid = repo_id(cwd)
    root = _git_toplevel(cwd) or cwd
    project = Path(root) / ".clara" / "clara.db"
    if project.is_file():
        return project, rid
    candidate = global_db_path()
    if candidate.is_file():
        return candidate, rid
    return None, rid


def open_store(path: Path) -> sqlite3.Connection | None:
    """Open the store and run the schema check.

    Returns ``None`` — after one stderr line — when the database schema is
    newer than this code supports (never write, emit nothing). Other schema
    bookkeeping failures (locked file, read-only mount) are logged and
    tolerated: the read below may still succeed.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    try:
        ensure_schema(conn)
    except SchemaTooNew as exc:
        print(f"clara fastpath: {path}: {exc}; emitting no context", file=sys.stderr)
        conn.close()
        return None
    except sqlite3.Error as exc:
        print(f"clara fastpath: {path}: schema check skipped ({exc})", file=sys.stderr)
    return conn


def fetch_active(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Active memories, newest first, with pre-parsed JSON payloads.

    ``strftime('%s', ...)`` parses the ISO timestamps SQLAlchemy wrote
    (including a ``+00:00`` suffix) into a UTC epoch inside SQLite, so no
    datetime module is needed here.
    """
    rows = conn.execute(
        "SELECT memory_type, content, confidence, metadata, "
        "       created_at, CAST(strftime('%s', updated_at) AS INTEGER) "
        "FROM memories WHERE status = 'active' "
        "ORDER BY updated_at DESC LIMIT ?",
        (CANDIDATE_LIMIT,),
    ).fetchall()
    memories: list[dict[str, object]] = []
    for mem_type, content, confidence, metadata, created_at, updated_epoch in rows:
        try:
            payload = json.loads(content) if isinstance(content, str) else (content or {})
        except ValueError:
            payload = {}
        try:
            meta = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
        except ValueError:
            meta = {}
        memories.append(
            {
                "type": str(mem_type),
                "content": payload if isinstance(payload, dict) else {},
                "confidence": float(confidence if confidence is not None else 0.5),
                "metadata": meta if isinstance(meta, dict) else {},
                "created_at": str(created_at or ""),
                "updated_epoch": int(updated_epoch) if updated_epoch is not None else None,
            }
        )
    return memories
