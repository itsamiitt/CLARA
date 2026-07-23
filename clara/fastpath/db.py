"""
CLARA fastpath — store resolution and raw-SQL access.

Stdlib-only (see the package docstring's hard rule). Store location comes
from :mod:`clara.store` — the same resolver the MCP server and CLI use, so
what the SessionStart hook injects is exactly what the tools read and write.

Never creates files or directories — a missing store means "no context" —
and never migrates: connections are opened ``query_only``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from clara.db.migrations import SchemaTooNew, check_version
from clara.repoid import repo_id
from clara.store import global_db_path, resolve_store as _resolve
from clara.store import git_toplevel as _git_toplevel  # noqa: F401 — re-export for tests

_BUSY_TIMEOUT_MS = 3_000  # session start must not hang on a locked store

# Candidate cap before re-ranking in Python; matches
# clara.retrieval.lexical.DEFAULT_CANDIDATE_LIMIT.
CANDIDATE_LIMIT = 1000

__all__ = ["CANDIDATE_LIMIT", "global_db_path", "resolve_store", "open_store", "fetch_active"]


def resolve_store(cwd: str) -> tuple[Path | None, str]:
    """Return ``(db_path or None, repo_id)`` for the session working dir.

    Thin delegate to :func:`clara.store.resolve_store` so the hook reads the
    exact store the MCP server and CLI write.
    """
    rid = repo_id(cwd)
    res = _resolve(cwd, create=False)
    return (res.db_path if res.exists else None), rid


def open_store(path: Path) -> sqlite3.Connection | None:
    """Open the store read-only and run the version check.

    Returns ``None`` — after one stderr line — when the database schema is
    newer than this code supports. The fastpath never migrates and never
    writes: ``PRAGMA query_only`` enforces the contract mechanically. Other
    schema bookkeeping failures (locked file) are logged and tolerated: the
    read below may still succeed.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
    try:
        check_version(conn)
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
        "       created_at, CAST(strftime('%s', updated_at) AS INTEGER), "
        "       memory_id "
        "FROM memories WHERE status = 'active' "
        "ORDER BY updated_at DESC LIMIT ?",
        (CANDIDATE_LIMIT,),
    ).fetchall()
    memories: list[dict[str, object]] = []
    for mem_type, content, confidence, metadata, created_at, updated_epoch, memory_id in rows:
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
                "memory_id": str(memory_id or ""),
            }
        )
    return memories
