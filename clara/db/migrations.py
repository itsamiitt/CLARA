"""
CLARA — forward-only SQLite schema versioning.

Tracks the schema version in a ``schema_info`` table and applies pending
migrations idempotently, one transaction per migration. Built on stdlib
``sqlite3`` on purpose: the Claude Code plugin layers open the store
directly, without the async SQLAlchemy stack.

Intentionally NOT wired into the existing runtime paths this milestone —
the live schema is still created by SQLAlchemy ``create_all`` (see
``clara/integrations/local_memory.py`` and ``clara/agent.py``); wiring
lands with the plugin bootstrap. This module sets no journal-mode pragmas
(WAL is the runtime engine's concern), which also keeps the "never writes
when the schema is too new" guarantee byte-stable on disk.

The one rule callers must honor: if the database's version is NEWER than
this code knows (``SCHEMA_VERSION``), never write — ``ensure_schema``
raises ``SchemaTooNew`` before touching the file, and ``open_db`` reopens
the store read-only with a single stderr warning.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import pathname2url

SCHEMA_VERSION = 1

_BUSY_TIMEOUT_MS = 30_000  # matches SQLITE_BUSY_TIMEOUT_MS in clara/agent.py


class SchemaTooNew(RuntimeError):
    """The database was written by a newer CLARA than this code."""

    def __init__(self, found: int, supported: int) -> None:
        super().__init__(
            f"database schema version {found} is newer than this CLARA supports ({supported})"
        )
        self.found = found
        self.supported = supported


def _migration_1(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE schema_info (version INTEGER NOT NULL PRIMARY KEY, migrated_at TEXT NOT NULL)"
    )


_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_1),
]


def get_version(conn: sqlite3.Connection) -> int:
    """Current schema version; 0 for a database without ``schema_info``."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_info'"
    ).fetchone()
    if row is None:
        return 0
    value = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_info").fetchone()[0]
    return int(value)


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Apply pending migrations; return the resulting schema version.

    Idempotent and forward-only. Raises :class:`SchemaTooNew` — before any
    write — when the database version is newer than ``SCHEMA_VERSION``.
    """
    current = get_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaTooNew(current, SCHEMA_VERSION)
    for version, migrate in _MIGRATIONS:
        if version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            if get_version(conn) >= version:
                # Lost the race to a concurrent migrator; its commit stands.
                conn.execute("ROLLBACK")
                continue
            migrate(conn)
            conn.execute(
                "INSERT INTO schema_info (version, migrated_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return get_version(conn)


def open_db(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open the store at ``path``, migrating forward as needed.

    When the database schema is newer than this code supports, the store is
    reopened read-only (URI ``mode=ro`` plus ``PRAGMA query_only``) and one
    warning line goes to stderr — the file is never written.
    """
    resolved = Path(path).resolve()
    conn = sqlite3.connect(resolved, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    try:
        ensure_schema(conn)
    except SchemaTooNew as exc:
        conn.close()
        conn = sqlite3.connect(f"file:{pathname2url(str(resolved))}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        print(
            f"warning: {resolved}: schema v{exc.found} is newer than supported "
            f"v{exc.supported}; opening read-only — upgrade clara-memory to write",
            file=sys.stderr,
        )
    return conn
