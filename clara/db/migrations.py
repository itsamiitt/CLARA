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

import contextlib
import os
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

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


# Knowledge-graph projection tables (see clara/graph/). The graph is derived
# from the memories table — every row here is rebuildable via
# `clara graph rebuild`, and edges are invalidated, never deleted.
_GRAPH_DDL = [
    """
    CREATE TABLE graph_nodes (
        node_id TEXT PRIMARY KEY,
        user_id TEXT,
        canonical_name TEXT NOT NULL,
        display_name TEXT NOT NULL,
        entity_type TEXT NOT NULL DEFAULT 'concept',
        world_model_id TEXT,
        properties TEXT DEFAULT '{}',
        mention_count INTEGER DEFAULT 0,
        expandable INTEGER NOT NULL DEFAULT 1,
        status TEXT DEFAULT 'active',
        merged_into TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    """,
    """
    CREATE UNIQUE INDEX uq_graph_nodes_identity
        ON graph_nodes (coalesce(user_id, ''), entity_type, canonical_name)
        WHERE status = 'active'
    """,
    """
    CREATE TABLE graph_aliases (
        alias_norm TEXT NOT NULL,
        node_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT '',
        source TEXT DEFAULT 'auto',
        PRIMARY KEY (alias_norm, user_id)
    )
    """,
    """
    CREATE TABLE graph_edges (
        edge_id TEXT PRIMARY KEY,
        user_id TEXT,
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        belief_id TEXT,
        confidence REAL DEFAULT 0.8,
        weight REAL DEFAULT 1.0,
        valid_from TIMESTAMP NOT NULL,
        invalid_at TIMESTAMP,
        temporal_precision TEXT DEFAULT 'exact',
        metadata TEXT DEFAULT '{}'
    )
    """,
    "CREATE INDEX ix_graph_edges_src_valid ON graph_edges (src_id) WHERE invalid_at IS NULL",
    "CREATE INDEX ix_graph_edges_dst_valid ON graph_edges (dst_id) WHERE invalid_at IS NULL",
    "CREATE INDEX ix_graph_edges_belief ON graph_edges (belief_id)",
]


def _migration_2(conn: sqlite3.Connection) -> None:
    for statement in _GRAPH_DDL:
        conn.execute(statement)
    # FTS5/trigram may be unavailable in this SQLite build — entity resolution
    # then degrades to a scan; the migration itself must still succeed.
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(
            "CREATE VIRTUAL TABLE graph_nodes_fts USING fts5("
            "node_id UNINDEXED, names, entity_type, tokenize='trigram')"
        )


_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_1),
    (2, _migration_2),
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
    # Lazy: urllib.request is comparatively heavy and only the too-new branch
    # needs it — keeping it out of module import keeps clara.fastpath cheap.
    from urllib.request import pathname2url

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
