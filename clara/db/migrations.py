"""
CLARA — forward-only SQLite schema versioning.

Tracks the schema version in a ``schema_info`` table and applies pending
migrations idempotently, one transaction per migration. Built on stdlib
``sqlite3`` on purpose: the Claude Code plugin layers open the store
directly, without the async SQLAlchemy stack.

Since v4 this is the *primary* schema path for file-backed stores: the
memories table itself is created here (frozen DDL, IF NOT EXISTS), and
SQLAlchemy ``create_all`` remains only as a checkfirst no-op safety net and
for ``:memory:`` engines (see ``clara/integrations/local_memory.py``).

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

SCHEMA_VERSION = 7

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


# Curator document ledger (see clara/docs/). Deterministic signals about repo
# documents, keyed on repo_id so worktrees share one ledger. Rows are
# invalidated, never deleted; `clara docs scan` is the universal repair path.
_DOCS_DDL = [
    """
    CREATE TABLE doc_registry (
        doc_id TEXT PRIMARY KEY,
        user_id TEXT,
        repo_id TEXT NOT NULL,
        rel_path TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        doc_type TEXT DEFAULT 'unknown',
        tier TEXT DEFAULT 'T2',
        lifecycle TEXT DEFAULT 'active',
        superseded_by TEXT,
        fulfilled_by TEXT DEFAULT '{}',
        type_source TEXT DEFAULT 'heuristic',
        signals TEXT DEFAULT '{}',
        valid_from TIMESTAMP NOT NULL,
        invalid_at TIMESTAMP,
        last_verified TIMESTAMP,
        updated_at TIMESTAMP
    )
    """,
    """
    CREATE UNIQUE INDEX uq_doc_registry_path
        ON doc_registry (repo_id, rel_path) WHERE invalid_at IS NULL
    """,
    "CREATE INDEX ix_doc_registry_repo ON doc_registry (repo_id)",
    """
    CREATE TABLE doc_refs (
        doc_id TEXT NOT NULL,
        ref_kind TEXT NOT NULL,
        ref_value TEXT NOT NULL,
        resolved INTEGER DEFAULT 1,
        PRIMARY KEY (doc_id, ref_kind, ref_value)
    )
    """,
    """
    CREATE TABLE doc_attestations (
        att_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        actor TEXT,
        verdict TEXT,
        rationale TEXT,
        evidence TEXT DEFAULT '{}',
        created_at TIMESTAMP
    )
    """,
    "CREATE INDEX ix_doc_attestations_doc ON doc_attestations (doc_id)",
]


def _migration_3(conn: sqlite3.Connection) -> None:
    for statement in _DOCS_DDL:
        conn.execute(statement)


# The memories table, frozen exactly as SQLAlchemy's SQLite dialect emits it
# (see tests/test_db_migrations.py::test_create_all_parity — the drift alarm).
# Legacy stores built by ``Base.metadata.create_all`` already have all of
# these objects, so every statement is IF NOT EXISTS and the migration
# no-ops there; fresh stores get the canonical schema with no SQLAlchemy in
# the loop. This is what makes the memories table migratable at all: from
# v4 on, ALTERs land here as ordinary numbered migrations.
_MEMORIES_DDL = [
    """
    CREATE TABLE IF NOT EXISTS memories (
        memory_id CHAR(32) NOT NULL,
        user_id TEXT,
        memory_type VARCHAR(11) NOT NULL,
        content JSON NOT NULL,
        confidence FLOAT DEFAULT (0.5) NOT NULL,
        status VARCHAR(10) DEFAULT 'active' NOT NULL,
        decay_rate FLOAT DEFAULT (0.02) NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        metadata JSON,
        CONSTRAINT pk_memories PRIMARY KEY (memory_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_memories_created_at ON memories (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_memories_memory_type ON memories (memory_type)",
    "CREATE INDEX IF NOT EXISTS ix_memories_status ON memories (status)",
    "CREATE INDEX IF NOT EXISTS ix_memories_type_status ON memories (memory_type, status)",
    "CREATE INDEX IF NOT EXISTS ix_memories_user_id ON memories (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_memories_user_type_status"
    " ON memories (user_id, memory_type, status)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_world_model_identity
        ON memories (coalesce(user_id, ''),
                     json_extract(content, '$.entity_type'),
                     json_extract(content, '$.name'))
        WHERE memory_type = 'world_model' AND status = 'active'
    """,
]


def _migration_4(conn: sqlite3.Connection) -> None:
    for statement in _MEMORIES_DDL:
        conn.execute(statement)


# Searchable text per memories row. Shared with clara/db/fts.py (which
# imports it) so triggers written by either path index identical text.
# v5 extends the original content-only expression with tags and the
# native-import origin file so tag search works on the FTS path too.
FTS_TEXT_EXPR = (
    "lower(CAST({row}.content AS TEXT)"
    " || ' ' || coalesce(CAST(json_extract({row}.metadata, '$.tags') AS TEXT), '')"
    " || ' ' || coalesce(json_extract({row}.metadata, '$.origin_file'), ''))"
)

_FTS_COLUMNS = "memory_id UNINDEXED, user_id UNINDEXED, memory_type UNINDEXED, status UNINDEXED, text"


def _fts_trigger_ddl(table: str) -> list[str]:
    """Column-scoped sync triggers.

    A bare AFTER UPDATE trigger reindexes the row (porter AND trigram
    tokenization) on *every* write — which made the nightly decay pass and
    per-search access recording pay a full-store FTS rebuild. Only content /
    status / user_id changes, plus metadata changes that touch the indexed
    keys (tags, origin_file), actually need the index refreshed.
    """
    insert = (
        f"INSERT INTO {table}(memory_id, user_id, memory_type, status, text) "
        f"VALUES (new.memory_id, coalesce(new.user_id, ''), new.memory_type, "
        f"new.status, {FTS_TEXT_EXPR.format(row='new')});"
    )
    refresh = f"DELETE FROM {table} WHERE memory_id = old.memory_id; {insert}"
    meta_changed = (
        "coalesce(CAST(json_extract(new.metadata, '$.tags') AS TEXT), '') IS NOT "
        "coalesce(CAST(json_extract(old.metadata, '$.tags') AS TEXT), '') "
        "OR coalesce(json_extract(new.metadata, '$.origin_file'), '') IS NOT "
        "coalesce(json_extract(old.metadata, '$.origin_file'), '')"
    )
    # NOTE on retirement (archive/deprecate/supersede): there is deliberately
    # NO per-row status-retire trigger. memory_id is UNINDEXED in FTS5, so a
    # per-row DELETE scans the whole index — bulk archival would be O(rows ×
    # index). Stale entries are filtered out at hydration (search re-checks
    # status against the memories table) and swept in one pass by
    # fts_gc_statements() during daily maintenance.
    return [
        f"CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON memories BEGIN {insert} END",
        f"CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON memories BEGIN "
        f"DELETE FROM {table} WHERE memory_id = old.memory_id; END",
        # Active rows: content/tenant changes refresh the index entry.
        f"CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE OF content, user_id "
        f"ON memories WHEN new.status = 'active' BEGIN {refresh} END",
        # Reactivation (rare: restore/import) re-inserts the entry; the
        # per-row scan cost is fine at this frequency.
        f"CREATE TRIGGER IF NOT EXISTS {table}_aua AFTER UPDATE OF status ON memories "
        f"WHEN new.status = 'active' AND old.status != 'active' BEGIN {refresh} END",
        f"CREATE TRIGGER IF NOT EXISTS {table}_aum AFTER UPDATE OF metadata ON memories "
        f"WHEN new.status = 'active' AND ({meta_changed}) BEGIN {refresh} END",
    ]


def fts_gc_statements() -> list[str]:
    """One-pass sweep of retired rows out of both FTS tables (run by the
    daily maintenance pass; each statement is one index scan total)."""
    return [
        f"DELETE FROM {table} WHERE memory_id IN "
        f"(SELECT memory_id FROM memories WHERE status != 'active')"
        for table in ("memories_fts", "memories_fts_tri")
    ]


def _fts_backfill(table: str) -> str:
    # Only active rows: retiring a row deletes its index entry (see the
    # _aur trigger), so the index holds exactly the searchable set.
    return (
        f"INSERT INTO {table}(memory_id, user_id, memory_type, status, text) "
        f"SELECT memory_id, coalesce(user_id, ''), memory_type, status, "
        f"{FTS_TEXT_EXPR.format(row='memories')} FROM memories "
        f"WHERE status = 'active' "
        f"AND memory_id NOT IN (SELECT memory_id FROM {table})"
    )


def _rebuild_fts(conn: sqlite3.Connection, table: str, tokenize: str) -> None:
    for suffix in ("_ai", "_ad", "_au", "_aur", "_aua", "_aum"):
        conn.execute(f"DROP TRIGGER IF EXISTS {table}{suffix}")
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        f"CREATE VIRTUAL TABLE {table} USING fts5({_FTS_COLUMNS}, tokenize='{tokenize}')"
    )
    for statement in _fts_trigger_ddl(table):
        conn.execute(statement)
    conn.execute(_fts_backfill(table))


def _migration_5(conn: sqlite3.Connection) -> None:
    """Rebuild memories_fts with the v5 text expression (content + tags +
    origin_file) and add a trigram twin for CJK / substring queries.

    Both are best-effort: FTS5 or its trigram tokenizer (SQLite >= 3.34) may
    be missing from this build — search then degrades to the scan path, and
    the migration still records v5.
    """
    with contextlib.suppress(sqlite3.OperationalError):
        _rebuild_fts(conn, "memories_fts", "porter unicode61")
    with contextlib.suppress(sqlite3.OperationalError):
        _rebuild_fts(conn, "memories_fts_tri", "trigram")


def _migration_6(conn: sqlite3.Connection) -> None:
    """Re-run the FTS rebuild: v5's bare AFTER UPDATE triggers reindexed
    every row on bookkeeping writes; v6 installs the column-scoped set."""
    _migration_5(conn)


def _migration_7(conn: sqlite3.Connection) -> None:
    """Composite index on the hot sort key.

    The SessionStart fastpath and the ILIKE fallback both do
    ``WHERE status='active' ORDER BY updated_at DESC``; without this index
    that is a full scan + filesort of every active row on each session start.
    Kept in sync with the matching Index in clara/db/models.py (parity test).
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_status_updated "
        "ON memories (status, updated_at DESC)"
    )
    # graph_aliases is queried by node_id on every add_alias (resolve.py) but
    # its PK is (alias_norm, user_id), so that lookup was a full scan.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_graph_aliases_node "
        "ON graph_aliases (node_id)"
    )


_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_1),
    (2, _migration_2),
    (3, _migration_3),
    (4, _migration_4),
    (5, _migration_5),
    (6, _migration_6),
    (7, _migration_7),
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


def check_version(conn: sqlite3.Connection) -> int:
    """Read-only version gate: raise :class:`SchemaTooNew` when the database
    is ahead of this code, else return the current version. Never writes —
    this is what the fastpath calls instead of :func:`ensure_schema`."""
    current = get_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaTooNew(current, SCHEMA_VERSION)
    return current


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
