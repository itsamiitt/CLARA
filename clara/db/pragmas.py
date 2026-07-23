"""
CLARA — one PRAGMA policy for every SQLite open path.

Two profiles, chosen by role:

========== ============== ============ ================= ===========
profile    busy_timeout   journal     synchronous       query_only
========== ============== ============ ================= ===========
RUNTIME    30 000 ms      WAL          NORMAL            off
FASTPATH   3 000 ms       (inherited)  (inherited)       ON
========== ============== ============ ================= ===========

RUNTIME is every writer (async engine, migrations, raw sqlite3 docs/CLI
reads that share writer connections). FASTPATH is the SessionStart hook:
the short timeout keeps session start from hanging on a locked store, and
``query_only`` mechanically enforces the fastpath's "never writes" contract.
WAL is a persistent database-file property, so readers inherit it from the
first writer without setting it themselves.

Stdlib-only so the fastpath may import it.
"""

from __future__ import annotations

import sqlite3

RUNTIME_BUSY_TIMEOUT_MS = 30_000
FASTPATH_BUSY_TIMEOUT_MS = 3_000


def apply_runtime(conn: sqlite3.Connection) -> None:
    """Writer profile: long busy timeout, WAL, NORMAL sync."""
    conn.execute(f"PRAGMA busy_timeout = {RUNTIME_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")


def apply_fastpath_read(conn: sqlite3.Connection) -> None:
    """Hook reader profile: short busy timeout, writes forbidden."""
    conn.execute(f"PRAGMA busy_timeout = {FASTPATH_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
