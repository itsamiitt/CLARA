"""
Change journal — the indexing pipeline's queue.

Any event source (a hook, a scan, a git diff) enqueues with one cheap INSERT.
A worker claims a batch, processes it, and deletes what it finished. Nothing
here parses or indexes; this module only owns the queue's integrity.

Crash safety, in the same shape as the maintenance lock in clara/maintenance.py:
a worker stamps `claimed_by`/`claimed_at`, and a claim older than
:data:`STALE_CLAIM_SECONDS` is treated as abandoned and released. A worker that
dies mid-batch therefore loses no work -- the rows return to the queue rather
than disappearing with it.

Claiming uses BEGIN IMMEDIATE + UPDATE + SELECT rather than
``UPDATE ... RETURNING``. RETURNING needs SQLite 3.35 (2021) and CLARA supports
whatever sqlite3 the user's Python 3.10+ shipped with; the transaction gives
the same single-winner guarantee everywhere.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

# A claim older than this is assumed to belong to a dead worker.
STALE_CLAIM_SECONDS = 15 * 60

# Change kinds the journal accepts. 'git' and 'manifest' are repo-level and
# carry no path.
CHANGES = frozenset(
    {"added", "modified", "removed", "renamed", "git", "manifest"}
)

_REPO_LEVEL = frozenset({"git", "manifest"})


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One claimed change, ready to process."""

    seq: int
    repo_id: str
    path: str | None
    change: str
    old_path: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def enqueue(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    change: str,
    path: str | None = None,
    old_path: str | None = None,
) -> int:
    """Record one change. Returns its sequence number.

    Deliberately does not deduplicate: two edits to one file between worker
    runs enqueue twice, and the worker's content-hash gate makes the second a
    no-op. Deduplicating here would need a read before every write on the hot
    path, to save work that is already O(1) to skip.
    """
    if change not in CHANGES:
        raise ValueError(f"unknown change {change!r}; expected one of {sorted(CHANGES)}")
    if change not in _REPO_LEVEL and not path:
        raise ValueError(f"change {change!r} requires a path")
    cursor = conn.execute(
        "INSERT INTO change_journal (repo_id, path, change, old_path, detected_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (repo_id, path, change, old_path, _now()),
    )
    # Commit immediately. A queued change that is lost to a crash is a file
    # that never gets re-indexed, and the whole point of the journal is that
    # the work survives the process. It also leaves the connection clean:
    # sqlite3 opens an implicit transaction for DML, which would collide with
    # the BEGIN IMMEDIATE in claim_batch.
    conn.commit()
    return int(cursor.lastrowid or 0)


def pending_count(conn: sqlite3.Connection, repo_id: str | None = None) -> int:
    """Unclaimed entries, for status output and tests."""
    if repo_id is None:
        row = conn.execute(
            "SELECT count(*) FROM change_journal WHERE claimed_by IS NULL"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT count(*) FROM change_journal "
            "WHERE claimed_by IS NULL AND repo_id = ?",
            (repo_id,),
        ).fetchone()
    return int(row[0])


def release_stale_claims(
    conn: sqlite3.Connection, *, older_than_seconds: int = STALE_CLAIM_SECONDS
) -> int:
    """Return abandoned claims to the queue. Returns how many were released."""
    cutoff = datetime.now(timezone.utc).timestamp() - older_than_seconds
    cursor = conn.execute(
        "UPDATE change_journal SET claimed_by = NULL, claimed_at = NULL "
        "WHERE claimed_by IS NOT NULL "
        "AND strftime('%s', claimed_at) < ?",
        (str(int(cutoff)),),
    )
    return int(cursor.rowcount or 0)


def claim_batch(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    worker: str,
    limit: int = 200,
) -> list[JournalEntry]:
    """Claim up to *limit* unclaimed entries for *repo_id*.

    Single-winner across concurrent workers: the whole claim runs inside
    BEGIN IMMEDIATE, so a second worker either waits and sees the rows already
    claimed, or is told the database is locked and retries later.
    """
    if limit <= 0:
        return []
    # Any implicit transaction the caller left open would make BEGIN IMMEDIATE
    # fail with "cannot start a transaction within a transaction".
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        release_stale_claims(conn)
        rows = conn.execute(
            "SELECT seq FROM change_journal "
            "WHERE repo_id = ? AND claimed_by IS NULL "
            "ORDER BY seq LIMIT ?",
            (repo_id, limit),
        ).fetchall()
        if not rows:
            conn.execute("COMMIT")
            return []
        seqs = [int(r[0]) for r in rows]
        placeholders = ", ".join("?" for _ in seqs)
        conn.execute(
            f"UPDATE change_journal SET claimed_by = ?, claimed_at = ? "  # noqa: S608
            f"WHERE seq IN ({placeholders})",
            (worker, _now(), *seqs),
        )
        claimed = conn.execute(
            f"SELECT seq, repo_id, path, change, old_path FROM change_journal "  # noqa: S608
            f"WHERE seq IN ({placeholders}) ORDER BY seq",
            seqs,
        ).fetchall()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return [
        JournalEntry(seq=int(r[0]), repo_id=r[1], path=r[2], change=r[3], old_path=r[4])
        for r in claimed
    ]


def complete(conn: sqlite3.Connection, seqs: list[int] | tuple[int, ...]) -> int:
    """Delete finished entries. Returns how many were removed."""
    if not seqs:
        return 0
    placeholders = ", ".join("?" for _ in seqs)
    cursor = conn.execute(
        f"DELETE FROM change_journal WHERE seq IN ({placeholders})",  # noqa: S608
        tuple(seqs),
    )
    conn.commit()
    return int(cursor.rowcount or 0)
