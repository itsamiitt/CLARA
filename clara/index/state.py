"""
Index state — the content-hash gate that makes re-indexing cheap.

One row per (repo, path, kind) recording the hash of what was last indexed.
A file whose bytes have not changed is skipped in O(1), which is the same
mechanism clara/docs/scan.py already uses for documents.

``generation`` is a monotonic counter per repo. A rebuild bumps it; rows left
behind at an older generation are the ones no longer seen on disk, so a sweep
can retire them without diffing directory listings.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# blake2b over the file bytes. Not a security boundary -- this only answers
# "are these the same bytes I indexed last time" -- but it is faster than
# sha256 and collisions here would mean a stale index, so it is not md5 either.
_HASH_CHUNK = 1 << 20


def content_hash(path: Path) -> str | None:
    """blake2b of the file's bytes, or None if it cannot be read."""
    digest = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def is_unchanged(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    path: str,
    kind: str,
    current_hash: str | None,
) -> bool:
    """True when *path* was already indexed with exactly *current_hash*.

    An unreadable file (hash None) is never "unchanged": it may have been
    deleted or become unreadable, and both are worth re-processing.
    """
    if current_hash is None:
        return False
    row = conn.execute(
        "SELECT content_hash FROM index_state "
        "WHERE repo_id = ? AND path = ? AND kind = ?",
        (repo_id, path, kind),
    ).fetchone()
    return row is not None and row[0] == current_hash


def record_indexed(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    path: str,
    kind: str,
    content_hash: str | None,
    lang: str | None = None,
    generation: int = 0,
) -> None:
    """Record that *path* is indexed at *content_hash*."""
    conn.execute(
        "INSERT INTO index_state "
        "(repo_id, path, kind, content_hash, lang, last_indexed, generation) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(repo_id, path, kind) DO UPDATE SET "
        "content_hash = excluded.content_hash, lang = excluded.lang, "
        "last_indexed = excluded.last_indexed, generation = excluded.generation",
        (repo_id, path, kind, content_hash, lang, _now(), generation),
    )


def forget_path(conn: sqlite3.Connection, repo_id: str, *, path: str) -> int:
    """Drop every index-state row for *path* (all kinds). Returns rows removed."""
    cursor = conn.execute(
        "DELETE FROM index_state WHERE repo_id = ? AND path = ?", (repo_id, path)
    )
    return int(cursor.rowcount or 0)


def current_generation(conn: sqlite3.Connection, repo_id: str) -> int:
    row = conn.execute(
        "SELECT coalesce(max(generation), 0) FROM index_state WHERE repo_id = ?",
        (repo_id,),
    ).fetchone()
    return int(row[0])
