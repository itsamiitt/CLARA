"""
CLARA — sidecar counter for the status line.

Claude Code pulls the status line on session events and, when
``statusLine.refreshInterval`` is set, on a timer of a few seconds. Opening
SQLite on that cadence would be wasteful and can contend with a live writer,
so the active-memory count lives in a tiny sidecar file next to the store that
the write paths refresh.

Stdlib-only on purpose: the status line runs in a cold interpreter where import
cost is the dominant latency, exactly like ``clara.fastpath``.

Fail-soft everywhere: a counter is cosmetic. A failure to read or write it must
never break a save, and must never make the status line raise — a status line
that exits non-zero blanks the bar entirely.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import tempfile
import time
from pathlib import Path

SUFFIX = ".stats"

# The count is refreshed by every write path, so a hit is normally seconds old.
# The TTL is only a backstop for a store written by an older CLARA (or by a
# process that crashed before refreshing): past it, the reader recounts once.
DEFAULT_MAX_AGE_S = 300.0

_COUNT_SQL = "SELECT COUNT(*) FROM memories WHERE status = 'active'"


def path_for(db_path: str | os.PathLike[str]) -> Path:
    """Sidecar path for *db_path* (``<store>.stats``)."""
    return Path(str(db_path) + SUFFIX)


def write(db_path: str | os.PathLike[str], count: int) -> None:
    """Record *count* for the store at *db_path*. Never raises.

    Written via a temporary file plus ``os.replace`` so a reader can never
    observe a half-written (or empty) counter.
    """
    target = path_for(db_path)
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".stats-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(int(count)))
            os.replace(tmp_name, target)
        except BaseException:
            # Do not leave the temp file behind if the swap failed.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    except (OSError, ValueError):
        return


def read(
    db_path: str | os.PathLike[str],
    *,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> int | None:
    """Cached active-memory count, or ``None`` when absent/stale/unreadable."""
    cache = path_for(db_path)
    try:
        if not cache.is_file():
            return None
        if time.time() - cache.stat().st_mtime >= max_age_s:
            return None
        return int(cache.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return None


def refresh(db_path: str | os.PathLike[str]) -> int | None:
    """Recount from the store and update the sidecar. Never raises.

    Uses a short busy timeout: refreshing a cosmetic counter must never wait
    behind a real writer.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=0.5)
    except sqlite3.Error:
        return None
    try:
        conn.execute("PRAGMA busy_timeout = 500")
        count = int(conn.execute(_COUNT_SQL).fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()
    write(db_path, count)
    return count
