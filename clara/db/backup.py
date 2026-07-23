"""
CLARA — store backups: rotated snapshots, pre-migration safety, restore.

``VACUUM INTO`` (SQLite >= 3.27 — everywhere Python >= 3.10 ships) writes a
compact, consistent snapshot without blocking readers; the
``sqlite3.Connection.backup`` API is the fallback. Backups live in a
``backups/`` directory beside the store:

    ~/.clara/backups/clara-20260723T101500Z-daily.db

Rotation keeps ``CLARA_BACKUP_KEEP`` (default 7) newest; the oldest are
deleted only after a new backup succeeds. Stdlib-only.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KEEP = 7


def _keep() -> int:
    try:
        return max(1, int(os.environ.get("CLARA_BACKUP_KEEP", DEFAULT_KEEP)))
    except ValueError:
        return DEFAULT_KEEP


def backup_dir(db_path: str | os.PathLike[str]) -> Path:
    return Path(db_path).resolve().parent / "backups"


def backup_db(db_path: str | os.PathLike[str], reason: str = "manual") -> Path | None:
    """Snapshot the store; return the backup path or ``None`` on failure.

    Never raises — a failed backup is logged and the caller proceeds (memory
    availability beats backup availability), except callers that check the
    ``None`` return and choose to stop.
    """
    source = Path(db_path)
    if not source.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "-" for c in reason)[:40]
    dest_dir = backup_dir(source)
    dest = dest_dir / f"clara-{stamp}-{safe_reason}.db"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(source)
        try:
            try:
                conn.execute("VACUUM INTO ?", (str(dest),))
            except sqlite3.OperationalError:
                # VACUUM INTO refuses inside a transaction or on old builds —
                # fall back to the online backup API.
                if dest.exists():
                    dest.unlink()
                target = sqlite3.connect(dest)
                try:
                    conn.backup(target)
                finally:
                    target.close()
        finally:
            conn.close()
        from clara.store import secure_store_file

        secure_store_file(dest)
        _rotate(dest_dir)
        logger.info("backed up %s -> %s", source, dest)
        return dest
    except (OSError, sqlite3.Error):
        logger.warning("backup of %s failed", source, exc_info=True)
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return None


def _rotate(dest_dir: Path) -> None:
    snapshots = sorted(
        dest_dir.glob("clara-*.db"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in snapshots[_keep():]:
        try:
            stale.unlink()
        except OSError:
            logger.warning("could not rotate old backup %s", stale)


def backup_before_migration(conn: sqlite3.Connection, db_path: str) -> None:
    """Snapshot the store when versioned migrations are pending.

    Called by the schema path with an open connection; the check is one
    read, and the backup runs on its own connection so ``VACUUM INTO``
    isn't inside the caller's transaction.
    """
    from clara.db.migrations import SCHEMA_VERSION, get_version

    try:
        current = get_version(conn)
    except sqlite3.Error:
        return
    if current == 0 or current >= SCHEMA_VERSION:
        return  # fresh store (nothing to lose) or already current
    backup_db(db_path, reason=f"pre-migration-v{SCHEMA_VERSION}")


def latest_backup(db_path: str | os.PathLike[str]) -> Path | None:
    snapshots = sorted(
        backup_dir(db_path).glob("clara-*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None
