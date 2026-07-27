"""
CLARA — opportunistic housekeeping (backup, decay, pruning, graph, export).

Replaces the APScheduler cron jobs for the zero-backend profile: no daemon and
no timer, the pass rides the first store access of the day. An O_EXCL lock file
makes it single-winner across concurrent sessions; the ``.maintenance`` marker
records cadence only. Every failure is logged and swallowed -- memory
availability must never depend on housekeeping.

Lives here, not in clara.integrations.mcp_server, because the MCP server used
to be the only caller and therefore the only way a store ever got maintained.
A CLI-only user was promised decay, pruning and rotated backups by the README
and received none of them: verified, `clara remember` plus four `clara stats`
left no marker and no daily backup. `clara maintain` now calls the same pass.

The store-resolution *anchor* is a parameter rather than something this module
decides. The MCP server resolves it from workspace roots (which follow the
client across a mid-session cd); the CLI simply uses its cwd. Baking either
rule in here would have given one of the two callers the wrong store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from pathlib import Path

from clara.integrations.local_memory import LocalMemory

logger = logging.getLogger(__name__)

# One pass per day; a lock older than this is treated as abandoned.
MAINTENANCE_INTERVAL_SECONDS = 24 * 3600
_MAINTENANCE_LOCK_STALE_S = 6 * 3600


def _index_repo_sync(db_path: str, root: Path) -> tuple[int, int]:
    """Index *root* on this thread, opening and closing its own connection."""
    import sqlite3

    from clara.index import indexer
    from clara.repoid import repo_id

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        result = indexer.index_repo(conn, repo_id(str(root)), root)
    finally:
        conn.close()
    return result.processed, result.skipped_unchanged


async def _session_anchor_for_index(anchor: str) -> str | None:
    """Only index somewhere that looks like a repo checkout.

    Indexing the home directory because a session started there would walk an
    enormous tree for nothing, so a directory with no git toplevel is skipped.
    """
    from clara.store import git_toplevel

    return anchor if git_toplevel(anchor) else None


async def run_if_due(
    memory: LocalMemory,
    db_path: str,
    *,
    anchor: str,
    force: bool = False,
) -> str | None:
    """Run backup + decay + pruning when the last pass is stale.

    Replaces the APScheduler cron jobs for the zero-backend profile: no
    daemon, no timer — maintenance rides the first store access of the day.
    An O_EXCL lock file makes the pass single-winner across concurrent
    sessions; the ``.maintenance`` marker records cadence only. All failures
    are logged and swallowed; memory availability must never depend on
    housekeeping.
    """
    marker = Path(db_path + ".maintenance")
    lock_path = Path(db_path + ".maintenance.lock")
    try:
        if marker.exists() and not force:
            age = time.time() - marker.stat().st_mtime
            if age < MAINTENANCE_INTERVAL_SECONDS:
                return None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime < _MAINTENANCE_LOCK_STALE_S:
                    return None  # another session is running the pass
                lock_path.unlink()
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (OSError, FileExistsError):
                return None
        try:
            os.write(fd, f"{os.getpid()},{time.time():.0f}".encode())
            os.close(fd)

            # Backup first: VACUUM INTO must not run inside a write
            # transaction, and a pre-decay snapshot is the more useful one.
            from clara.db.backup import backup_db

            backup_db(db_path, reason="daily")

            from clara.scheduler.decay import DecayScheduler

            scheduler = DecayScheduler(memory.session_factory)
            decay_summary = await scheduler.run_daily_decay()
            prune_summary = await scheduler.run_weekly_pruning()

            # Sweep retired rows out of the FTS indexes (there is no per-row
            # retire trigger — see clara/db/migrations.py fts_gc_statements).
            try:
                import sqlite3 as _sqlite3

                from clara.db.migrations import fts_gc_statements

                def _fts_gc() -> None:
                    gc_conn = _sqlite3.connect(db_path)
                    try:
                        gc_conn.execute("PRAGMA busy_timeout = 30000")
                        for statement in fts_gc_statements():
                            # FTS table absent in this SQLite build — skip the GC.
                            with contextlib.suppress(_sqlite3.OperationalError):
                                gc_conn.execute(statement)
                        gc_conn.commit()
                    finally:
                        gc_conn.close()

                await asyncio.to_thread(_fts_gc)
            except Exception:  # noqa: BLE001 — index GC is best-effort
                logger.exception("FTS garbage collection failed")
            graph_summary = "graph: skipped"
            try:
                from clara.config import ClaraConfig
                from clara.graph.maintain import maintenance_summary, run_graph_maintenance

                config = ClaraConfig.from_env()
                async with memory.session() as session:
                    graph_counts = await run_graph_maintenance(
                        session,
                        archival_threshold=config.archival_threshold,
                        stale_days=config.event_stale_days,
                    )
                graph_summary = maintenance_summary(graph_counts)
            except Exception:  # noqa: BLE001 — graph housekeeping is best-effort
                logger.exception("Graph maintenance failed")
            index_summary = "index: skipped"
            try:
                # Keep the code graph current without asking the user to run
                # anything. Incremental by content hash, so the steady-state
                # cost is one hash per file; only a repo that changed pays to
                # re-parse. Best-effort like every other step here -- an
                # indexing failure must not cost the user their backup or
                # decay pass.
                from clara.store import git_toplevel as _git_toplevel

                index_anchor = await _session_anchor_for_index(anchor)
                if index_anchor is not None:
                    root = Path(_git_toplevel(index_anchor) or index_anchor)
                    # The connection is opened *inside* the worker thread.
                    # sqlite3 objects are bound to their creating thread, so
                    # handing one to asyncio.to_thread fails with "SQLite
                    # objects created in a thread can only be used in that same
                    # thread" -- which the fail-soft wrapper then swallowed into
                    # a silent "index: skipped".
                    counts = await asyncio.to_thread(_index_repo_sync, db_path, root)
                    index_summary = (
                        f"index: {counts[0]} parsed, {counts[1]} unchanged"
                    )
            except Exception as exc:  # noqa: BLE001 — indexing is best-effort
                logger.exception("Code indexing failed")
                # The summary is the only line a person sees, and "skipped"
                # for both "no repo here" and "crashed" made a real failure
                # (a cross-thread sqlite connection) invisible for days. Name
                # the failure; the log has the traceback.
                index_summary = f"index: FAILED ({type(exc).__name__}: {exc})"

            sync_summary = "sync: skipped"
            try:
                from clara.bridge.exporter import export_native

                exported = await asyncio.to_thread(export_native, db_path, anchor)
                sync_summary = f"sync: {exported}"
            except ImportError:
                pass
            except Exception:  # noqa: BLE001 — bridge export is best-effort
                logger.exception("Native-memory export failed")
            marker.touch()
            summary = (
                f"decay: {decay_summary}  prune: {prune_summary}  "
                f"{graph_summary}  {index_summary}  {sync_summary}"
            )
            logger.info("Opportunistic maintenance: %s", summary)
            return summary
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()
    except Exception:  # noqa: BLE001 — housekeeping must never block memory
        logger.exception("Opportunistic maintenance failed")
    return None
