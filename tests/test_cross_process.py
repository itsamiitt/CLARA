"""Multi-process integrity: several OS processes writing one store.

This is CLARA's central safety claim and the last part of it that was only ever
checked by hand. In a real session the MCP server, the `clara` CLI, the
SessionStart fastpath and the PostToolUse hooks all open the *same* SQLite file
at the same time, in different processes. `asyncio.gather` cannot model that:
it shares one connection pool and one lock.

What must hold under genuine concurrency:
  * no writes are lost (SQLITE_BUSY must be retried, not swallowed),
  * the database stays structurally sound (`PRAGMA integrity_check`),
  * the FTS index does not drift from the base table,
  * a reader never sees a torn row.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from clara.db.migrations import ensure_schema

# Spawning interpreters is slow on Windows; keep the fan-out meaningful but
# bounded so this stays usable in the default tier.
_WRITERS = 4
_ROWS_EACH = 15
_TIMEOUT_S = 240


def _make_store(tmp_path) -> str:
    db = tmp_path / "clara.db"
    conn = sqlite3.connect(db)
    try:
        ensure_schema(conn)
    finally:
        conn.close()
    return str(db)


_WRITER = textwrap.dedent(
    """
    import asyncio, sys
    from clara.integrations.local_memory import LocalMemory

    async def main(db, tag, count):
        memory = await LocalMemory.create(db)
        try:
            for i in range(count):
                await memory.save(
                    mem_type="belief",
                    subject=f"{tag}",
                    relation="uses",
                    object=f"tool-{i}",
                )
        finally:
            await memory.close()

    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
    """
)

_READER = textwrap.dedent(
    """
    import asyncio, json, sys
    from clara.integrations.local_memory import LocalMemory

    async def main(db, rounds):
        memory = await LocalMemory.create(db)
        seen = []
        try:
            for _ in range(rounds):
                # A torn read would surface as a missing/!=dict content field.
                result = await memory.recent(n=50)
                for hit in result["hits"]:
                    assert isinstance(hit["content"], dict), hit
                seen.append(result["total"])
        finally:
            await memory.close()
        print(json.dumps({"totals": seen}))

    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
    """
)


def _spawn(code: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _integrity(db: str) -> str:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


def _counts(db: str) -> tuple[int, int]:
    """(active rows, rows indexed in FTS) — these must not drift apart."""
    conn = sqlite3.connect(db)
    try:
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]
        try:
            indexed = conn.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE status = 'active'"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            indexed = -1  # FTS5 unavailable in this build
        return active, indexed
    finally:
        conn.close()


@pytest.fixture(scope="module")
def concurrent_store(tmp_path_factory):
    """One real fan-out of writer processes, shared by the assertions below.

    Spawning interpreters dominates the runtime on Windows, and every writer
    assertion inspects the same end state — so do the expensive part once.
    """
    tmp_path = tmp_path_factory.mktemp("crossproc")
    db = _make_store(tmp_path)
    procs = [
        _spawn(_WRITER, db, f"writer{i}", str(_ROWS_EACH)) for i in range(_WRITERS)
    ]
    failures = []
    for proc in procs:
        _out, err = proc.communicate(timeout=_TIMEOUT_S)
        if proc.returncode != 0:
            failures.append(err.strip()[-400:])
    return db, failures


class TestConcurrentWriters:
    def test_every_writer_process_succeeded(self, concurrent_store):
        _db, failures = concurrent_store
        assert not failures, "writer process failed:\n" + "\n---\n".join(failures)

    def test_no_writes_are_lost(self, concurrent_store):
        db, _ = concurrent_store
        active, _indexed = _counts(db)
        assert active == _WRITERS * _ROWS_EACH, (
            f"expected {_WRITERS * _ROWS_EACH} rows, found {active} — "
            "a concurrent write was lost"
        )

    def test_database_stays_intact(self, concurrent_store):
        db, _ = concurrent_store
        assert _integrity(db) == "ok"

    def test_fts_index_does_not_drift(self, concurrent_store):
        db, _ = concurrent_store
        active, indexed = _counts(db)
        if indexed < 0:
            pytest.skip("FTS5 unavailable in this SQLite build")
        assert indexed == active, (
            f"FTS holds {indexed} active rows but the table holds {active}; "
            "the search index drifted under concurrency"
        )


class TestReaderDuringWrites:
    def test_reader_never_sees_a_torn_row(self, tmp_path):
        db = _make_store(tmp_path)
        writers = [
            _spawn(_WRITER, db, f"writer{i}", str(_ROWS_EACH))
            for i in range(_WRITERS)
        ]
        reader = _spawn(_READER, db, "8")

        for proc in writers:
            proc.communicate(timeout=_TIMEOUT_S)
        out, err = reader.communicate(timeout=_TIMEOUT_S)

        assert reader.returncode == 0, f"reader failed:\n{err.strip()[-400:]}"
        payload = json.loads(out.strip().splitlines()[-1])
        # Counts may legitimately differ per round (writes are landing), but
        # every round must have completed without a malformed row.
        assert len(payload["totals"]) == 8
        assert _integrity(db) == "ok"
