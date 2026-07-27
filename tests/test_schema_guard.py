"""A store from a newer CLARA must never be written to.

clara/db/migrations.py states the rule: "if the database's version is NEWER
than this code knows (SCHEMA_VERSION), never write". ensure_schema raises
SchemaTooNew to enforce it.

LocalMemory did not honour it. _ensure_versioned_schema caught every exception
as "migrations could not run, degrade gracefully", which is right for a
read-only mount or a locked file and exactly wrong here -- those mean "no
migration", this means "the file belongs to a schema you do not understand".
It logged a traceback and carried on, and create() then ran create_all and
ensure_fts, applying this build's older DDL on top. Measured against a store
marked v99: the row count still went 1 -> 2.

The store is now opened read-only instead: reads keep working, SQLite itself
refuses writes, so a downgrade degrades visibly rather than corrupting data.
"""

from __future__ import annotations

import sqlite3

import pytest

from clara.db.migrations import SCHEMA_VERSION, SchemaTooNew, ensure_schema
from clara.integrations.local_memory import LocalMemory


def _store_from_the_future(path) -> str:
    """A valid store whose recorded schema version is newer than we support."""
    db = str(path / "future.db")
    conn = sqlite3.connect(db)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO schema_info (version, migrated_at) VALUES (?, '2030-01-01')",
        (SCHEMA_VERSION + 91,),
    )
    conn.commit()
    conn.close()
    return db


def _rows(db: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    finally:
        conn.close()


class TestEnsureSchemaContract:
    def test_ensure_schema_refuses_a_newer_store(self, tmp_path):
        db = _store_from_the_future(tmp_path)
        conn = sqlite3.connect(db)
        try:
            with pytest.raises(SchemaTooNew):
                ensure_schema(conn)
        finally:
            conn.close()


@pytest.mark.asyncio
class TestLocalMemoryHonoursIt:
    async def test_opens_read_only(self, tmp_path):
        memory = await LocalMemory.create(_store_from_the_future(tmp_path))
        try:
            assert memory.read_only is True
        finally:
            await memory.close()

    async def test_write_is_refused_and_nothing_lands(self, tmp_path):
        db = _store_from_the_future(tmp_path)
        before = _rows(db)
        memory = await LocalMemory.create(db)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                await memory.save(
                    subject="user", relation="uses", object="Redis", domain="caching"
                )
            # Refused before any SQL runs. SQLite would reject it anyway, but
            # only after emitting the full INSERT and every bound parameter,
            # which over MCP is what the model receives as the tool result.
            message = str(excinfo.value)
            assert "newer version of CLARA" in message
            assert "pip install -U clara-memory" in message
            assert "readonly database" not in message
        finally:
            await memory.close()
        assert _rows(db) == before, "a write landed in a store from a newer CLARA"

    async def test_reads_still_work(self, tmp_path):
        """Read-only, not unusable: the user keeps access to what they have."""
        db = str(tmp_path / "clara.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(subject="user", relation="uses", object="Postgres")
        finally:
            await memory.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO schema_info (version, migrated_at) VALUES (?, '2030-01-01')",
            (SCHEMA_VERSION + 91,),
        )
        conn.commit()
        conn.close()

        memory = await LocalMemory.create(db)
        try:
            assert memory.read_only is True
            found = await memory.search("Postgres")
            assert found["total"] >= 1
        finally:
            await memory.close()

    async def test_a_current_store_is_writable(self, tmp_path):
        """The guard must not fire on an ordinary store."""
        db = str(tmp_path / "normal.db")
        memory = await LocalMemory.create(db)
        try:
            assert memory.read_only is False
            await memory.save(subject="user", relation="uses", object="Rust")
        finally:
            await memory.close()
        assert _rows(db) == 1

    async def test_graph_augmented_search_still_works_read_only(self, tmp_path):
        """Reads that touch the graph must not be caught by the write guard.

        The guard was first inserted by pattern-matching and one call landed in
        _graph_augment -- a read-side helper that renders the [GRAPH] section of
        a search result. search() wraps that call in `except Exception` so the
        graph can never break search, which means the misplacement did not fail
        anything loudly: it silently dropped the [GRAPH] section on read-only
        stores. So this asserts the section is actually there, not merely that
        the search returned. Asserting only `total >= 1` passes either way --
        verified, that first version of this test could not fail.
        """
        db = str(tmp_path / "clara.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(subject="api", relation="runs_on", object="fly.io")
            await memory.save(subject="api", relation="uses", object="Postgres")
        finally:
            await memory.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO schema_info (version, migrated_at) VALUES (?, '2030-01-01')",
            (SCHEMA_VERSION + 91,),
        )
        conn.commit()
        conn.close()

        memory = await LocalMemory.create(db)
        try:
            assert memory.read_only is True
            found = await memory.search("api", graph_depth=1)
            assert found["total"] >= 1
            assert "graph" in found, "graph augmentation was suppressed read-only"
            assert "[GRAPH]" in found["context"]
        finally:
            await memory.close()


class TestReadOnlyCauseIsDiagnosedCorrectly:
    """"Cannot write" has two causes that need opposite advice.

    SQLite reports both as "attempt to write a readonly database":

      * the schema is newer than this build, so CLARA opened the store
        read-only on purpose -> upgrade clara-memory;
      * the file itself is unwritable (chmod 444, read-only mount, full disk)
        -> fix the file.

    An earlier version of the CLI matched that SQLite string and blamed the
    first for both, so a permissions problem told the user to run
    `pip install -U clara-memory` -- advice that cannot possibly help. Verified
    against a chmod 444 store: that is exactly what it printed.
    """

    def _run(self, argv):
        import pytest as _pytest

        from clara import cli

        with _pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        return int(excinfo.value.code or 0)

    def test_unwritable_file_blames_the_file(self, tmp_path, monkeypatch, capsys):
        import os
        import stat

        db = tmp_path / "clara.db"
        monkeypatch.setenv("CLARA_DB_PATH", str(db))
        monkeypatch.setenv("CLARA_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        assert self._run(["remember", "I use Postgres for storage"]) == 0
        os.chmod(db, stat.S_IREAD)
        capsys.readouterr()

        try:
            assert self._run(["remember", "I use Redis for caching"]) == 1
            err = capsys.readouterr().err
            assert "cannot be written" in err
            assert "pip install" not in err, (
                "a permissions problem must not tell the user to upgrade CLARA"
            )
        finally:
            os.chmod(db, stat.S_IWRITE | stat.S_IREAD)

    def test_newer_schema_blames_the_version(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "clara.db"
        monkeypatch.setenv("CLARA_DB_PATH", str(db))
        monkeypatch.setenv("CLARA_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        assert self._run(["remember", "I use Postgres for storage"]) == 0

        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO schema_info (version, migrated_at) VALUES (?, '2030-01-01')",
            (SCHEMA_VERSION + 91,),
        )
        conn.commit()
        conn.close()
        capsys.readouterr()

        assert self._run(["remember", "I use Kafka for events"]) == 1, (
            "a deliberate refusal is a failed operation (1), not an internal "
            "error (70)"
        )
        err = capsys.readouterr().err
        assert "newer version of CLARA" in err
        assert "chmod" not in err, "this is not a permissions problem"

    def test_the_refusal_has_its_own_exception_type(self):
        """Matched by type, not by message text — the text is user-facing copy
        and will be reworded."""
        from clara.integrations.local_memory import StoreReadOnly

        assert issubclass(StoreReadOnly, RuntimeError)
