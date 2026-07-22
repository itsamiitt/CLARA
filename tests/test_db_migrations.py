"""Tests for SQLite schema versioning (clara/db/migrations.py)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

import pytest

from clara.db import migrations
from clara.db.migrations import (
    SCHEMA_VERSION,
    SchemaTooNew,
    ensure_schema,
    get_version,
    open_db,
)


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_newer(path) -> None:
    """Create a DB whose schema version is far newer than the code knows."""
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO schema_info (version, migrated_at) VALUES (?, ?)",
        (999, "2099-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()


class TestEnsureSchema:
    def test_fresh_db_reaches_current_version(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "m.db")
        assert ensure_schema(conn) == SCHEMA_VERSION
        rows = conn.execute("SELECT version, migrated_at FROM schema_info").fetchall()
        conn.close()
        assert [row[0] for row in rows] == [1]
        assert datetime.fromisoformat(rows[0][1]).tzinfo is not None

    def test_idempotent_second_run(self, tmp_path):
        db = tmp_path / "m.db"
        conn = sqlite3.connect(db)
        first = ensure_schema(conn)
        conn.close()
        digest = _hash(db)
        conn = sqlite3.connect(db)
        second = ensure_schema(conn)
        count = conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0]
        conn.close()
        assert first == second == SCHEMA_VERSION
        assert count == SCHEMA_VERSION
        assert _hash(db) == digest

    def test_get_version_fresh_db_is_zero(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "v.db")
        assert get_version(conn) == 0
        conn.close()

    def test_newer_schema_raises_and_never_writes(self, tmp_path):
        db = tmp_path / "n.db"
        _seed_newer(db)
        digest = _hash(db)
        conn = sqlite3.connect(db)
        with pytest.raises(SchemaTooNew) as excinfo:
            ensure_schema(conn)
        conn.close()
        assert excinfo.value.found == 999
        assert excinfo.value.supported == SCHEMA_VERSION
        assert _hash(db) == digest

    def test_failed_migration_rolls_back(self, tmp_path, monkeypatch):
        conn = sqlite3.connect(tmp_path / "r.db")
        ensure_schema(conn)

        def boom(c: sqlite3.Connection) -> None:
            c.execute("CREATE TABLE half_done (x INTEGER)")
            raise RuntimeError("boom")

        monkeypatch.setattr(migrations, "_MIGRATIONS", [*migrations._MIGRATIONS, (2, boom)])
        with pytest.raises(RuntimeError, match="boom"):
            ensure_schema(conn)
        assert get_version(conn) == 1
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        conn.close()
        assert "half_done" not in tables


class TestOpenDb:
    def test_fresh_path_is_writable_at_current_version(self, tmp_path):
        conn = open_db(tmp_path / "new.db")
        assert get_version(conn) == SCHEMA_VERSION
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t (x) VALUES (1)")
        assert conn.execute("SELECT x FROM t").fetchone() == (1,)
        conn.close()

    def test_newer_schema_opens_readonly_with_one_stderr_line(self, tmp_path, capsys):
        db = tmp_path / "n.db"
        _seed_newer(db)
        digest = _hash(db)
        conn = open_db(db)
        err = capsys.readouterr().err
        assert "read-only" in err
        assert err.strip() and "\n" not in err.strip()
        assert conn.execute("SELECT MAX(version) FROM schema_info").fetchone()[0] == 999
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO schema_info (version, migrated_at) VALUES (1000, 'x')")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE nope (x INTEGER)")
        conn.close()
        assert _hash(db) == digest

    def test_open_db_windows_path_with_spaces(self, tmp_path):
        target = tmp_path / "dir with space" / "a.db"
        target.parent.mkdir()
        conn = open_db(target)
        assert get_version(conn) == SCHEMA_VERSION
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.close()
