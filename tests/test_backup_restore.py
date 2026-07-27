"""Tests for `clara backup` / `clara restore`.

These are the commands a user reaches for when something has already gone
wrong, so the failure modes matter more than the happy path: restore must
refuse anything that is not a CLARA store, must not touch the live store when
it refuses, and must take a pre-restore backup before it does replace it.

There was no coverage here at all before.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clara import cli
from clara.db.migrations import SCHEMA_VERSION, ensure_schema


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "clara.db"
    monkeypatch.setenv("CLARA_DB_PATH", str(db))
    monkeypatch.delenv("CLARA_HOME", raising=False)
    return db


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


def _seed(count: int = 2) -> None:
    for i in range(count):
        assert _run(["remember", f"I use Tool{i} for testing"]) == 0


def _rows(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    finally:
        conn.close()


def _backups(db: Path) -> list[Path]:
    return sorted((db.parent / "backups").glob("*.db"))


class TestRoundTrip:
    def test_backup_then_restore_recovers_exactly(self, store, capsys):
        _seed(2)
        assert _run(["backup", "--reason", "test"]) == 0
        backup = _backups(store)[-1]
        baseline = _rows(store)

        _seed(1)
        assert _rows(store) == baseline + 1

        capsys.readouterr()
        assert _run(["restore", str(backup), "--force"]) == 0
        assert _rows(store) == baseline

    def test_restore_takes_a_pre_restore_backup(self, store):
        """The current store is worth saving too: restoring the wrong file is
        the most likely way to reach for restore a second time."""
        _seed(1)
        assert _run(["backup", "--reason", "test"]) == 0
        backup = _backups(store)[-1]

        assert _run(["restore", str(backup), "--force"]) == 0
        reasons = [p.name for p in _backups(store)]
        assert any("pre-restore" in name for name in reasons), reasons


class TestRestoreRefusesBadInput:
    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("junk.txt", b"this is not a database\n"),
            ("truncated.db", b"SQLite format 3\x00" + b"\x00" * 64),
            ("export.jsonl", b'{"kind": "header", "format": 1}\n'),
        ],
    )
    def test_not_a_store_exits_1_and_leaves_the_store_alone(
        self, store, tmp_path, name, content, capsys
    ):
        """Exit 1, not 70.

        PRAGMA integrity_check raises sqlite3.DatabaseError on a non-SQLite
        file. Uncaught it reached the top-level handler and exited 70, which
        this CLI documents as "unexpected internal failure" -- telling the user
        to report a bug when the fix is to pass a different path.
        """
        _seed(2)
        before = store.read_bytes()
        bad = tmp_path / name
        bad.write_bytes(content)

        capsys.readouterr()
        assert _run(["restore", str(bad), "--force"]) == 1
        err = capsys.readouterr().err
        assert "not a CLARA store" in err
        assert store.read_bytes() == before, "a refused restore modified the store"

    def test_missing_file_exits_1(self, store, tmp_path, capsys):
        _seed(1)
        assert _run(["restore", str(tmp_path / "nope.db"), "--force"]) == 1
        assert "no such file" in capsys.readouterr().err

    def test_newer_schema_is_refused(self, store, tmp_path, capsys):
        """Restoring a store from a newer CLARA would hand this build a schema
        it cannot migrate; the same rule as clara/db/migrations.py."""
        _seed(1)
        future = tmp_path / "future.db"
        conn = sqlite3.connect(future)
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO schema_info (version, migrated_at) VALUES (?, '2030-01-01')",
            (SCHEMA_VERSION + 91,),
        )
        conn.commit()
        conn.close()

        before = store.read_bytes()
        capsys.readouterr()
        assert _run(["restore", str(future), "--force"]) == 1
        assert "newer than this CLARA supports" in capsys.readouterr().err
        assert store.read_bytes() == before

    def test_without_force_it_refuses_and_says_so(self, store, capsys):
        _seed(1)
        assert _run(["backup", "--reason", "test"]) == 0
        backup = _backups(store)[-1]
        before = store.read_bytes()

        capsys.readouterr()
        assert _run(["restore", str(backup)]) == 1
        assert "--force" in capsys.readouterr().out
        assert store.read_bytes() == before
