"""What a user sees when the store is corrupt.

This is the moment they most need to be told what to do, and it used to be the
moment CLARA handled worst: `clara doctor` opened with a raw SQLite traceback
from _ensure_versioned_schema, then reported the integrity failure and, in a
separate line, that a backup existed — leaving the reader to connect the two.

Verified end to end: corrupting 2 KB at offset 4096 makes quick_check fail,
doctor exits 2, and following the printed `clara restore <backup>` brings the
store back to exit 0 with the memory intact.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from clara import cli


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


def _corrupt(db: Path) -> None:
    """Overwrite a page well past the header, so the file still opens."""
    with open(db, "r+b") as handle:
        handle.seek(4096)
        handle.write(b"\xde\xad\xbe\xef" * 512)


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "clara.db"
    monkeypatch.setenv("CLARA_DB_PATH", str(db))
    monkeypatch.setenv("CLARA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert _run(["remember", "I use Postgres for storage"]) == 0
    return db


class TestDoctorOnCorruption:
    def test_reports_unusable_and_names_the_restore_command(self, store, capsys):
        assert _run(["backup", "--reason", "safety"]) == 0
        capsys.readouterr()
        _corrupt(store)

        assert _run(["doctor"]) == 2, "a corrupt store must be reported unusable"
        out = capsys.readouterr().out
        assert "sqlite integrity" in out
        assert "what to do:" in out, "doctor stopped at the diagnosis"
        assert "clara restore" in out, "the user must be told the command"
        backups = list((store.parent / "backups").glob("*.db"))
        assert str(backups[0]) in out, "and which file to restore from"

    def test_without_a_backup_it_offers_the_rescue_path(self, store, capsys):
        _corrupt(store)
        capsys.readouterr()

        assert _run(["doctor"]) == 2
        out = capsys.readouterr().out
        assert "no backup found" in out
        assert "clara export" in out and "clara import" in out, (
            "with nothing to restore from, salvage what still reads"
        )

    def test_following_the_advice_actually_recovers(self, store, capsys):
        assert _run(["backup", "--reason", "safety"]) == 0
        backup = next((store.parent / "backups").glob("*.db"))
        _corrupt(store)
        assert _run(["doctor"]) == 2
        capsys.readouterr()

        assert _run(["restore", str(backup), "--force"]) == 0
        assert _run(["doctor"]) == 0, "the store is still not healthy after restore"
        capsys.readouterr()
        assert _run(["list"]) == 0
        assert "Postgres" in capsys.readouterr().out, "the memory did not come back"

    def test_a_healthy_store_gets_no_recovery_block(self, store, capsys):
        assert _run(["doctor"]) == 0
        assert "what to do:" not in capsys.readouterr().out


class TestNoTracebacksByDefault:
    """A stack trace is not a diagnosis for the person reading it."""

    def _doctor(self, env_extra: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        # Drop pytest-cov's hooks: the child would write statement-only
        # coverage beside the parent's branch data, and combining the two ends
        # the run in "Can't combine statement coverage data with branch data"
        # — an INTERNALERROR raised after every test has already passed.
        for key in [k for k in env if k.startswith(("COV_CORE", "COVERAGE"))]:
            env.pop(key, None)
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "clara.cli", "doctor"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
            stdin=subprocess.DEVNULL, timeout=180,
        )

    def test_corrupt_store_prints_no_traceback(self, store, tmp_path):
        _corrupt(store)
        result = self._doctor({"CLARA_DB_PATH": str(store), "CLARA_HOME": str(tmp_path)},
                              tmp_path)
        assert "Traceback" not in result.stderr, result.stderr
        assert "sqlite3.DatabaseError" not in result.stderr
        # the cause is still reported, just not as a stack
        assert "clara:" in result.stderr

    def test_clara_debug_restores_the_traceback(self, store, tmp_path):
        """The detail is kept for whoever is actually debugging."""
        _corrupt(store)
        result = self._doctor(
            {"CLARA_DB_PATH": str(store), "CLARA_HOME": str(tmp_path), "CLARA_DEBUG": "1"},
            tmp_path,
        )
        assert "Traceback" in result.stderr


class TestCorruptionIsDetectable:
    def test_the_fixture_really_corrupts_the_file(self, store):
        """Guard the guard: if the write stopped breaking the file, every test
        above would pass while asserting nothing."""
        conn = sqlite3.connect(store)
        try:
            assert conn.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
        finally:
            conn.close()
        _corrupt(store)
        conn = sqlite3.connect(store)
        try:
            assert conn.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok"
        finally:
            conn.close()
