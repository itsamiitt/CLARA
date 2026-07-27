"""`clara maintain` — housekeeping for stores the MCP server never opens.

The pass rides the first store access of the day, but only the MCP server ever
triggered it. Verified before this existed: `clara remember` followed by four
`clara stats` left no .maintenance marker and no daily backup, so a CLI-only
user got none of the decay, pruning or rotated backups the README promised.

The pass itself is unchanged and still lives behind the same O_EXCL
single-winner lock; only the anchor differs between callers, and it is now a
parameter rather than something the module decides.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clara import cli


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "clara.db"
    monkeypatch.setenv("CLARA_DB_PATH", str(db))
    monkeypatch.setenv("CLARA_HOME", str(tmp_path))
    # The pass ends with a native-memory export; keep it off the real ~/.claude.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.chdir(tmp_path)
    assert _run(["remember", "I use Postgres for storage"]) == 0
    return db


def _marker(db: Path) -> Path:
    return Path(str(db) + ".maintenance")


class TestMaintain:
    def test_a_cli_only_store_gets_maintained(self, store, capsys):
        assert not _marker(store).exists(), "precondition: nothing has run yet"
        assert not (store.parent / "backups").exists()

        assert _run(["maintain"]) == 0
        out = capsys.readouterr().out
        assert "decay:" in out and "prune:" in out
        assert _marker(store).exists(), "the cadence marker was not written"
        backups = list((store.parent / "backups").glob("*.db"))
        assert len(backups) == 1, backups
        assert "daily" in backups[0].name

    def test_second_run_is_a_no_op(self, store, capsys):
        assert _run(["maintain"]) == 0
        capsys.readouterr()
        assert _run(["maintain"]) == 0
        out = capsys.readouterr().out
        assert "already ran" in out
        assert "--force" in out, "the message must name the override"
        assert len(list((store.parent / "backups").glob("*.db"))) == 1

    def test_force_overrides_the_daily_gate(self, store, capsys):
        """Asserts the pass ran again, not how many files it left.

        Backup names are second-resolution (clara-%Y%m%dT%H%M%SZ-reason.db), so
        two runs inside the same second write the same filename and the second
        replaces the first. Harmless -- both are snapshots of near-identical
        state -- but it makes a file count a clock-speed assertion.
        """
        assert _run(["maintain"]) == 0
        first = _marker(store).stat().st_mtime_ns
        capsys.readouterr()

        assert _run(["maintain", "--force"]) == 0
        out = capsys.readouterr().out
        assert "already ran" not in out, "--force was refused"
        assert "decay:" in out, "the pass did not actually run"
        assert _marker(store).stat().st_mtime_ns >= first

    def test_the_lock_is_always_released(self, store):
        assert _run(["maintain"]) == 0
        assert not Path(str(store) + ".maintenance.lock").exists()

    def test_memories_survive_a_pass(self, store):
        assert _run(["maintain", "--force"]) == 0
        conn = sqlite3.connect(store)
        try:
            assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        finally:
            conn.close()


class TestSharedWithTheServer:
    def test_the_mcp_server_delegates_to_the_same_pass(self):
        """One implementation, two anchors.

        A second copy would drift, and this one holds the lock protocol that
        keeps concurrent sessions from running it twice.
        """
        import inspect

        from clara.integrations import mcp_server

        source = inspect.getsource(mcp_server._run_maintenance_if_due)
        assert "run_if_due" in source
        assert "_session_anchor" in source, "the server must pass its own anchor"

    def test_anchor_is_required(self):
        """It must not silently default to cwd: the server's cwd can be stale
        while the client has moved, which would resolve the wrong store."""
        import inspect

        from clara.maintenance import run_if_due

        parameters = inspect.signature(run_if_due).parameters
        assert parameters["anchor"].default is inspect.Parameter.empty
        assert parameters["anchor"].kind is inspect.Parameter.KEYWORD_ONLY
