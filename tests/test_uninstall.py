"""Tests for `clara uninstall`.

Two promises to keep. Without --purge-memories the store survives; with it,
nothing CLARA wrote is left behind. Both were broken in small ways that only
show up by listing the directory afterwards rather than reading the code:

  * the private CPython under pytools/ survived every uninstall when it sat
    directly under the base dir rather than inside the plugin data dir --
    15,147 files on the machine this was found on;
  * quarantine/ and proposals/ (doc-ledger projections, hook fuel) survived;
  * clara.db.stats -- the status-line sidecar holding counts derived from the
    store -- survived --purge-memories, so a command promising irreversible
    deletion left a file describing what was deleted.

There was no coverage for this command at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clara import cli


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A CLARA home with every artifact the runtime is known to create."""
    base = tmp_path / ".clara"
    plugin = base / "plugin"
    for directory in (
        plugin / "venv-abc123",
        base / "pytools" / "python-3.12" / "tools",
        base / "session-flags",
        base / "session-cwd",
        base / "quarantine",
        base / "proposals",
    ):
        directory.mkdir(parents=True)
    (base / "pytools" / "python-3.12" / "tools" / "python.exe").write_bytes(b"x")
    (base / "quarantine" / "repo.tsv").write_text("docs/old.md\tarchived\n")
    (base / "proposals" / "repo.txt").write_text("PLAN.md\n")
    (plugin / "install.log").write_text("log\n")

    monkeypatch.setenv("CLARA_HOME", str(base))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(plugin))
    monkeypatch.setenv("CLARA_DB_PATH", str(base / "clara.db"))

    assert _run(["remember", "I use Postgres for storage"]) == 0
    assert _run(["backup", "--reason", "test"]) == 0
    (base / "clara.db.stats").write_text('{"count": 1}')
    return base


def _leftovers(base: Path) -> set[str]:
    return {p.name for p in base.iterdir()}


class TestKeepsMemories:
    def test_runtime_is_removed_but_the_store_survives(self, installed):
        assert _run(["uninstall"]) == 0
        left = _leftovers(installed)
        assert "clara.db" in left, "uninstall deleted memories without --purge-memories"
        assert "backups" in left
        for gone in ("plugin", "pytools", "session-flags", "session-cwd",
                     "quarantine", "proposals"):
            assert gone not in left, f"{gone} survived uninstall"

    def test_memories_are_still_readable(self, installed, capsys):
        assert _run(["uninstall"]) == 0
        capsys.readouterr()
        assert _run(["list"]) == 0
        assert "Postgres" in capsys.readouterr().out

    def test_it_says_where_the_memories_are(self, installed, capsys):
        assert _run(["uninstall"]) == 0
        out = capsys.readouterr().out
        assert "memories kept at" in out
        assert "--purge-memories" in out


class TestPurge:
    def test_purge_leaves_nothing_behind(self, installed):
        assert _run(["uninstall", "--purge-memories"]) == 0
        assert _leftovers(installed) == set(), (
            "--purge-memories promises irreversible deletion; these survived: "
            f"{_leftovers(installed)}"
        )

    def test_stats_sidecar_is_purged(self, installed):
        """It holds counts derived from the store, and a stale one makes the
        status line report a count for a store that no longer exists."""
        assert (installed / "clara.db.stats").exists()
        assert _run(["uninstall", "--purge-memories"]) == 0
        assert not (installed / "clara.db.stats").exists()


class TestIdempotent:
    def test_second_uninstall_is_clean(self, installed, capsys):
        assert _run(["uninstall", "--purge-memories"]) == 0
        capsys.readouterr()
        assert _run(["uninstall", "--purge-memories"]) == 0
        assert "nothing to remove" in capsys.readouterr().out
