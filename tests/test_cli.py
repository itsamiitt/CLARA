"""Tests for the `clara` CLI (clara/cli.py)."""

from __future__ import annotations

import pytest

from clara import cli


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the global store at a per-test temp file."""
    monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
    yield


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


class TestInit:
    def test_init_creates_store(self, tmp_path, capsys):
        assert _run(["init", "--agent", "claude-code"]) == 0
        out = capsys.readouterr().out
        assert "CLARA store ready" in out
        assert "claude mcp add clara" in out
        assert (tmp_path / "clara.db").exists()

    def test_init_project_scoped(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert _run(["init", "--project", "--agent", "generic"]) == 0
        assert (tmp_path / ".clara" / ".gitignore").exists()


class TestRememberAndRecall:
    def test_remember_then_context(self, capsys):
        assert _run(["remember", "I use PostgreSQL for the backend"]) == 0
        out = capsys.readouterr().out
        assert "saved [belief] user uses PostgreSQL" in out

        assert _run(["context", "postgresql"]) == 0
        out = capsys.readouterr().out
        assert "MEMORY CONTEXT" in out
        assert "PostgreSQL" in out

    def test_remember_event_and_world_model(self, capsys):
        assert _run(["remember", "We deployed the payments service. "
                     "The api service runs on Fly.io"]) == 0
        out = capsys.readouterr().out
        assert "saved [event]" in out
        assert "saved [world_model]" in out

    def test_remember_nothing_recognized(self, capsys):
        assert _run(["remember", "hello there"]) == 1
        assert "Nothing durable recognized" in capsys.readouterr().out

    def test_negation_switch(self, capsys):
        assert _run(["remember", "I switched from npm to pnpm"]) == 0
        out = capsys.readouterr().out
        assert "(negation)" in out


class TestListForgetStats:
    def test_list_and_forget(self, capsys):
        _run(["remember", "I use Rust"])
        capsys.readouterr()
        assert _run(["list"]) == 0
        out = capsys.readouterr().out
        assert "[belief]" in out
        memory_id = out.split()[0]

        assert _run(["forget", memory_id]) == 0
        capsys.readouterr()
        assert _run(["context", "rust"]) == 0
        assert "Rust" not in capsys.readouterr().out

    def test_forget_unknown_id(self, capsys):
        assert _run(["forget", "00000000-0000-0000-0000-000000000000"]) == 2

    def test_stats(self, capsys):
        assert _run(["stats"]) == 0
        assert "active_by_type" in capsys.readouterr().out


class TestDoctor:
    def test_doctor_healthy(self, capsys):
        assert _run(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "[ok] sqlite integrity" in out
        assert "[ok] fts5 index" in out

    def test_doctor_quiet(self, capsys):
        assert _run(["doctor", "--quiet"]) == 0
        assert capsys.readouterr().out == ""
