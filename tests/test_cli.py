"""Tests for the `clara` CLI (clara/cli.py)."""

from __future__ import annotations

import pytest

from clara import cli


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the global store at a per-test temp file.

    CLAUDE_CONFIG_DIR is isolated too: doctor's host-integration report reads
    the real ~/.claude otherwise, and a machine with a genuinely degraded
    plugin install (the exact state that motivated the report) would flip
    these tests' exit codes.
    """
    monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
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


class TestGraphCli:
    def test_graph_stats_and_show(self, capsys):
        assert _run(["remember", "I use PostgreSQL for the backend"]) == 0
        capsys.readouterr()
        assert _run(["graph", "stats"]) == 0
        out = capsys.readouterr().out
        assert "nodes:" in out and "edges:" in out

        assert _run(["graph", "show", "postgres"]) == 0
        out = capsys.readouterr().out
        assert "postgresql" in out.lower()

    def test_graph_show_unknown_entity(self, capsys):
        assert _run(["remember", "I use PostgreSQL for the backend"]) == 0
        capsys.readouterr()
        assert _run(["graph", "show", "definitely-not-there"]) == 1

    def test_graph_rebuild_and_doctor(self, capsys):
        assert _run(["remember", "I use PostgreSQL for the backend"]) == 0
        capsys.readouterr()
        assert _run(["graph", "rebuild"]) == 0
        assert "graph rebuilt" in capsys.readouterr().out
        assert _run(["graph", "doctor"]) == 0
        assert "no issues" in capsys.readouterr().out


class TestProjectGitignore:
    def test_init_writes_scoped_gitignore(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert _run(["init", "--project", "--agent", "generic"]) == 0
        content = (tmp_path / ".clara" / ".gitignore").read_text(encoding="utf-8")
        assert content == "clara.db*\n.maintenance*\nbackups/\nquarantine/\n"
        assert "warning:" not in capsys.readouterr().err

    def test_init_migrates_star_gitignore(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        clara_dir = tmp_path / ".clara"
        clara_dir.mkdir()
        (clara_dir / ".gitignore").write_text("*\n", encoding="utf-8")
        assert _run(["init", "--project", "--agent", "generic"]) == 0
        content = (clara_dir / ".gitignore").read_text(encoding="utf-8")
        assert content == "clara.db*\n.maintenance*\nbackups/\nquarantine/\n"
        assert capsys.readouterr().err.count("warning:") == 1

    def test_init_leaves_custom_gitignore_alone(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        clara_dir = tmp_path / ".clara"
        clara_dir.mkdir()
        custom = "clara.db\nmy-custom\n"
        (clara_dir / ".gitignore").write_text(custom, encoding="utf-8")
        assert _run(["init", "--project", "--agent", "generic"]) == 0
        assert (clara_dir / ".gitignore").read_text(encoding="utf-8") == custom
        assert "warning:" not in capsys.readouterr().err


class TestAsciiConsole:
    """CLARA must not fail a command because the console is legacy-encoded.

    Windows consoles still run legacy code pages, and LC_ALL=C or
    PYTHONIOENCODING=ascii both produce an ASCII stdout. CLARA's output uses a
    few non-ASCII glyphs, and printing one there raises UnicodeEncodeError.
    argparse does not catch it, so `clara sync` exited 70 *after* completing the
    import -- the work succeeded and the command still reported failure.

    Exercised through a real subprocess: the bug lives in how the interpreter
    configures stdout at startup, which in-process capture replaces and would
    therefore hide.
    """

    def _clara(self, args, cwd, env_extra):
        import os
        import subprocess
        import sys

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "ascii"
        env.pop("CLARA_DB_PATH", None)
        # Drop pytest-cov's hooks: the child would write statement-only
        # coverage beside the parent's branch data, and combining the two
        # fails the whole run with "Can't combine statement coverage data
        # with branch data" -- an INTERNALERROR after every test passed.
        for key in [k for k in env if k.startswith(("COV_CORE", "COVERAGE"))]:
            env.pop(key, None)
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "clara.cli", *args],
            cwd=str(cwd), env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=180,
        )

    def test_sync_survives_an_ascii_stdout(self, tmp_path):
        home = tmp_path / "home"
        (home / ".clara").mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        # The second line does not extract, which is what triggers the
        # "did not extract — re-run with --verbatim" notice containing U+2014.
        (project / "CLAUDE.md").write_text(
            "# Notes\n- We use Kafka for events.\n- Something vague here.\n",
            encoding="utf-8",
        )
        result = self._clara(
            ["sync"], project, {"CLARA_HOME": str(home / ".clara")}
        )
        assert result.returncode == 0, (
            f"clara sync failed on an ASCII console:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "codec can't encode" not in (result.stdout + result.stderr)

    def test_stats_survives_an_ascii_stdout(self, tmp_path):
        home = tmp_path / "home"
        (home / ".clara").mkdir(parents=True)
        result = self._clara(["stats"], tmp_path, {"CLARA_HOME": str(home / ".clara")})
        assert result.returncode == 0, result.stderr
        assert "codec can't encode" not in (result.stdout + result.stderr)
