"""CLI subcommands verified by hand but not by the suite.

A sweep of the whole `clara` surface found no defects, which is the point of
writing these down: `graph path`, `graph export`, `project`, `docs archive` /
`docs restore`, `forget --archive` and the export filters all behaved
correctly and none of them was covered. Behaviour confirmed once by hand and
never asserted is behaviour that regresses quietly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from clara import cli


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo with a manifest and a plan doc, and an isolated store."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True,
                   stdin=subprocess.DEVNULL)
    (root / "package.json").write_text(
        json.dumps({"name": "demo", "dependencies": {"react": "^18"}}), encoding="utf-8"
    )
    (root / "PLAN.md").write_text("# Plan\n- [x] done\n", encoding="utf-8")
    monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
    monkeypatch.setenv("CLARA_HOME", str(tmp_path))
    monkeypatch.chdir(root)
    return root


class TestGraphPath:
    def test_reports_a_multi_hop_path(self, repo, capsys):
        assert _run(["remember", "I use Postgres for storage"]) == 0
        import asyncio
        import os

        from clara.integrations.local_memory import LocalMemory

        async def link():
            memory = await LocalMemory.create(os.environ["CLARA_DB_PATH"])
            try:
                await memory.memory_link(src="api", relation="runs_on", dst="fly.io")
                await memory.memory_link(src="fly.io", relation="hosts", dst="edge")
            finally:
                await memory.close()

        asyncio.run(link())
        capsys.readouterr()

        assert _run(["graph", "path", "api", "edge"]) == 0
        out = capsys.readouterr().out
        assert "2 hops" in out
        assert "api runs_on fly.io" in out
        assert "fly.io hosts edge" in out

    def test_dotted_names_survive(self, repo, capsys):
        """`fly.io` must not be truncated to `fly` — a fixed corruption bug."""
        import asyncio
        import os

        from clara.integrations.local_memory import LocalMemory

        async def link():
            memory = await LocalMemory.create(os.environ["CLARA_DB_PATH"])
            try:
                await memory.memory_link(src="api", relation="runs_on", dst="fly.io")
            finally:
                await memory.close()

        asyncio.run(link())
        capsys.readouterr()
        assert _run(["graph", "show", "fly.io"]) == 0
        assert "fly.io" in capsys.readouterr().out

    def test_absent_path_is_reported_not_crashed(self, repo, capsys):
        assert _run(["remember", "I use Postgres for storage"]) == 0
        capsys.readouterr()
        assert _run(["graph", "path", "nowhere", "nohow"]) == 1
        combined = capsys.readouterr()
        assert "no path" in (combined.out + combined.err).lower()

    def test_export_emits_mermaid(self, repo, capsys):
        assert _run(["remember", "I use Postgres for storage"]) == 0
        capsys.readouterr()
        assert _run(["graph", "export"]) == 0
        assert "graph LR" in capsys.readouterr().out


class TestProjectCommand:
    def test_reads_the_manifest(self, repo, capsys):
        assert _run(["project"]) == 0
        out = capsys.readouterr().out
        assert "demo" in out
        assert "react" in out
        assert "javascript" in out

    def test_evidence_names_the_source_key(self, repo, capsys):
        """The point of --evidence is that every claim is attributable."""
        assert _run(["project", "--evidence"]) == 0
        out = capsys.readouterr().out
        assert "evidence:" in out
        assert "package.json" in out
        assert "dependencies:react" in out

    def test_json_is_machine_readable(self, repo, capsys):
        assert _run(["project", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "demo"
        assert payload["detected"]["framework"] == ["react"]
        assert payload["evidence"]

    def test_no_manifest_says_so(self, tmp_path, monkeypatch, capsys):
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "c.db"))
        monkeypatch.chdir(bare)
        assert _run(["project"]) == 0
        assert "nothing detected" in capsys.readouterr().out


def _commit(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True,
                   capture_output=True, stdin=subprocess.DEVNULL)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


class TestDocsArchiveRestore:
    def test_archive_moves_a_committed_file_and_restore_puts_it_back(
        self, repo, capsys
    ):
        _commit(repo)
        assert _run(["docs", "scan"]) == 0
        capsys.readouterr()

        assert _run(["docs", "archive", "PLAN.md"]) == 0
        assert "file moved" in capsys.readouterr().out
        assert not (repo / "PLAN.md").exists()
        archived = repo / "docs" / "archive" / "PLAN.md"
        assert archived.is_file(), "archive should move the file, not delete it"

        assert _run(["docs", "restore", "docs/archive/PLAN.md"]) == 0
        assert "restored" in capsys.readouterr().out
        assert (repo / "PLAN.md").is_file()

    def test_an_untracked_file_is_archived_in_the_ledger_only(self, repo, capsys):
        """The move is `git mv`, which cannot move a file git does not know.

        Rather than fail the archive, CLARA records the lifecycle transition
        and leaves the file alone — and says so, by omitting "(file moved)".
        Discovered by sweeping the CLI: the same command behaves differently
        before and after the first commit, which is worth pinning so neither
        half changes silently.
        """
        assert _run(["docs", "scan"]) == 0
        capsys.readouterr()

        assert _run(["docs", "archive", "PLAN.md"]) == 0
        out = capsys.readouterr().out
        assert "archived" in out
        assert "file moved" not in out, "an untracked file cannot have been git mv'd"
        assert (repo / "PLAN.md").is_file(), "the untracked file must be left in place"
        assert not (repo / "docs" / "archive" / "PLAN.md").exists()

        capsys.readouterr()
        assert _run(["docs", "status", "PLAN.md"]) == 0
        assert "archived" in capsys.readouterr().out, "the ledger must still record it"


class TestForgetArchive:
    def test_archive_retires_without_deleting(self, repo, capsys):
        import os
        import sqlite3

        assert _run(["remember", "I use Postgres for storage"]) == 0
        out = capsys.readouterr().out
        memory_id = out.strip().split("-> ")[-1].strip()

        assert _run(["forget", memory_id, "--archive"]) == 0
        conn = sqlite3.connect(os.environ["CLARA_DB_PATH"])
        try:
            row = conn.execute(
                "SELECT status FROM memories WHERE memory_id = ?",
                (memory_id.replace("-", ""),),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, "the row was deleted; CLARA retires, never deletes"
        assert row[0] == "archived"


class TestExportFilters:
    def _count(self, captured: str) -> int:
        return sum(
            1 for line in captured.splitlines()
            if line.strip().startswith("{") and '"kind": "memory"' in line
        )

    def test_type_filter_and_out_file(self, repo, capsys, tmp_path):
        assert _run(["remember", "I use Postgres for storage"]) == 0
        assert _run(["remember", "we deployed the api to production"]) == 0
        capsys.readouterr()

        assert _run(["export"]) == 0
        everything = self._count(capsys.readouterr().out)
        assert everything >= 2

        assert _run(["export", "--type", "belief"]) == 0
        beliefs = self._count(capsys.readouterr().out)
        assert 0 < beliefs <= everything

        out_file = tmp_path / "dump.jsonl"
        assert _run(["export", "--out", str(out_file)]) == 0
        assert out_file.is_file()
        # header + one line per memory
        assert len(out_file.read_text(encoding="utf-8").strip().splitlines()) == everything + 1

    def test_since_in_the_future_returns_nothing(self, repo, capsys):
        assert _run(["remember", "I use Postgres for storage"]) == 0
        capsys.readouterr()
        assert _run(["export", "--since", "2099-01-01"]) == 0
        assert self._count(capsys.readouterr().out) == 0
