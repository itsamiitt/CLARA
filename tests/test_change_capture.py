"""
Change capture + Stop flush — the journal's live event sources.

The journal and its indexer worker existed since the code index shipped, but
nothing fed the journal during a session: everything waited for the daily
maintenance walk. change_capture (PostToolUse) enqueues file and repo events
the moment they happen, and stop_flush (Stop) drains a bounded batch so the
index stays hot without a daemon.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from clara.db.migrations import ensure_schema
from clara.fastpath import change_capture, stop_flush
from clara.repoid import repo_id

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git")


def _git(repo: Path, *args: str) -> None:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr


@pytest.fixture()
def repo(tmp_path):
    """A git repo with one committed python file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / "mod.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--quiet", "-m", "init")
    return root


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A schema-complete store the resolver finds via CLARA_DB_PATH, plus an
    isolated CLARA_HOME for the sidecar flags."""
    db = tmp_path / "store" / "clara.db"
    db.parent.mkdir()
    conn = sqlite3.connect(db)
    ensure_schema(conn)
    conn.close()
    monkeypatch.setenv("CLARA_DB_PATH", str(db))
    monkeypatch.setenv("CLARA_HOME", str(tmp_path / "clara-home"))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    return db


def _journal_rows(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT repo_id, path, change FROM change_journal ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()


class TestClassifyBash:
    @pytest.mark.parametrize(("command", "expected"), [
        ("git commit -m 'x'", "git"),
        ("git checkout main", "git"),
        ("git rebase origin/main", "git"),
        ("git status", None),
        ("git log --oneline", None),
        ("pnpm add react", "manifest"),
        ("npm install", "manifest"),
        ("pip install requests", "manifest"),
        ("uv pip install requests", "manifest"),
        ("uv add httpx", "manifest"),
        ("uv remove httpx", "manifest"),
        ("uv sync", "manifest"),
        ("poetry add httpx", "manifest"),
        ("go get example.com/pkg", "manifest"),
        ("ls -la", None),
        ("python -m pytest", None),
    ])
    def test_classification(self, command, expected):
        assert change_capture.classify_bash(command) == expected


@requires_git
class TestCapture:
    def test_edit_enqueues_the_file(self, repo, store):
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(repo / "mod.py")},
        }
        assert change_capture.capture(payload, str(repo)) is True
        rows = _journal_rows(store)
        assert rows == [(repo_id(str(repo)), "mod.py", "modified")]

    def test_edit_marks_the_dirty_flag(self, repo, store, tmp_path):
        payload = {
            "tool_name": "Write",
            "tool_input": {"file_path": str(repo / "mod.py")},
        }
        assert change_capture.capture(payload, str(repo)) is True
        flag = tmp_path / "clara-home" / "journal-dirty" / repo_id(str(repo))
        assert flag.is_file()

    def test_bash_git_mutation_enqueues_repo_level(self, repo, store):
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m done"},
        }
        assert change_capture.capture(payload, str(repo)) is True
        rows = _journal_rows(store)
        assert rows == [(repo_id(str(repo)), None, "git")]

    def test_non_indexable_suffix_is_skipped(self, repo, store):
        (repo / "notes.md").write_text("x", encoding="utf-8")
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(repo / "notes.md")},
        }
        assert change_capture.capture(payload, str(repo)) is False
        assert _journal_rows(store) == []

    def test_file_outside_the_repo_is_skipped(self, repo, store, tmp_path):
        outside = tmp_path / "elsewhere.py"
        outside.write_text("x = 1\n", encoding="utf-8")
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(outside)},
        }
        assert change_capture.capture(payload, str(repo)) is False
        assert _journal_rows(store) == []

    def test_missing_store_writes_nothing(self, repo, tmp_path, monkeypatch):
        absent = tmp_path / "no-store" / "clara.db"
        monkeypatch.setenv("CLARA_DB_PATH", str(absent))
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "clara-home"))
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(repo / "mod.py")},
        }
        assert change_capture.capture(payload, str(repo)) is False
        assert not absent.exists()


@requires_git
class TestStopFlush:
    def test_flush_drains_and_indexes(self, repo, store, tmp_path):
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(repo / "mod.py")},
        }
        assert change_capture.capture(payload, str(repo)) is True

        stop_flush.flush(str(repo), "test-session")

        assert _journal_rows(store) == []
        conn = sqlite3.connect(store)
        try:
            nodes = conn.execute(
                "SELECT count(*) FROM code_nodes WHERE repo_id = ?",
                (repo_id(str(repo)),),
            ).fetchone()[0]
        finally:
            conn.close()
        assert nodes > 0, "draining the journal must index the journalled file"
        flag = tmp_path / "clara-home" / "journal-dirty" / repo_id(str(repo))
        assert not flag.exists(), "an empty queue must clear the dirty flag"
        cursor = stop_flush._cursor_file(repo_id(str(repo)), str(repo))
        assert cursor.is_file(), "the flush must record the drained HEAD"

    def test_head_delta_enqueues_outside_edits(self, repo, store, tmp_path):
        rid = repo_id(str(repo))
        # First flush: records the cursor at the current HEAD.
        stop_flush.flush(str(repo), "s1")
        # An "outside" change: commit made with no session hook watching.
        (repo / "mod.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "--quiet", "-m", "outside edit")

        conn = sqlite3.connect(store)
        try:
            new_head = stop_flush.enqueue_head_delta(conn, rid, str(repo))
        finally:
            conn.close()
        assert new_head is not None
        assert (rid, "mod.py", "modified") in _journal_rows(store)
