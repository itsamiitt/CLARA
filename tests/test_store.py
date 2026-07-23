"""Tests for the unified store resolver (clara/store.py) — the split-brain fix."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clara.store import (
    StoreResolution,
    git_toplevel,
    global_db_path,
    orphaned_project_stores,
    resolve_store,
)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("CLARA_DB_PATH", raising=False)
    monkeypatch.setenv("CLARA_HOME", str(home / ".clara"))
    return home


class TestResolutionOrder:
    def test_explicit_env_wins_over_everything(self, tmp_path, monkeypatch, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / ".clara").mkdir()
        (repo / ".clara" / "clara.db").touch()
        explicit = tmp_path / "explicit.db"
        monkeypatch.setenv("CLARA_DB_PATH", str(explicit))
        res = resolve_store(str(repo))
        assert res.scope == "explicit"
        assert res.db_path == explicit
        assert res.exists is False

    def test_project_store_when_file_exists(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / ".clara").mkdir()
        (repo / ".clara" / "clara.db").touch()
        res = resolve_store(str(repo))
        assert res.scope == "project"
        assert res.db_path == repo / ".clara" / "clara.db"
        assert res.exists is True

    def test_project_store_found_from_subdirectory(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        sub = repo / "src" / "deep"
        sub.mkdir(parents=True)
        _git_init(repo)
        (repo / ".clara").mkdir()
        (repo / ".clara" / "clara.db").touch()
        res = resolve_store(str(sub))
        assert res.scope == "project"
        assert res.db_path == repo / ".clara" / "clara.db"

    def test_global_when_no_project_file(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        res = resolve_store(str(repo))
        assert res.scope == "global"
        assert res.db_path == global_db_path()

    def test_non_repo_dir_resolves_global(self, tmp_path, isolated_home):
        plain = tmp_path / "plain"
        plain.mkdir()
        res = resolve_store(str(plain))
        assert res.scope == "global"
        assert res.repo_root is None

    def test_create_mkdirs_parent(self, tmp_path, isolated_home):
        res = resolve_store(str(tmp_path), create=True)
        assert res.db_path.parent.is_dir()

    def test_worktree_shares_toplevel(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "f.txt").write_text("x")
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
             "add", "."],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "init"],
            check=True, capture_output=True,
        )
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-q", str(wt)],
            check=True, capture_output=True,
        )
        # A worktree has its own toplevel; each may carry its own project
        # store, but with none present both resolve global.
        assert resolve_store(str(wt)).scope == "global"


class TestThreeWayParity:
    """fastpath, MCP server, and CLI must target the same file."""

    def test_all_paths_agree(self, tmp_path, monkeypatch, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / ".clara").mkdir()
        (repo / ".clara" / "clara.db").touch()
        monkeypatch.chdir(repo)

        from clara.fastpath import db as fastdb
        from clara.integrations import mcp_server

        fast_path, _rid = fastdb.resolve_store(str(repo))
        unified = resolve_store(str(repo))
        # default_db_path resolves from cwd — chdir above pins it to repo.
        mcp_path = Path(mcp_server.default_db_path())
        assert fast_path == unified.db_path
        assert mcp_path == unified.db_path

    def test_global_agreement_without_project(self, tmp_path, monkeypatch, isolated_home):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)

        from clara.fastpath import db as fastdb
        from clara.integrations import mcp_server

        fast_path, _rid = fastdb.resolve_store(str(plain))
        assert fast_path is None  # global store file does not exist yet
        mcp_path = Path(mcp_server.default_db_path())  # creates parent
        assert mcp_path == global_db_path()
        # After the writer created it, the fastpath sees the same file.
        mcp_path.touch()
        fast_path2, _ = fastdb.resolve_store(str(plain))
        assert fast_path2 == mcp_path


class TestOrphanDetection:
    def test_orphaned_subdir_store_reported(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        sub = repo / "src"
        sub.mkdir(parents=True)
        _git_init(repo)
        (sub / ".clara").mkdir()
        (sub / ".clara" / "clara.db").touch()
        res = resolve_store(str(sub))
        orphans = orphaned_project_stores(str(sub), res.db_path)
        assert [o for o in orphans if o == sub / ".clara" / "clara.db"]

    def test_no_orphans_when_store_is_resolved(self, tmp_path, isolated_home):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / ".clara").mkdir()
        (repo / ".clara" / "clara.db").touch()
        res = resolve_store(str(repo))
        assert orphaned_project_stores(str(repo), res.db_path) == []


class TestGitToplevel:
    def test_inside_and_outside(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        top = git_toplevel(str(repo))
        assert top is not None
        assert Path(top).resolve() == repo.resolve()
        plain = tmp_path / "plain"
        plain.mkdir()
        assert git_toplevel(str(plain)) is None


class TestResolutionDataclass:
    def test_frozen(self, tmp_path, isolated_home):
        res = resolve_store(str(tmp_path))
        assert isinstance(res, StoreResolution)
        with pytest.raises(Exception):
            res.scope = "hacked"  # type: ignore[misc]
