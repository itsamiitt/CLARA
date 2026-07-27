"""Tests for the unified store resolver (clara/store.py) — the split-brain fix."""

from __future__ import annotations

import dataclasses
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
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.scope = "hacked"  # type: ignore[misc]


class TestProjectStoreWithoutGit:
    """Store selection must not depend on a git subprocess succeeding.

    Regression: `git rev-parse` was timing out inside the MCP server (it
    inherited the transport's stdin), so `git_toplevel` returned None on every
    call. `resolve_store` then fell back to the anchor itself, which meant a
    session opened in a subdirectory silently missed its project store and
    read and wrote the global one instead.
    """

    def _repo_with_project_store(self, tmp_path):
        subprocess.run(
            ["git", "init", "-q", str(tmp_path)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        (tmp_path / ".clara").mkdir()
        (tmp_path / ".clara" / "clara.db").write_bytes(b"")
        nested = tmp_path / "src" / "deep"
        nested.mkdir(parents=True)
        return nested

    def test_subdirectory_finds_project_store_with_git(self, tmp_path, isolated_home):
        nested = self._repo_with_project_store(tmp_path)
        assert resolve_store(str(nested)).scope == "project"

    def test_subdirectory_finds_project_store_without_git(
        self, tmp_path, isolated_home, monkeypatch
    ):
        nested = self._repo_with_project_store(tmp_path)
        monkeypatch.setattr("clara.store.git_toplevel", lambda _cwd: None)
        resolution = resolve_store(str(nested))
        assert resolution.scope == "project"
        assert resolution.db_path == tmp_path / ".clara" / "clara.db"

    def test_walk_up_never_claims_the_global_store(
        self, tmp_path, isolated_home, monkeypatch
    ):
        # ~/.clara/clara.db sits above every path under the home directory, so
        # an unrelated directory must not walk up, find it, and report it as a
        # *project* store. Path.home() is patched as well as CLARA_HOME: the
        # walk consults the real home otherwise, and an earlier version of this
        # test wrote into it.
        fake_home = tmp_path / "home"
        (fake_home / ".clara").mkdir(parents=True)
        (fake_home / ".clara" / "clara.db").write_bytes(b"")
        monkeypatch.setenv("CLARA_HOME", str(fake_home / ".clara"))
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: fake_home))

        unrelated = fake_home / "somewhere" / "else"
        unrelated.mkdir(parents=True)
        monkeypatch.setattr("clara.store.git_toplevel", lambda _cwd: None)
        assert resolve_store(str(unrelated)).scope == "global"

    def test_no_project_store_still_resolves_global(
        self, tmp_path, isolated_home, monkeypatch
    ):
        monkeypatch.setattr("clara.store.git_toplevel", lambda _cwd: None)
        assert resolve_store(str(tmp_path)).scope == "global"


class TestNoPrivateReachIns:
    """Subsystems must use LocalMemory's public accessors, not its privates.

    Audit finding A2: the CLI, docs, bridge and MCP layers all programmed
    against ``_session_factory`` / ``_db_path`` / ``_engine`` because no public
    accessor existed, which made every internal refactor a breaking change for
    four modules. The accessors were added first and the callers migrated
    afterwards; without this guard the coupling quietly grows back.
    """

    def test_no_module_reaches_into_the_privates(self):
        import re

        root = Path(__file__).parents[1] / "clara"
        owners = {"local_memory.py", "agent.py"}  # may touch their own state
        # Only flag access through *another* object. `self._session_factory` is
        # a class holding its own state (DecayScheduler, BackgroundWriter) and
        # is not the coupling this guards against.
        pattern = re.compile(
            r"\b(?!self\b)\w+\.(_session_factory|_db_path|_engine)\b"
        )
        offenders = []
        for path in root.rglob("*.py"):
            if path.name in owners:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                match = pattern.search(line)
                if match:
                    offenders.append(
                        f"{path.relative_to(root)}:{lineno} {match.group(0)}"
                    )
        assert not offenders, (
            "use the public accessors (session(), session_factory, db_path, "
            "engine) instead of reaching through privates:\n  "
            + "\n  ".join(offenders)
        )

