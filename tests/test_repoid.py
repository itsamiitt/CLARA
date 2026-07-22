"""Tests for stable repository identity (clara/repoid.py)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess

import pytest

from clara.repoid import normalize_git_url, repo_id

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(cwd, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _make_repo(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "-c", "init.defaultBranch=main", "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "--allow-empty", "-m", "init")


def _sha16(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class TestNormalizeGitUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("git@github.com:Acme/Widgets.git", "github.com/acme/widgets"),
            ("https://github.com/acme/widgets.git", "github.com/acme/widgets"),
            ("ssh://git@github.com/acme/widgets", "github.com/acme/widgets"),
            ("https://user@host:8443/org/repo/", "host/org/repo"),
            ("C:/repos/x.git", "c:/repos/x"),
            ("/srv/git/repo.git", "srv/git/repo"),
        ],
    )
    def test_normalize(self, url, expected):
        assert normalize_git_url(url) == expected


class TestRepoId:
    @requires_git
    def test_id_is_sha256_prefix_of_root_commit(self, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo)
        root = _git(repo, "rev-list", "--max-parents=0", "HEAD")
        assert repo_id(repo) == _sha16(root)

    @requires_git
    def test_two_worktrees_same_id(self, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo)
        worktree = tmp_path / "wt"
        _git(repo, "worktree", "add", str(worktree))
        assert repo_id(repo) == repo_id(worktree)
        assert re.fullmatch(r"[0-9a-f]{16}", repo_id(repo))

    @requires_git
    def test_subdirectory_same_id(self, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo)
        sub = repo / "sub"
        sub.mkdir()
        assert repo_id(sub) == repo_id(repo)

    @requires_git
    def test_no_commits_falls_back_to_origin_url(self, tmp_path):
        repo = tmp_path / "empty"
        repo.mkdir()
        _git(repo, "-c", "init.defaultBranch=main", "init")
        _git(repo, "remote", "add", "origin", "git@github.com:acme/widgets.git")
        assert repo_id(repo) == _sha16("github.com/acme/widgets")

    def test_no_repo_falls_back_to_realpath(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert repo_id(plain) == _sha16(os.path.normcase(str(plain.resolve())))

    @requires_git
    def test_stable_across_calls(self, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo)
        assert repo_id(repo) == repo_id(repo)
