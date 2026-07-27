"""CLARA must write native files where Claude Code actually reads them.

Claude Code lets users relocate its config directory with CLAUDE_CONFIG_DIR;
its changelog records "Respect CLAUDE_CONFIG_DIR everywhere" and three later
fixes for places that did not. CLARA hardcoded ~/.claude in five places, so
with the variable set:

  * `clara sync` reported "MEMORY.md updated" while writing into a directory
    Claude Code never opens — a silent no-op for that user;
  * autoMemoryDirectory / autoMemoryEnabled were read from the wrong
    settings.json, so both disable switches were ignored;
  * `clara statusline --install` wrote a statusLine that could never appear.

Found by running the full user journey with the variable set and then looking
for the files it claimed to have written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clara.bridge import paths


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    target.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(target))
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", raising=False)
    return target


class TestResolver:
    def test_uses_the_override(self, config_dir):
        assert paths.claude_config_dir() == config_dir

    def test_falls_back_to_home(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        assert paths.claude_config_dir() == Path.home() / ".claude"

    def test_blank_is_ignored(self, monkeypatch):
        """An empty value means unset, not "use the current directory"."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "   ")
        assert paths.claude_config_dir() == Path.home() / ".claude"

    def test_user_home_is_expanded(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/somewhere-else")
        resolved = paths.claude_config_dir()
        assert "~" not in str(resolved)
        assert resolved == Path.home() / "somewhere-else"


class TestNativeFilesFollowIt:
    def test_auto_memory_dir_is_redirected(self, config_dir, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        directory = paths.auto_memory_dir(str(project))
        assert directory is not None
        assert str(directory).startswith(str(config_dir)), (
            f"auto-memory went to {directory}, outside {config_dir}"
        )
        assert Path.home() / ".claude" not in directory.parents

    def test_memory_md_and_topic_file_follow(self, config_dir, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        for resolved in (paths.memory_md_path(str(project)),
                         paths.topic_file_path(str(project))):
            assert resolved is not None
            assert str(resolved).startswith(str(config_dir))

    def test_user_level_claude_md_is_read_from_there(self, config_dir, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (config_dir / "CLAUDE.md").write_text("- we use Postgres\n", encoding="utf-8")
        found = paths.claude_md_paths(str(project))
        assert any(str(p).startswith(str(config_dir)) for p in found), found

    def test_settings_are_read_from_there(self, config_dir, tmp_path):
        """The disable switch lives in settings.json; reading the wrong file
        means honouring a default the user never chose."""
        (config_dir / "settings.json").write_text(
            json.dumps({"autoMemoryEnabled": False}), encoding="utf-8"
        )
        project = tmp_path / "proj"
        project.mkdir()
        assert paths.auto_memory_dir(str(project)) is None, (
            "autoMemoryEnabled=false in the real config dir was ignored"
        )


class TestStatuslineFollowsIt:
    def test_settings_target_is_redirected(self, config_dir):
        from clara.statusline_setup import settings_path

        assert str(settings_path()).startswith(str(config_dir)), (
            "installing a statusLine into ~/.claude while Claude Code reads "
            "elsewhere gives the user a status bar that never appears"
        )


class TestWorktreesShareAutoMemory:
    """Claude Code keys auto memory on the repository, not the checkout.

    Its changelog: "Project configs & auto memory now shared across git
    worktrees of the same repository". CLARA used `git rev-parse
    --show-toplevel`, which returns the *linked* worktree's own path, so every
    worktree got its own auto-memory directory. Measured before the fix: main
    resolved to ...-clara-wt-main and the worktree to ...-clara-wt-wt, so a
    sync run inside a worktree wrote where Claude Code never reads — the same
    silent no-op as the hardcoded config dir.

    Note this deliberately differs from CLARA's *store* resolution, where each
    worktree may carry its own .clara/clara.db. That is CLARA's own data; this
    is Claude Code's, and has to match Claude Code.
    """

    @staticmethod
    def _repo_with_worktree(tmp_path):
        import subprocess

        main = tmp_path / "main"
        main.mkdir()
        run = lambda *a: subprocess.run(  # noqa: E731
            a, cwd=str(main), check=True, capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        subprocess.run(["git", "init", "-q", str(main)], check=True,
                       capture_output=True, stdin=subprocess.DEVNULL)
        (main / "f.txt").write_text("hi", encoding="utf-8")
        run("git", "add", "-A")
        run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
        worktree = tmp_path / "wt"
        run("git", "worktree", "add", "-q", str(worktree))
        return main, worktree

    def test_both_checkouts_resolve_to_the_main_root(self, tmp_path):
        main, worktree = self._repo_with_worktree(tmp_path)
        assert paths.project_root(str(worktree)) == paths.project_root(str(main))
        assert Path(paths.project_root(str(worktree))).name == "main"

    def test_auto_memory_directory_is_shared(self, config_dir, tmp_path):
        main, worktree = self._repo_with_worktree(tmp_path)
        from_main = paths.auto_memory_dir(str(main))
        from_worktree = paths.auto_memory_dir(str(worktree))
        assert from_main is not None and from_worktree is not None
        assert from_main == from_worktree, (
            "a worktree writes MEMORY.md where Claude Code never reads it"
        )

    def test_memory_md_and_topic_file_are_shared(self, config_dir, tmp_path):
        main, worktree = self._repo_with_worktree(tmp_path)
        assert paths.memory_md_path(str(main)) == paths.memory_md_path(str(worktree))
        assert paths.topic_file_path(str(main)) == paths.topic_file_path(str(worktree))

    def test_a_plain_directory_still_resolves(self, tmp_path):
        """No repo at all must keep working — the anchor is its own root."""
        plain = tmp_path / "plain"
        plain.mkdir()
        assert paths.project_root(str(plain)) == str(plain)

    def test_an_ordinary_repo_is_unchanged(self, tmp_path):
        import subprocess

        repo = tmp_path / "solo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True,
                       capture_output=True, stdin=subprocess.DEVNULL)
        resolved = paths.project_root(str(repo))
        assert Path(resolved).resolve() == repo.resolve()
