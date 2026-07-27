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
