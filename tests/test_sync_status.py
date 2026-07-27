"""
`clara sync status` must say which files import will actually read.

It listed only the CLAUDE.md variants, but MEMORY.md is an import source too.
Someone whose hand-written note did not import could not tell from the output
whether that file had even been looked at — which is exactly the position I was
in when a 284-character path made every stat report the file as absent while
`ls` showed it sitting there.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from clara.cli import _readability


class TestReadability:
    def test_existing_file_will_be_read(self, tmp_path) -> None:
        target = tmp_path / "MEMORY.md"
        target.write_text("- a note", encoding="utf-8")
        assert _readability(target) == "(will be read)"

    def test_missing_file_is_not_created_yet(self, tmp_path) -> None:
        # Distinct from unreadable: nothing is wrong, there is just no file.
        assert _readability(tmp_path / "MEMORY.md") == "(not created yet)"

    @pytest.mark.skipif(os.name != "nt", reason="MAX_PATH is a Windows limit")
    def test_over_long_windows_path_is_named(self, tmp_path) -> None:
        # Windows reports a path past its 260-character limit as simply
        # absent. Saying "not created yet" there sends someone looking for a
        # file that exists.
        deep = tmp_path
        while len(str(deep)) < 300:
            deep = deep / "a-reasonably-long-directory-name"
        verdict = _readability(deep / "MEMORY.md")
        assert "260" in verdict
        assert "characters" in verdict


def _env(tmp_path):
    """Subprocess env with auto-memory on, pointed entirely at tmp_path.

    conftest disables auto-memory for every test so nothing can touch the
    developer's real ~/.claude files. Status has nothing to report with it
    off, so it is re-enabled here — safe because CLARA_HOME and
    CLAUDE_CONFIG_DIR both resolve under tmp_path, which is the same thing the
    bridge tests do inside their own fake home.
    """
    env = {
        **os.environ,
        "CLARA_HOME": str(tmp_path / "home"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "cfg"),
    }
    env.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)
    return env


class TestStatusOutput:
    def test_memory_md_is_listed_as_an_import_source(self, tmp_path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (tmp_path / "cfg").mkdir()
        env = _env(tmp_path)
        done = subprocess.run(
            [sys.executable, "-m", "clara.cli", "sync", "status"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=project, env=env,
        )
        assert done.returncode == 0, done.stderr
        assert "import sources:" in done.stdout
        # The MEMORY.md target must appear in the list, with a verdict, not
        # only in the "target" line above it.
        assert "MEMORY.md (" in done.stdout

    def test_status_does_not_need_an_existing_store(self, tmp_path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        env = _env(tmp_path)
        done = subprocess.run(
            [sys.executable, "-m", "clara.cli", "sync", "status"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=project, env=env,
        )
        assert done.returncode == 0
        assert "not created yet" in done.stdout
