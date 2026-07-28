"""
doctor must check the shim — it is what Claude Code actually spawns.

Found on a real install: the plugin's shim held clara-mcp.exe but not
clara.exe, so /clara:sync, /clara:docs and `clara sync` had nothing to shell
out to. doctor reported every check ok, because it looked at the store, the
schema and the venv but never at the two files the host and the slash commands
actually run.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def doctor(tmp_path, data_dir):
    env = {
        **os.environ,
        "CLARA_HOME": str(tmp_path / "home"),
        "CLAUDE_PLUGIN_DATA": str(data_dir),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }
    done = subprocess.run(
        [sys.executable, "-m", "clara.cli", "doctor"],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path,
    )
    assert done.returncode in (0, 1), done.stderr[-300:]
    return done.stdout


@pytest.fixture()
def installed(tmp_path):
    """A plugin data dir that looks installed: a current venv and a shim."""
    data = tmp_path / "data"
    (data / "shim").mkdir(parents=True)
    (data / "current").mkdir()
    return data


class TestShimCompleteness:
    def test_a_complete_shim_reports_ok(self, tmp_path, installed) -> None:
        for name in ("clara-mcp.exe", "clara.exe"):
            (installed / "shim" / name).write_bytes(b"stub")
        out = doctor(tmp_path, installed)
        assert "[ok] shim clara-mcp: present" in out
        assert "[ok] shim clara: present" in out
        assert "MISSING" not in out

    def test_missing_cli_shim_is_flagged(self, tmp_path, installed) -> None:
        # Exactly the real install's shape.
        (installed / "shim" / "clara-mcp.exe").write_bytes(b"stub")
        out = doctor(tmp_path, installed)
        assert "[warn] shim clara: MISSING" in out
        assert "what to do:" in out

    def test_missing_mcp_shim_is_flagged(self, tmp_path, installed) -> None:
        (installed / "shim" / "clara.exe").write_bytes(b"stub")
        out = doctor(tmp_path, installed)
        assert "[warn] shim clara-mcp: MISSING" in out
        # The consequence, not just the fact: memory tools will not load.
        assert "memory tools" in out

    def test_extensionless_shim_counts(self, tmp_path, installed) -> None:
        # POSIX installs have no .exe suffix; both layouts are valid.
        (installed / "shim" / "clara-mcp").write_bytes(b"stub")
        (installed / "shim" / "clara").write_bytes(b"stub")
        out = doctor(tmp_path, installed)
        assert "MISSING" not in out

    def test_nothing_installed_yet_is_not_a_warning(self, tmp_path) -> None:
        # Before the first bootstrap there is no venv, so an absent shim is
        # expected rather than broken — warning there would cry wolf at every
        # brand-new user.
        data = tmp_path / "data"
        data.mkdir()
        out = doctor(tmp_path, data)
        assert "MISSING" not in out
        assert "not installed" in out


class TestFailedInstallIsSurfaced:
    def test_marker_is_reported_with_the_log(self, tmp_path, installed) -> None:
        (installed / "shim" / "clara-mcp.exe").write_bytes(b"stub")
        (installed / "shim" / "clara.exe").write_bytes(b"stub")
        (installed / "install.failed").write_text("Mon Jul 27 00:00:00 2026", "utf-8")
        out = doctor(tmp_path, installed)
        assert "last background install FAILED" in out
        assert "install.log" in out

    def test_no_marker_no_noise(self, tmp_path, installed) -> None:
        (installed / "shim" / "clara-mcp.exe").write_bytes(b"stub")
        (installed / "shim" / "clara.exe").write_bytes(b"stub")
        assert "install FAILED" not in doctor(tmp_path, installed)


class TestStaleShimIsSurfaced:
    """A shim refresh that failed against a locked exe must not stay silent.

    Found on a real install: the running clara-mcp.exe denied both overwrite
    and rename, bootstrap logged "shim refresh failed (non-fatal)" into a file
    nobody reads, and the host kept spawning week-old server code. The
    shim.stale marker is the visible trail; doctor must surface it.
    """

    def test_stale_marker_is_flagged(self, tmp_path, installed) -> None:
        (installed / "shim" / "clara-mcp.exe").write_bytes(b"stub")
        (installed / "shim" / "clara.exe").write_bytes(b"stub")
        (installed / "shim.stale").write_text("Mon Jul 28 09:00:00 2026", "utf-8")
        out = doctor(tmp_path, installed)
        assert "MCP shim refresh FAILED" in out
        assert "close Claude Code sessions" in out

    def test_no_stale_marker_no_noise(self, tmp_path, installed) -> None:
        (installed / "shim" / "clara-mcp.exe").write_bytes(b"stub")
        (installed / "shim" / "clara.exe").write_bytes(b"stub")
        assert "shim refresh FAILED" not in doctor(tmp_path, installed)


class TestStoreBreakdown:
    """doctor's store line says whose memories they are.

    "11 active" alone hid that 9 belonged to another repository — the same
    blindness every read surface had before locality. The split only prints
    when the sample covers the whole store; an exact-looking number from a
    partial sample would be a lie with extra steps.
    """

    def test_mixed_store_shows_the_split(self, tmp_path) -> None:
        import asyncio
        import subprocess as sp

        from clara.integrations.local_memory import LocalMemory

        for name in ("repoA", "repoB"):
            (tmp_path / name).mkdir()
            sp.run(["git", "init", "-q", str(tmp_path / name)], capture_output=True)
        home = tmp_path / "clarahome"
        home.mkdir()

        async def seed():
            import os as _os

            _os.chdir(tmp_path / "repoA")
            memory = await LocalMemory.create(str(home / "clara.db"))
            await memory.save(mem_type="belief", subject="payments",
                              relation="uses", object="stripe")
            await memory.close()

        cwd = os.getcwd()
        try:
            asyncio.run(seed())
        finally:
            os.chdir(cwd)

        env = {
            **os.environ,
            "CLARA_HOME": str(home),
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
        done = subprocess.run(
            [sys.executable, "-m", "clara.cli", "doctor"],
            capture_output=True, text=True, encoding="utf-8",
            env=env, cwd=tmp_path / "repoB",
        )
        assert "1 active (0 this project, 1 from other projects)" in done.stdout

    def test_single_project_store_stays_plain(self, tmp_path) -> None:
        import asyncio
        import subprocess as sp

        from clara.integrations.local_memory import LocalMemory

        (tmp_path / "repoA").mkdir()
        sp.run(["git", "init", "-q", str(tmp_path / "repoA")], capture_output=True)
        home = tmp_path / "clarahome"
        home.mkdir()

        async def seed():
            import os as _os

            _os.chdir(tmp_path / "repoA")
            memory = await LocalMemory.create(str(home / "clara.db"))
            await memory.save(mem_type="belief", subject="payments",
                              relation="uses", object="stripe")
            await memory.close()

        cwd = os.getcwd()
        try:
            asyncio.run(seed())
        finally:
            os.chdir(cwd)

        env = {
            **os.environ,
            "CLARA_HOME": str(home),
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
        done = subprocess.run(
            [sys.executable, "-m", "clara.cli", "doctor"],
            capture_output=True, text=True, encoding="utf-8",
            env=env, cwd=tmp_path / "repoA",
        )
        assert "1 active\n" in done.stdout or "1 active\r\n" in done.stdout
