"""
A failed background install must say so, and stop saying so once it works.

Before this, every session after a failed install printed the same line —
"CLARA is installing in the background, memory will be available next
session." Verified against the real hook: three consecutive failures, three
identical optimistic messages, no mention of the log. Someone behind a proxy
that blocks PyPI would wait indefinitely for something that was never going to
arrive, with nothing to act on.

Retrying is the right behaviour and is unchanged. Only the reporting changed.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from tests.test_review_regressions import _ROOT


def _bash() -> str | None:
    for candidate in ("bash", "/usr/bin/bash", "C:/Program Files/Git/bin/bash.exe"):
        found = shutil.which(candidate) or (
            candidate if os.path.exists(candidate) else None
        )
        if found:
            return found
    return None


@pytest.fixture()
def plugin(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return data


def run_hook(data, tmp_path):
    bash = _bash()
    if bash is None:
        pytest.skip("no working bash")
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_DATA": str(data),
        "CLAUDE_PLUGIN_ROOT": str(_ROOT),
        "CLARA_HOME": str(tmp_path / "home"),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        # Nothing resolvable, so the install attempt cannot succeed and the
        # hook takes the "install running" branch quickly.
        "PIP_INDEX_URL": "http://127.0.0.1:9/nope",
        "UV_INDEX_URL": "http://127.0.0.1:9/nope",
        "PIP_RETRIES": "1",
        "PIP_TIMEOUT": "5",
    }
    return subprocess.run(
        [bash, str(_ROOT / "scripts" / "session-start.sh")],
        capture_output=True, text=True, encoding="utf-8", timeout=180,
        env=env, input='{"source":"startup"}', cwd=tmp_path,
    )


class TestFailureIsReported:
    def test_no_failure_on_record_reads_optimistically(self, plugin, tmp_path):
        done = run_hook(plugin, tmp_path)
        assert done.returncode == 0
        if "background" not in done.stdout:
            pytest.skip("hook did not take the installing branch here")
        # Nothing has failed yet, so there is nothing to warn about.
        assert "last attempt failed" not in done.stdout

    def test_a_recorded_failure_is_named(self, plugin, tmp_path):
        (plugin / "install.failed").write_text("Mon Jul 27 00:00:00 2026", "utf-8")
        done = run_hook(plugin, tmp_path)
        assert done.returncode == 0, done.stderr[-300:]
        if "background" not in done.stdout and "retrying" not in done.stdout:
            pytest.skip("hook did not take the installing branch here")
        assert "last attempt failed" in done.stdout
        # An error the reader cannot act on is barely better than silence.
        assert "install.log" in done.stdout
        assert "PyPI" in done.stdout

    def test_the_session_still_exits_zero(self, plugin, tmp_path):
        # The contract that matters most: memory never blocks a session, and a
        # failing install is exactly when it would be tempting to break it.
        (plugin / "install.failed").write_text("whenever", encoding="utf-8")
        assert run_hook(plugin, tmp_path).returncode == 0


class TestBothImplementationsAgree:
    """The .sh and .ps1 hooks are one behaviour with two implementations."""

    def test_bootstrap_writes_and_clears_the_marker(self):
        for script in ("bootstrap.sh", "win/bootstrap.ps1"):
            body = (_ROOT / "scripts" / script).read_text(encoding="utf-8")
            assert "install.failed" in body, f"{script} never records a failure"
            # Written on failure and removed on success -- a marker that is
            # never cleared warns forever after one bad day.
            assert body.count("install.failed") >= 2, (
                f"{script} must both write and clear the marker"
            )

    def test_session_start_reports_it(self):
        for script in ("session-start.sh", "win/session-start.ps1"):
            body = (_ROOT / "scripts" / script).read_text(encoding="utf-8")
            assert "install.failed" in body
            assert "last attempt failed" in body
            assert "install.log" in body
