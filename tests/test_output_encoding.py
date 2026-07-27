"""
Captured output must be valid UTF-8.

CLARA renders a few non-ASCII glyphs (em dash, arrows). `errors="replace"`
already stopped an ASCII locale from crashing a command, but it does not choose
an *encoding*: a pipe gets the locale's, so on Windows the SessionStart hook
payload went out as cp1252 and the em dash became byte 0x97. That payload is
not valid UTF-8, so a host decoding it got a decode error or a replacement
character inside the injected memory block — on the main path, on the platform
most non-technical users are on.

These run the real entry points as subprocesses with captured stdout, because
that is the only arrangement in which the bug exists: in-process, pytest's
capture object is not a pipe with a locale encoding.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

EM_DASH = "—"


@pytest.fixture()
def store(tmp_path):
    """A store holding one belief, so the output has something to render."""
    env = {**os.environ, "CLARA_HOME": str(tmp_path / "store")}
    subprocess.run(
        [sys.executable, "-m", "clara.cli", "remember",
         "we use postgres for the main database"],
        capture_output=True, env=env, check=False,
    )
    return env


def run(env, *args, stdin=b""):
    return subprocess.run(
        [sys.executable, "-m", *args], capture_output=True, env=env, input=stdin
    )


class TestCapturedOutputIsUtf8:
    def test_session_start_hook_payload(self, store) -> None:
        # The exact path Claude Code captures on every session start.
        done = run(store, "clara.fastpath.context", stdin=b'{"source":"startup"}')
        assert done.returncode == 0
        done.stdout.decode("utf-8")  # must not raise

    def test_clara_context(self, store) -> None:
        done = run(store, "clara.cli", "context")
        assert done.returncode == 0
        text = done.stdout.decode("utf-8")
        # Not a vacuous check: this output really does carry the glyph that
        # was being mis-encoded.
        assert EM_DASH in text

    def test_clara_doctor(self, store) -> None:
        done = run(store, "clara.cli", "doctor")
        done.stdout.decode("utf-8")

    def test_no_cp1252_em_dash_byte_anywhere(self, store) -> None:
        # 0x97 is cp1252's em dash and is never a valid standalone UTF-8 byte.
        for args, stdin in (
            (("clara.fastpath.context",), b'{"source":"startup"}'),
            (("clara.cli", "context"), b""),
            (("clara.cli", "doctor"), b""),
        ):
            done = run(store, *args, stdin=stdin)
            assert b"\x97" not in done.stdout, f"{args} emitted cp1252 bytes"


class TestGuaranteesThatMustSurvive:
    """The fix must not cost what the previous behaviour bought."""

    def test_ascii_locale_still_does_not_crash(self, store) -> None:
        # The original reason errors="replace" exists: an ASCII stdout must
        # degrade a glyph, never fail a command that already did its work.
        env = {**store, "PYTHONIOENCODING": "ascii"}
        done = run(env, "clara.cli", "context")
        assert done.returncode == 0

    def test_explicit_pythonioencoding_is_respected(self, store) -> None:
        # An explicit setting is a deliberate instruction; only the encoding
        # Python inferred from the locale is overridden.
        env = {**store, "PYTHONIOENCODING": "ascii"}
        done = run(env, "clara.cli", "context")
        assert done.returncode == 0
        assert EM_DASH.encode("utf-8") not in done.stdout

    def test_hook_still_exits_zero_under_ascii(self, store) -> None:
        env = {**store, "PYTHONIOENCODING": "ascii"}
        done = run(env, "clara.fastpath.context", stdin=b'{"source":"startup"}')
        assert done.returncode == 0
