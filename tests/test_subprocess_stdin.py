"""
Every subprocess call must close stdin and bound its runtime.

This is not style. The MCP server speaks JSON-RPC over its own stdin, and a
child process that inherits it can read from that pipe — consuming the client's
requests. The bytes are gone, no reply is possible, and the server sits idle
waiting for input that was already eaten.

Observed in production on a plugin build that predated the fix: eight
concurrent memory_save calls, some saved, others never answered at all. The
server process had one thread, 0.015 s of CPU after 45 minutes, and no
-wal/-shm files — it had never reached the database. Eight subprocess calls in
that build inherited stdin, and repoid.py and store.py shell out to
`git rev-parse` on essentially every tool call.

A timeout alone does not save it: git exits after ten seconds, but the protocol
bytes it swallowed are unrecoverable.

The fix was applied file by file with nothing to hold it in place, so this test
is the thing that keeps it applied.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CALL = re.compile(r"subprocess\.(run|Popen|check_output|call|check_call)\s*\(")


def _calls():
    """Every subprocess call site under clara/, with its full argument text."""
    for path in sorted((_ROOT / "clara").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in _CALL.finditer(source):
            depth = 0
            i = match.end() - 1
            while i < len(source):
                if source[i] == "(":
                    depth += 1
                elif source[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            yield (
                path.relative_to(_ROOT).as_posix(),
                source[: match.start()].count("\n") + 1,
                source[match.start() : i + 1],
            )


def test_there_are_subprocess_calls_to_check():
    # Guards the guard: a regex that silently matches nothing would make every
    # assertion below vacuously true.
    assert len(list(_calls())) >= 5


@pytest.mark.parametrize(
    ("path", "line", "call"), [pytest.param(*c, id=f"{c[0]}:{c[1]}") for c in _calls()]
)
class TestEveryCallIsSafe:
    def test_closes_stdin(self, path, line, call):
        assert "stdin=" in call, (
            f"{path}:{line} inherits stdin. Under the MCP server that is the "
            f"JSON-RPC pipe, and the child can eat the client's requests. "
            f"Pass stdin=subprocess.DEVNULL."
        )
        assert "DEVNULL" in call or "devnull" in call, (
            f"{path}:{line} sets stdin to something other than DEVNULL — "
            f"intentional? Nothing here should read from the parent's stdin."
        )

    def test_is_bounded(self, path, line, call):
        # A hung child blocks whatever awaited it; the session hook's
        # "never blocks a session" contract depends on this.
        assert "timeout=" in call, f"{path}:{line} has no timeout"
