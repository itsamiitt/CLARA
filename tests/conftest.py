from __future__ import annotations

import os
from pathlib import Path

import pytest

from clara.retrieval.engine import LanceRetrievalEngine

# Snapshot the real store location once, before any test can redirect it.
_REAL_CLARA_HOME = Path(
    os.environ.get("CLARA_HOME") or (Path.home() / ".clara")
).expanduser()


@pytest.fixture(autouse=True)
def lance_fixture(tmp_path, monkeypatch):
    """Isolate the embedded LanceDB table for every test case."""
    lance_path = tmp_path / "clara_vectors"
    monkeypatch.setenv("CLARA_LANCE_PATH", str(lance_path))
    LanceRetrievalEngine.reset_defaults()
    LanceRetrievalEngine.configure_default_path(str(lance_path))
    yield str(lance_path)
    LanceRetrievalEngine.reset_defaults()


@pytest.fixture(autouse=True)
def _isolate_native_memory(monkeypatch):
    """Never let a test write the developer's real ~/.claude memory files.

    The MCP maintenance path exports to the auto-memory directory in a
    background task; a test that reaches it without patching Path.home()
    would touch real files. Bridge tests re-enable this deliberately inside
    their own fake home (monkeypatch.delenv in their fixture)."""
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")


@pytest.fixture(autouse=True)
def _protect_real_store():
    """Fail any test that writes the developer's real ~/.clara store.

    Setting CLARA_HOME is not enough on its own: code (and tests) may consult
    ``Path.home()`` directly, which no environment variable redirects. A test
    that did exactly that truncated the real clara.db to zero bytes, and since
    ``clara backup`` had never run there was nothing to restore from.

    This compares the store's identity before and after each test and fails
    loudly rather than letting the damage pass silently. It never creates the
    file, so a machine without a store stays without one.
    """
    store = _REAL_CLARA_HOME / "clara.db"

    def snapshot() -> tuple[bool, int, float] | None:
        try:
            stat = store.stat()
        except OSError:
            return None
        return (True, stat.st_size, stat.st_mtime)

    before = snapshot()
    yield
    after = snapshot()
    if before != after:
        raise AssertionError(
            f"test modified the real CLARA store at {store} "
            f"({before} -> {after}). Point CLARA_HOME *and* Path.home() at a "
            "tmp_path instead."
        )
