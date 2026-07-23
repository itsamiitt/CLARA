from __future__ import annotations

import pytest

from clara.retrieval.engine import LanceRetrievalEngine


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
