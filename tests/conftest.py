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
