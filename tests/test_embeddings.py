"""
Tests for clara.retrieval.embeddings — EmbeddingEngine with mocked backends.

All tests use mocks for both OpenAI and sentence-transformers so that no real
API calls or model downloads happen during test runs.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clara.retrieval.embeddings import (
    DEFAULT_BACKEND,
    ENV_BACKEND,
    ENV_OPENAI_KEY,
    LOCAL_DIMENSIONS,
    OPENAI_BATCH_LIMIT,
    OPENAI_DIMENSIONS,
    EmbeddingEngine,
    _create_backend,
    _LocalBackend,
    _OpenAIBackend,
    get_engine,
    normalize_embedding_dimensions,
    reset_engine,
)


class TestNormalizeDimensions:
    def test_pads_short_vector(self):
        out = normalize_embedding_dimensions([1.0, 2.0], target_dimensions=5)
        assert out == [1.0, 2.0, 0.0, 0.0, 0.0]

    def test_truncates_oversize_vector(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="clara.retrieval.embeddings"):
            out = normalize_embedding_dimensions([0.1] * 2000, target_dimensions=1536)
        assert len(out) == 1536
        assert any("Truncating" in rec.message for rec in caplog.records)

    def test_exact_length_unchanged(self):
        vec = [0.5] * 4
        assert normalize_embedding_dimensions(vec, target_dimensions=4) is vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_openai_response(vectors: list[list[float]]):
    """Build a mock response that mimics openai.embeddings.create()."""
    data = [SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)]
    return SimpleNamespace(data=data)


def _random_vector(dims: int) -> list[float]:
    return [float(x) for x in np.random.default_rng(42).random(dims)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the singleton is torn down before and after every test."""
    reset_engine()
    yield
    reset_engine()


# ---------------------------------------------------------------------------
# _OpenAIBackend
# ---------------------------------------------------------------------------

class TestOpenAIBackend:
    """Tests for the OpenAI embedding backend (mocked)."""

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_embed_single(self, mock_openai_module):
        """embed() should call the OpenAI API and return a flat list."""
        expected = _random_vector(OPENAI_DIMENSIONS)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _fake_openai_response([expected])
        mock_openai_module.OpenAI.return_value = mock_client

        with patch("clara.retrieval.embeddings.openai", mock_openai_module):
            backend = _OpenAIBackend()

        result = backend.embed("hello world")

        assert result == expected
        assert len(result) == OPENAI_DIMENSIONS
        mock_client.embeddings.create.assert_called_once()

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_embed_batch(self, mock_openai_module):
        """embed_batch() should return one vector per input text."""
        v1 = _random_vector(OPENAI_DIMENSIONS)
        v2 = _random_vector(OPENAI_DIMENSIONS)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _fake_openai_response([v1, v2])
        mock_openai_module.OpenAI.return_value = mock_client

        with patch("clara.retrieval.embeddings.openai", mock_openai_module):
            backend = _OpenAIBackend()

        result = backend.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert result[0] == v1
        assert result[1] == v2

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_embed_batch_empty(self, mock_openai_module):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with patch("clara.retrieval.embeddings.openai", mock_openai_module):
            backend = _OpenAIBackend()

        assert backend.embed_batch([]) == []
        mock_client.embeddings.create.assert_not_called()

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_embed_batch_chunking(self, mock_openai_module):
        """Batches larger than OPENAI_BATCH_LIMIT should be split into chunks."""
        mock_client = MagicMock()
        # Each call returns one vector per input text
        small_vec = [0.1] * OPENAI_DIMENSIONS
        mock_client.embeddings.create.side_effect = lambda **kw: (
            _fake_openai_response([small_vec] * len(kw["input"]))
        )
        mock_openai_module.OpenAI.return_value = mock_client

        with patch("clara.retrieval.embeddings.openai", mock_openai_module):
            backend = _OpenAIBackend()

        n = OPENAI_BATCH_LIMIT + 5
        texts = [f"text_{i}" for i in range(n)]
        result = backend.embed_batch(texts)

        assert len(result) == n
        # Should have been called twice: one full chunk + one remainder
        assert mock_client.embeddings.create.call_count == 2

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        """Instantiation without OPENAI_API_KEY should raise."""
        with (
            patch("clara.retrieval.embeddings.openai", create=True),
            pytest.raises(EnvironmentError, match="OPENAI_API_KEY"),
        ):
            _OpenAIBackend()

    def test_missing_openai_package_raises(self):
        """When openai is None (not installed), instantiation should raise."""
        with (
            patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"}),
            patch("clara.retrieval.embeddings.openai", None),
            pytest.raises(ImportError, match="openai"),
        ):
            _OpenAIBackend()

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test-key"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_dimensions(self, mock_openai_module):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with patch("clara.retrieval.embeddings.openai", mock_openai_module):
            backend = _OpenAIBackend()

        assert backend.dimensions == OPENAI_DIMENSIONS


# ---------------------------------------------------------------------------
# _LocalBackend
# ---------------------------------------------------------------------------

class TestLocalBackend:
    """Tests for the local sentence-transformers backend (mocked)."""

    @patch("clara.retrieval.embeddings.SentenceTransformer", create=True)
    def test_embed_single(self, MockST):
        """embed() should encode a single string and return a flat list."""
        expected = np.array(_random_vector(LOCAL_DIMENSIONS), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = expected
        MockST.return_value = mock_model

        with patch(
            "clara.retrieval.embeddings.SentenceTransformer",
            MockST,
        ):
            backend = _LocalBackend()

        result = backend.embed("hello world")

        assert isinstance(result, list)
        assert len(result) == LOCAL_DIMENSIONS
        mock_model.encode.assert_called_once_with("hello world", convert_to_numpy=True)

    @patch("clara.retrieval.embeddings.SentenceTransformer", create=True)
    def test_embed_batch(self, MockST):
        """embed_batch() should encode multiple strings at once."""
        v1 = np.array(_random_vector(LOCAL_DIMENSIONS), dtype=np.float32)
        v2 = np.array(_random_vector(LOCAL_DIMENSIONS), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.stack([v1, v2])
        MockST.return_value = mock_model

        with patch(
            "clara.retrieval.embeddings.SentenceTransformer",
            MockST,
        ):
            backend = _LocalBackend()

        result = backend.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == LOCAL_DIMENSIONS
        mock_model.encode.assert_called_once_with(
            ["hello", "world"], convert_to_numpy=True, batch_size=64,
        )

    @patch("clara.retrieval.embeddings.SentenceTransformer", create=True)
    def test_embed_batch_empty(self, MockST):
        mock_model = MagicMock()
        MockST.return_value = mock_model

        with patch(
            "clara.retrieval.embeddings.SentenceTransformer",
            MockST,
        ):
            backend = _LocalBackend()

        assert backend.embed_batch([]) == []
        mock_model.encode.assert_not_called()

    @patch("clara.retrieval.embeddings.SentenceTransformer", create=True)
    def test_dimensions(self, MockST):
        MockST.return_value = MagicMock()

        with patch(
            "clara.retrieval.embeddings.SentenceTransformer",
            MockST,
        ):
            backend = _LocalBackend()

        assert backend.dimensions == LOCAL_DIMENSIONS

    def test_missing_package_raises(self):
        """When SentenceTransformer is None (not installed), instantiation should raise."""
        with (
            patch("clara.retrieval.embeddings.SentenceTransformer", None),
            pytest.raises(ImportError, match="sentence-transformers"),
        ):
            _LocalBackend()


# ---------------------------------------------------------------------------
# EmbeddingEngine (public interface)
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:
    """Tests for the public EmbeddingEngine wrapper."""

    def _make_engine(self, dims: int = 1536) -> EmbeddingEngine:
        backend = MagicMock()
        backend.dimensions = dims
        backend.embed.return_value = [0.0] * dims
        backend.embed_batch.return_value = [[0.0] * dims]
        return EmbeddingEngine(backend)

    def test_embed_delegates(self):
        engine = self._make_engine()
        result = engine.embed("test")
        assert len(result) == 1536
        engine._backend.embed.assert_called_once_with("test")

    def test_embed_batch_delegates(self):
        engine = self._make_engine()
        engine._backend.embed_batch.return_value = [[0.0] * 1536] * 3
        result = engine.embed_batch(["a", "b", "c"])
        assert len(result) == 3

    def test_embed_empty_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="empty"):
            engine.embed("")

    def test_embed_whitespace_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="whitespace"):
            engine.embed("   ")

    def test_embed_batch_empty_text_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="index 1"):
            engine.embed_batch(["valid", "", "also valid"])

    def test_embed_batch_empty_list(self):
        engine = self._make_engine()
        assert engine.embed_batch([]) == []

    def test_dimensions_property(self):
        engine = self._make_engine(dims=384)
        assert engine.dimensions == 384

    def test_backend_name_openai(self):
        backend = _OpenAIBackend.__new__(_OpenAIBackend)
        engine = EmbeddingEngine(backend)
        assert engine.backend_name == "openai"

    def test_backend_name_local(self):
        backend = _LocalBackend.__new__(_LocalBackend)
        engine = EmbeddingEngine(backend)
        assert engine.backend_name == "local"

    def test_backend_name_unknown(self):
        engine = self._make_engine()
        assert engine.backend_name == "MagicMock"


# ---------------------------------------------------------------------------
# Factory & singleton
# ---------------------------------------------------------------------------

class TestFactory:
    """Tests for _create_backend, get_engine, and the singleton pattern."""

    def test_create_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            _create_backend("nonexistent")

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_BACKEND: "openai"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_get_engine_openai(self, mock_openai_module):
        mock_openai_module.OpenAI.return_value = MagicMock()

        engine = get_engine()

        assert isinstance(engine, EmbeddingEngine)
        assert engine.backend_name == "openai"

    @patch.dict(os.environ, {ENV_BACKEND: "local"})
    @patch("clara.retrieval.embeddings.SentenceTransformer", create=True)
    def test_get_engine_local(self, MockST):
        MockST.return_value = MagicMock()

        with patch(
            "clara.retrieval.embeddings.SentenceTransformer",
            MockST,
        ):
            engine = get_engine()

        assert isinstance(engine, EmbeddingEngine)
        assert engine.backend_name == "local"

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_BACKEND: "openai"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_singleton_same_instance(self, mock_openai_module):
        """get_engine() must return the same object on repeated calls."""
        mock_openai_module.OpenAI.return_value = MagicMock()

        e1 = get_engine()
        e2 = get_engine()

        assert e1 is e2

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_BACKEND: "openai"})
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_reset_clears_singleton(self, mock_openai_module):
        """After reset_engine(), get_engine() must create a fresh instance."""
        mock_openai_module.OpenAI.return_value = MagicMock()

        e1 = get_engine()
        reset_engine()
        e2 = get_engine()

        assert e1 is not e2

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test"}, clear=False)
    @patch("clara.retrieval.embeddings.openai", create=True)
    def test_default_backend_is_openai(self, mock_openai_module):
        """Without CLARA_EMBEDDING_BACKEND set, default should be 'openai'."""
        mock_openai_module.OpenAI.return_value = MagicMock()

        # Make sure the env var is NOT set
        with patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test"}, clear=True):
            engine = get_engine()

        assert engine.backend_name == "openai"
        assert DEFAULT_BACKEND == "openai"
