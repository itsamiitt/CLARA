"""
CLARA — Embedding Engine

Provides a unified interface for generating dense vector embeddings from text.
Supports two backends, selected via the ``CLARA_EMBEDDING_BACKEND`` env var:

* **openai**  — OpenAI ``text-embedding-3-small`` (1536 dimensions).
                Requires the ``openai`` package and a valid ``OPENAI_API_KEY``.
* **local**   — ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions).
                Runs entirely on-device via the ``sentence-transformers`` package.

The engine follows a **singleton pattern**: only one backend client is ever
instantiated per process, regardless of how many times :func:`get_engine` is
called.  Use :func:`reset_engine` in tests to tear down the singleton.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

# Optional dependency imports — guarded so the module can load even if
# only one backend's dependencies are installed.
try:
    import openai as openai  # type: ignore[import-untyped]
except ImportError:
    openai: Any = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
except ImportError:
    SentenceTransformer: Any = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_BACKEND = "CLARA_EMBEDDING_BACKEND"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "CLARA_OPENAI_EMBEDDING_MODEL"

DEFAULT_BACKEND = "openai"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIMENSIONS = 1536

LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIMENSIONS = 384

OPENAI_BATCH_LIMIT = 2048  # Max texts per single API call


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalize_embedding_dimensions(
    vector: list[float],
    *,
    target_dimensions: int,
) -> list[float]:
    """Pad or truncate a vector so it fits the storage schema."""
    if len(vector) == target_dimensions:
        return vector
    if len(vector) > target_dimensions:
        return vector[:target_dimensions]
    return vector + ([0.0] * (target_dimensions - len(vector)))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _EmbeddingBackend(ABC):
    """Internal interface that each concrete backend implements."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class _OpenAIBackend(_EmbeddingBackend):
    """Wraps the OpenAI embeddings API (``text-embedding-3-small``)."""

    def __init__(self) -> None:
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for the OpenAI embedding "
                "backend.  Install it with:  pip install openai"
            )

        api_key = os.environ.get(ENV_OPENAI_KEY)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {ENV_OPENAI_KEY!r} is not set. "
                "It is required for the OpenAI embedding backend."
            )

        self._client = openai.OpenAI(api_key=api_key)
        self._model = os.environ.get(ENV_OPENAI_MODEL, OPENAI_MODEL)

    @property
    def dimensions(self) -> int:
        return OPENAI_DIMENSIONS

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return list(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # OpenAI supports batched input, but cap at OPENAI_BATCH_LIMIT per call
        for start in range(0, len(texts), OPENAI_BATCH_LIMIT):
            chunk = texts[start : start + OPENAI_BATCH_LIMIT]
            response = self._client.embeddings.create(
                input=chunk,
                model=self._model,
            )
            # API returns results in input order
            all_embeddings.extend(
                list(item.embedding) for item in response.data
            )

        return all_embeddings


# ---------------------------------------------------------------------------
# Local (sentence-transformers) backend
# ---------------------------------------------------------------------------

class _LocalBackend(_EmbeddingBackend):
    """Wraps ``sentence-transformers/all-MiniLM-L6-v2`` for local inference."""

    def __init__(self) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "The 'sentence-transformers' package is required for the local "
                "embedding backend.  Install it with:  pip install sentence-transformers"
            )

        self._model = SentenceTransformer(LOCAL_MODEL)

    @property
    def dimensions(self) -> int:
        return LOCAL_DIMENSIONS

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, batch_size=64)
        return [v.tolist() for v in vectors]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: EmbeddingEngine | None = None


class EmbeddingEngine:
    """Public embedding interface — delegates to one of the concrete backends.

    Obtain the process-wide singleton via :func:`get_engine`.  Do **not**
    instantiate directly unless you need an isolated instance (e.g. tests).

    Usage::

        engine = get_engine()
        vec = engine.embed("User prefers Rust for systems work.")
        vecs = engine.embed_batch(["hello", "world"])
    """

    def __init__(self, backend: _EmbeddingBackend) -> None:
        self._backend = backend

    # -- Public API --------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Dimensionality of the vectors produced by the current backend."""
        return self._backend.dimensions

    @property
    def backend_name(self) -> str:
        """Human-readable identifier for the active backend."""
        if isinstance(self._backend, _OpenAIBackend):
            return "openai"
        if isinstance(self._backend, _LocalBackend):
            return "local"
        return type(self._backend).__name__

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats with length equal to :attr:`dimensions`.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call.

        This is more efficient than calling :meth:`embed` in a loop because
        backends can batch network requests or GPU inference.

        Args:
            texts: A list of input strings.

        Returns:
            A list of embedding vectors, one per input text, in the same
            order as the input.

        Raises:
            ValueError: If any text in the batch is empty or whitespace-only.
        """
        if not texts:
            return []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(
                    f"Cannot embed empty or whitespace-only text at index {i}."
                )
        return self._backend.embed_batch(texts)


# ---------------------------------------------------------------------------
# Factory / singleton helpers
# ---------------------------------------------------------------------------

def _create_backend(name: str) -> _EmbeddingBackend:
    """Instantiate the backend identified by *name*."""
    name_lower = name.strip().lower()
    if name_lower == "openai":
        return _OpenAIBackend()
    if name_lower == "local":
        return _LocalBackend()
    raise ValueError(
        f"Unknown embedding backend {name!r}. "
        f"Supported values for {ENV_BACKEND}: 'openai', 'local'."
    )


def get_engine() -> EmbeddingEngine:
    """Return the process-wide :class:`EmbeddingEngine` singleton.

    The backend is determined by the ``CLARA_EMBEDDING_BACKEND`` environment
    variable (default: ``"openai"``).  The singleton is created on first call
    and reused thereafter.
    """
    global _instance
    if _instance is not None:
        return _instance

    with _lock:
        # Double-checked locking
        if _instance is not None:
            return _instance

        backend_name = os.environ.get(ENV_BACKEND, DEFAULT_BACKEND)
        backend = _create_backend(backend_name)
        _instance = EmbeddingEngine(backend)
        return _instance


def reset_engine() -> None:
    """Tear down the singleton — useful in tests or backend hot-swaps."""
    global _instance
    with _lock:
        _instance = None
