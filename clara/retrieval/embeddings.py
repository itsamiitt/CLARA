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

import logging
import os
import threading
from abc import ABC, abstractmethod

from clara.db.models import VECTOR_DIMENSIONS

logger = logging.getLogger(__name__)

# Track oversize lengths we've already warned about so truncation is logged
# once per distinct source dimension rather than on every embed call.
_TRUNCATION_WARNED: set[int] = set()

# Optional dependency imports — guarded so the module can load even if
# only one backend's dependencies are installed.
try:
    import openai as openai  # type: ignore[import-untyped]
except ImportError:
    openai = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_BACKEND = "CLARA_EMBEDDING_BACKEND"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "CLARA_OPENAI_EMBEDDING_MODEL"
ENV_OLLAMA_BASE_URL = "CLARA_OLLAMA_BASE_URL"
ENV_OLLAMA_EMBED_MODEL = "CLARA_OLLAMA_EMBED_MODEL"

DEFAULT_BACKEND = "openai"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIMENSIONS = 1536

LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIMENSIONS = 384

OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

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
        source_len = len(vector)
        if source_len not in _TRUNCATION_WARNED:
            _TRUNCATION_WARNED.add(source_len)
            logger.warning(
                "Truncating %d-dim embedding to %d dims; the tail is discarded. "
                "This is lossy — prefer a model whose dimension is <= %d.",
                source_len, target_dimensions, target_dimensions,
            )
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
            raise OSError(
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
        values: list[float] = vector.tolist()  # type: ignore[union-attr]
        return values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, batch_size=64)
        return [v.tolist() for v in vectors]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class _OllamaBackend(_EmbeddingBackend):
    """Wraps Ollama's local embedding API and normalizes vectors to storage size."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if _ollama_lib is None:
            raise ImportError(
                "The 'ollama' package is required for the Ollama embedding backend. "
                "Install it with: pip install 'clara-memory[ollama]'"
            )

        self._model = model or os.environ.get(ENV_OLLAMA_EMBED_MODEL, OLLAMA_MODEL)
        self._client = _ollama_lib.Client(
            host=base_url or os.environ.get(ENV_OLLAMA_BASE_URL, OLLAMA_BASE_URL)
        )
        self._ensure_model()

    def _ensure_model(self) -> None:
        listing = self._client.list()
        models = listing.get("models", []) if isinstance(listing, dict) else getattr(
            listing, "models", []
        )
        names = {
            str(item.get("name") or item.get("model") or "")
            for item in models
            if isinstance(item, dict)
        }
        if self._model not in names:
            self._client.pull(self._model)

    @property
    def dimensions(self) -> int:
        return VECTOR_DIMENSIONS

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings(model=self._model, prompt=text)
        raw = response.get("embedding", []) if isinstance(response, dict) else []
        return normalize_embedding_dimensions(
            [float(value) for value in raw],
            target_dimensions=VECTOR_DIMENSIONS,
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self.embed(text) for text in texts]


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
        if isinstance(self._backend, _OllamaBackend):
            return "ollama"
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

def _create_backend(
    name: str,
    *,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> _EmbeddingBackend:
    """Instantiate the backend identified by *name*."""
    name_lower = name.strip().lower()
    if name_lower == "openai":
        return _OpenAIBackend()
    if name_lower == "local":
        return _LocalBackend()
    if name_lower == "ollama":
        return _OllamaBackend(model=ollama_model, base_url=ollama_base_url)
    raise ValueError(
        f"Unknown embedding backend {name!r}. "
        f"Supported values for {ENV_BACKEND}: 'openai', 'local', 'ollama'."
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
