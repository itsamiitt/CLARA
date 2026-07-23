"""
CLARA — Centralized Configuration

Collects all CLARA settings into a single ``ClaraConfig`` dataclass,
reading from environment variables with sensible defaults.

This mirrors the existing per-module env-var pattern (used in
``extractor.py`` and ``embeddings.py``) but provides a single import
for any module that needs multiple settings.

Usage::

    from clara.config import ClaraConfig

    # From environment variables
    config = ClaraConfig.from_env()

    # Explicit overrides (e.g. tests)
    config = ClaraConfig(db_url="sqlite+aiosqlite:///clara.db", start_scheduler=False)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Default values (keep in sync with per-module defaults)
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite+aiosqlite:///clara.db"
_DEFAULT_EMBEDDING_BACKEND = "openai"
_DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_LLM_PROVIDER = "openai"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_RETRIEVAL_TOP_K = 8
_DEFAULT_SIMILARITY_THRESHOLD = 0.82
_DEFAULT_ARCHIVAL_THRESHOLD = 0.15
_DEFAULT_EVENT_STALE_DAYS = 90
_DEFAULT_SKILL_UNUSED_DAYS = 60
_DEFAULT_CACHE_URL: str | None = None
_DEFAULT_AUTH_REQUIRED = False
_DEFAULT_GRAPH_ENABLED = True
_DEFAULT_DOCS_ENABLED = True


@dataclass(frozen=True, slots=True)
class ClaraConfig:
    """All CLARA settings in one place.

    Attributes:
        db_url:
            SQLAlchemy async connection URL. Defaults to a local SQLite file.
        embedding_backend:
            ``"openai"`` or ``"local"`` (sentence-transformers).
        openai_embedding_model:
            Model name for OpenAI embeddings.
        llm_provider:
            ``"openai"``, ``"anthropic"``, or ``"ollama"`` — used by the fact extractor
            and reasoning engine.
        openai_model:
            OpenAI model for extraction / reasoning.
        anthropic_model:
            Anthropic model for extraction / reasoning.
        ollama_base_url:
            Base URL for the local Ollama server.
        ollama_llm_model:
            Ollama model for extraction / reasoning / reflection.
        ollama_embed_model:
            Ollama model for embeddings.
        retrieval_top_k:
            Default number of results returned by retrieval queries.
        similarity_threshold:
            Cosine similarity threshold for the update engine's
            duplicate-detection step.
        archival_threshold:
            Confidence score below which a memory is auto-archived during
            the daily decay job.
        event_stale_days:
            Events older than this many days (with no linked beliefs) are
            pruned during the weekly job.
        skill_unused_days:
            Skills not accessed for this many days are deprecated during
            the weekly job.
        start_scheduler:
            Whether to start the ``DecayScheduler`` on agent creation.
            Set to ``False`` in test environments.
        cache_url:
            Optional retrieval cache URL. Use ``"memory://"`` for the built-in
            in-process cache or a Redis URL for shared cache storage.
        auth_required:
            When ``True``, API routes require an ``X-User-ID`` header and
            reject requests that try to act on a different ``user_id``.
    """

    # Database
    db_url: str = _DEFAULT_DB_URL

    # Embeddings
    embedding_backend: str = _DEFAULT_EMBEDDING_BACKEND
    openai_embedding_model: str = _DEFAULT_OPENAI_EMBEDDING_MODEL

    # LLM (extraction / reasoning)
    llm_provider: str = _DEFAULT_LLM_PROVIDER
    openai_model: str = _DEFAULT_OPENAI_MODEL
    anthropic_model: str = _DEFAULT_ANTHROPIC_MODEL
    ollama_base_url: str = _DEFAULT_OLLAMA_BASE_URL
    ollama_llm_model: str = _DEFAULT_OLLAMA_MODEL
    ollama_embed_model: str = _DEFAULT_OLLAMA_EMBED_MODEL

    # Retrieval
    retrieval_top_k: int = _DEFAULT_RETRIEVAL_TOP_K
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD

    # Decay / pruning
    archival_threshold: float = _DEFAULT_ARCHIVAL_THRESHOLD
    event_stale_days: int = _DEFAULT_EVENT_STALE_DAYS
    skill_unused_days: int = _DEFAULT_SKILL_UNUSED_DAYS

    # Scheduler
    start_scheduler: bool = True
    cache_url: str | None = _DEFAULT_CACHE_URL
    auth_required: bool = _DEFAULT_AUTH_REQUIRED

    # Plugin add-on kill switches (see also clara.flags for the per-call
    # helpers long-lived processes use)
    graph_enabled: bool = _DEFAULT_GRAPH_ENABLED
    docs_enabled: bool = _DEFAULT_DOCS_ENABLED

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ClaraConfig:
        """Build a :class:`ClaraConfig` from environment variables.

        Every field falls back to its default when the corresponding env var
        is absent or empty.  Environment variables used:

        * ``CLARA_DB_URL``
        * ``CLARA_EMBEDDING_BACKEND``
        * ``CLARA_OPENAI_EMBEDDING_MODEL``
        * ``CLARA_LLM_PROVIDER``
        * ``CLARA_OPENAI_MODEL``
        * ``CLARA_ANTHROPIC_MODEL``
        * ``CLARA_OLLAMA_BASE_URL``
        * ``CLARA_OLLAMA_MODEL``
        * ``CLARA_OLLAMA_EMBED_MODEL``
        * ``CLARA_RETRIEVAL_TOP_K``
        * ``CLARA_SIMILARITY_THRESHOLD``
        * ``CLARA_ARCHIVAL_THRESHOLD``
        * ``CLARA_EVENT_STALE_DAYS``
        * ``CLARA_SKILL_UNUSED_DAYS``
        * ``CLARA_START_SCHEDULER``
        * ``CLARA_CACHE_URL``
        * ``CLARA_AUTH_REQUIRED``
        """
        def _str(key: str, default: str) -> str:
            return os.environ.get(key, "").strip() or default

        def _int(key: str, default: int) -> int:
            raw = os.environ.get(key, "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _float(key: str, default: float) -> float:
            raw = os.environ.get(key, "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _bool(key: str, default: bool) -> bool:
            raw = os.environ.get(key, "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        return cls(
            db_url=_str("CLARA_DB_URL", _DEFAULT_DB_URL),
            embedding_backend=_str("CLARA_EMBEDDING_BACKEND", _DEFAULT_EMBEDDING_BACKEND),
            openai_embedding_model=_str("CLARA_OPENAI_EMBEDDING_MODEL", _DEFAULT_OPENAI_EMBEDDING_MODEL),
            llm_provider=_str("CLARA_LLM_PROVIDER", _DEFAULT_LLM_PROVIDER),
            openai_model=_str("CLARA_OPENAI_MODEL", _DEFAULT_OPENAI_MODEL),
            anthropic_model=_str("CLARA_ANTHROPIC_MODEL", _DEFAULT_ANTHROPIC_MODEL),
            ollama_base_url=_str("CLARA_OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL),
            ollama_llm_model=_str("CLARA_OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
            ollama_embed_model=_str("CLARA_OLLAMA_EMBED_MODEL", _DEFAULT_OLLAMA_EMBED_MODEL),
            retrieval_top_k=_int("CLARA_RETRIEVAL_TOP_K", _DEFAULT_RETRIEVAL_TOP_K),
            similarity_threshold=_float("CLARA_SIMILARITY_THRESHOLD", _DEFAULT_SIMILARITY_THRESHOLD),
            archival_threshold=_float("CLARA_ARCHIVAL_THRESHOLD", _DEFAULT_ARCHIVAL_THRESHOLD),
            event_stale_days=_int("CLARA_EVENT_STALE_DAYS", _DEFAULT_EVENT_STALE_DAYS),
            skill_unused_days=_int("CLARA_SKILL_UNUSED_DAYS", _DEFAULT_SKILL_UNUSED_DAYS),
            start_scheduler=_bool("CLARA_START_SCHEDULER", True),
            cache_url=_str("CLARA_CACHE_URL", "memory://") if os.environ.get("CLARA_CACHE_URL") else None,
            auth_required=_bool("CLARA_AUTH_REQUIRED", _DEFAULT_AUTH_REQUIRED),
            graph_enabled=_bool("CLARA_GRAPH_ENABLED", _DEFAULT_GRAPH_ENABLED),
            docs_enabled=_bool("CLARA_DOCS_ENABLED", _DEFAULT_DOCS_ENABLED),
        )
