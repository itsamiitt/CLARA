"""
Tests for clara.config — ClaraConfig dataclass and from_env() factory.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from clara.config import ClaraConfig


class TestClaraConfigDefaults:
    """Verify that default values match the documented defaults."""

    def test_default_db_url(self):
        cfg = ClaraConfig()
        assert cfg.db_url == "sqlite+aiosqlite://"

    def test_default_embedding_backend(self):
        cfg = ClaraConfig()
        assert cfg.embedding_backend == "openai"

    def test_default_llm_provider(self):
        cfg = ClaraConfig()
        assert cfg.llm_provider == "openai"

    def test_default_openai_model(self):
        cfg = ClaraConfig()
        assert cfg.openai_model == "gpt-4o-mini"

    def test_default_anthropic_model(self):
        cfg = ClaraConfig()
        assert cfg.anthropic_model == "claude-3-5-haiku-20241022"

    def test_default_retrieval_top_k(self):
        cfg = ClaraConfig()
        assert cfg.retrieval_top_k == 8

    def test_default_similarity_threshold(self):
        cfg = ClaraConfig()
        assert cfg.similarity_threshold == 0.82

    def test_default_archival_threshold(self):
        cfg = ClaraConfig()
        assert cfg.archival_threshold == 0.15

    def test_default_event_stale_days(self):
        cfg = ClaraConfig()
        assert cfg.event_stale_days == 90

    def test_default_skill_unused_days(self):
        cfg = ClaraConfig()
        assert cfg.skill_unused_days == 60

    def test_default_start_scheduler(self):
        cfg = ClaraConfig()
        assert cfg.start_scheduler is True

    def test_default_cache_url(self):
        cfg = ClaraConfig()
        assert cfg.cache_url is None


class TestClaraConfigExplicitOverride:
    """Verify that explicit keyword arguments override defaults."""

    def test_override_db_url(self):
        cfg = ClaraConfig(db_url="postgresql+asyncpg://user:pass@host/db")
        assert cfg.db_url == "postgresql+asyncpg://user:pass@host/db"

    def test_override_multiple_fields(self):
        cfg = ClaraConfig(
            llm_provider="anthropic",
            retrieval_top_k=16,
            start_scheduler=False,
        )
        assert cfg.llm_provider == "anthropic"
        assert cfg.retrieval_top_k == 16
        assert cfg.start_scheduler is False
        # Non-overridden fields keep defaults
        assert cfg.db_url == "sqlite+aiosqlite://"
        assert cfg.embedding_backend == "openai"


class TestClaraConfigFrozen:
    """ClaraConfig is frozen — mutation should raise."""

    def test_cannot_mutate(self):
        cfg = ClaraConfig()
        with pytest.raises(AttributeError):
            cfg.db_url = "new_value"  # type: ignore[misc]


class TestClaraConfigFromEnv:
    """Verify from_env() reads environment variables correctly."""

    def test_from_env_uses_defaults_when_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.db_url == "sqlite+aiosqlite://"
        assert cfg.embedding_backend == "openai"
        assert cfg.llm_provider == "openai"
        assert cfg.start_scheduler is True

    def test_from_env_reads_db_url(self):
        env = {"CLARA_DB_URL": "postgresql+asyncpg://testhost/testdb"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.db_url == "postgresql+asyncpg://testhost/testdb"

    def test_from_env_reads_llm_provider(self):
        env = {"CLARA_LLM_PROVIDER": "anthropic"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.llm_provider == "anthropic"

    def test_from_env_reads_embedding_backend(self):
        env = {"CLARA_EMBEDDING_BACKEND": "local"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.embedding_backend == "local"

    def test_from_env_reads_cache_url(self):
        env = {"CLARA_CACHE_URL": "memory://"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.cache_url == "memory://"

    def test_from_env_reads_int_value(self):
        env = {"CLARA_RETRIEVAL_TOP_K": "16"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.retrieval_top_k == 16

    def test_from_env_reads_float_value(self):
        env = {"CLARA_SIMILARITY_THRESHOLD": "0.90"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.similarity_threshold == 0.90

    def test_from_env_scheduler_true_values(self):
        for val in ("true", "1", "yes", "on", "True", "TRUE"):
            env = {"CLARA_START_SCHEDULER": val}
            with patch.dict(os.environ, env, clear=True):
                cfg = ClaraConfig.from_env()
            assert cfg.start_scheduler is True, f"Failed for value {val!r}"

    def test_from_env_scheduler_false_values(self):
        for val in ("false", "0", "no", "off"):
            env = {"CLARA_START_SCHEDULER": val}
            with patch.dict(os.environ, env, clear=True):
                cfg = ClaraConfig.from_env()
            assert cfg.start_scheduler is False, f"Failed for value {val!r}"

    def test_from_env_invalid_int_falls_back(self):
        env = {"CLARA_RETRIEVAL_TOP_K": "not_a_number"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.retrieval_top_k == 8  # default

    def test_from_env_invalid_float_falls_back(self):
        env = {"CLARA_SIMILARITY_THRESHOLD": "bad"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.similarity_threshold == 0.82  # default

    def test_from_env_whitespace_stripped(self):
        env = {"CLARA_LLM_PROVIDER": "  anthropic  "}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.llm_provider == "anthropic"

    def test_from_env_empty_string_uses_default(self):
        env = {"CLARA_DB_URL": ""}
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()
        assert cfg.db_url == "sqlite+aiosqlite://"

    def test_from_env_all_fields_together(self):
        env = {
            "CLARA_DB_URL": "postgresql+asyncpg://prod/clara",
            "CLARA_EMBEDDING_BACKEND": "local",
            "CLARA_LLM_PROVIDER": "anthropic",
            "CLARA_OPENAI_MODEL": "gpt-4o",
            "CLARA_RETRIEVAL_TOP_K": "12",
            "CLARA_SIMILARITY_THRESHOLD": "0.75",
            "CLARA_ARCHIVAL_THRESHOLD": "0.20",
            "CLARA_EVENT_STALE_DAYS": "120",
            "CLARA_SKILL_UNUSED_DAYS": "90",
            "CLARA_START_SCHEDULER": "false",
            "CLARA_CACHE_URL": "memory://",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ClaraConfig.from_env()

        assert cfg.db_url == "postgresql+asyncpg://prod/clara"
        assert cfg.embedding_backend == "local"
        assert cfg.llm_provider == "anthropic"
        assert cfg.openai_model == "gpt-4o"
        assert cfg.retrieval_top_k == 12
        assert cfg.similarity_threshold == 0.75
        assert cfg.archival_threshold == 0.20
        assert cfg.event_stale_days == 120
        assert cfg.skill_unused_days == 90
        assert cfg.start_scheduler is False
        assert cfg.cache_url == "memory://"
