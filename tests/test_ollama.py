from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clara.db.models import VECTOR_DIMENSIONS
from clara.extraction.extractor import FactExtractor
from clara.reasoning.engine import ReasoningEngine
from clara.reflection.pipeline import PatternCandidate, ReflectionEngine
from clara.retrieval.embeddings import EmbeddingEngine, _OllamaBackend


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, text: str) -> list[float]:
        return [0.0] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


@pytest.fixture(autouse=True)
def _reset_ollama_state():
    # The per-module ready-sets were consolidated into clara.core.ollama.
    from clara.core import ollama as ollama_module

    ollama_module._models_ready.clear()
    yield
    ollama_module._models_ready.clear()


class TestOllamaEmbeddingBackend:
    def test_ollama_backend_raises_without_package(self):
        with (
            patch("clara.retrieval.embeddings._ollama_lib", None),
            pytest.raises(ImportError, match="ollama"),
        ):
            _OllamaBackend()

    def test_ollama_embed_normalizes_dimensions(self):
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"model": "nomic-embed-text"}]}
        mock_client.embeddings.return_value = {"embedding": [0.1] * 768}

        with patch("clara.retrieval.embeddings._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            backend = _OllamaBackend(model="nomic-embed-text")
            result = backend.embed("test text")

        assert len(result) == VECTOR_DIMENSIONS

    def test_ollama_embed_pulls_missing_model(self):
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}

        with patch("clara.retrieval.embeddings._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            _OllamaBackend(model="nomic-embed-text")

        mock_client.pull.assert_called_once_with("nomic-embed-text")


class TestOllamaExtractor:
    def test_extractor_raises_without_package(self):
        with (
            patch("clara.extraction.extractor._ollama_lib", None),
            pytest.raises(ImportError, match="ollama"),
        ):
            FactExtractor(provider="ollama")

    async def test_extractor_calls_ollama_with_json_format(self):
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"model": "llama3.2"}]}
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(
                content=(
                    '{"facts": [{"subject": "user", "relation": "uses", '
                    '"object": "Python", "domain": null, '
                    '"source_type": "user_direct", "confidence": 0.9, '
                    '"is_negation": false}]}'
                )
            )
        )

        with patch("clara.extraction.extractor._ollama_lib") as mock_lib, \
                patch("clara.core.ollama._ollama_lib", mock_lib):
            mock_lib.Client.return_value = mock_client
            extractor = FactExtractor(provider="ollama", model="llama3.2")
            facts = await extractor.extract("I use Python for data work.")

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"
        assert len(facts) == 1
        assert facts[0].subject == "user"
        assert facts[0].object == "Python"


class TestOllamaReasoningEngine:
    @pytest.mark.asyncio
    async def test_generate_response_uses_run_in_executor(self):
        engine = ReasoningEngine(
            MagicMock(),
            EmbeddingEngine(_FakeBackend()),
            MagicMock(),
            llm_provider="ollama",
        )
        engine._call_ollama = MagicMock(return_value="Executor reply")  # type: ignore[method-assign]

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value="Executor reply")

        with patch("clara.reasoning.engine.asyncio.get_running_loop", return_value=loop):
            response = await engine._generate_response("What do I use?", "context block")

        assert response == "Executor reply"
        loop.run_in_executor.assert_awaited_once()
        assert loop.run_in_executor.await_args.args[1] is engine._call_ollama

    def test_call_ollama_pulls_missing_model(self):
        engine = ReasoningEngine(
            MagicMock(),
            EmbeddingEngine(_FakeBackend()),
            MagicMock(),
            llm_provider="ollama",
            llm_model="llama3.2",
            ollama_base_url="http://localhost:11434",
        )

        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(content="Response using Ollama.")
        )

        with patch("clara.reasoning.engine._ollama_lib") as mock_lib, \
                patch("clara.core.ollama._ollama_lib", mock_lib):
            mock_lib.Client.return_value = mock_client
            response = engine._call_ollama("system prompt", "hello")

        assert response == "Response using Ollama."
        mock_client.pull.assert_called_once_with("llama3.2")


class TestOllamaReflectionEngine:
    @pytest.mark.asyncio
    async def test_generate_insight_uses_run_in_executor(self):
        engine = ReflectionEngine(
            MagicMock(),
            EmbeddingEngine(_FakeBackend()),
            llm_provider="ollama",
        )
        engine._call_ollama = MagicMock(return_value="Executor insight")  # type: ignore[method-assign]
        pattern = PatternCandidate(
            subject="user",
            relation="repeatedly_deployed",
            object="service",
            pattern_type="repeated_event_type",
            count=3,
        )

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value="Executor insight")

        with patch("clara.reflection.pipeline.asyncio.get_running_loop", return_value=loop):
            insight = await engine._generate_insight(pattern, evidence_lines=[])

        assert insight == "Executor insight"
        loop.run_in_executor.assert_awaited_once()
        assert loop.run_in_executor.await_args.args[1] is engine._call_ollama
