"""CLARA - Reasoning engine.

Assembles memory context, calls an LLM responder, and feeds any facts from the
response back into the update engine.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, TypeAlias

from sqlalchemy.ext.asyncio import AsyncSession

from clara.extraction.extractor import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    ENV_ANTHROPIC_KEY,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    ENV_OPENAI_KEY,
    ExtractedFact,
    FactExtractor,
    _anthropic,
    _openai,
)
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import RetrievalEngine, ScoredMemory
from clara.reasoning.context import ContextAssembler
from clara.update.engine import MemoryUpdateEngine, UpdateResult

logger = logging.getLogger(__name__)
try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib: Any = None  # type: ignore[assignment]

from clara.core.ollama import ensure_model as _ensure_ollama_model

DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided memory context when it is relevant. "
    "If the context does not support a claim, say you do not know instead of inventing details."
)

ResponseGenerator: TypeAlias = Callable[[str, str, str], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ReasoningResponse:
    """Result of the reasoning loop."""

    text: str
    memory_context: str
    facts_stored: list[UpdateResult]
    memories_used: list[ScoredMemory]


class ReasoningEngine:
    """Full reasoning loop over memory context + LLM response generation."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        extractor: FactExtractor,
        *,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        ollama_base_url: str | None = None,
        response_generator: ResponseGenerator | None = None,
        cache: MemoryCache | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedding_engine
        self._extractor = extractor
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._ollama_base_url = (
            ollama_base_url
            or os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
        )
        self._response_generator = response_generator
        self._retriever = RetrievalEngine(session, embedding_engine, cache=cache)
        self._assembler = ContextAssembler(self._retriever)
        self._updater = MemoryUpdateEngine(
            session,
            embedding_engine,
            self._retriever,
            cache=cache,
        )

    async def respond(
        self,
        query: str,
        *,
        user_id: str | None = None,
        system_prompt: str | None = None,
        top_k: int = 8,
    ) -> ReasoningResponse:
        """Run retrieval, reasoning, and response-fact persistence."""
        retrieval_result, memory_context = await self._assembler.assemble(
            query,
            user_id=user_id,
            top_k=top_k,
        )
        response_text = await self._generate_response(
            query,
            memory_context,
            system_prompt=system_prompt,
        )

        stored: list[UpdateResult] = []
        if response_text.strip():
            for fact in await self._extractor.extract(response_text):
                stored.append(await self._updater.process(fact, user_id=user_id))

        return ReasoningResponse(
            text=response_text,
            memory_context=memory_context,
            facts_stored=stored,
            memories_used=retrieval_result.all,
        )

    async def _generate_response(
        self,
        query: str,
        memory_context: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        final_system_prompt = "\n\n".join(
            part for part in (
                DEFAULT_REASONING_SYSTEM_PROMPT,
                system_prompt,
                memory_context,
            )
            if part
        )

        if self._response_generator is not None:
            response = self._response_generator(final_system_prompt, query, self._model_name())
            if inspect.isawaitable(response):
                response = await response
            return str(response)

        provider = self._llm_provider.strip().lower()
        if provider == "openai":
            return await self._call_openai(final_system_prompt, query)
        if provider == "anthropic":
            return await self._call_anthropic(final_system_prompt, query)
        if provider == "ollama":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._call_ollama,
                final_system_prompt,
                query,
            )
        raise ValueError(f"Unknown reasoning provider {self._llm_provider!r}.")

    def _model_name(self) -> str:
        if self._llm_model:
            return self._llm_model
        provider = self._llm_provider.strip().lower()
        if provider == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL
        if provider == "ollama":
            return os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL)
        return DEFAULT_OPENAI_MODEL

    async def _call_openai(self, system_prompt: str, query: str) -> str:
        if _openai is None:
            raise ImportError(
                "The 'openai' package is required for the OpenAI reasoning provider."
            )
        api_key = os.environ.get(ENV_OPENAI_KEY)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {ENV_OPENAI_KEY!r} is not set."
            )

        client = _openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self._model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def _call_anthropic(self, system_prompt: str, query: str) -> str:
        if _anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic reasoning provider."
            )
        api_key = os.environ.get(ENV_ANTHROPIC_KEY)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {ENV_ANTHROPIC_KEY!r} is not set."
            )

        client = _anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self._model_name(),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
            temperature=0.2,
        )
        return response.content[0].text or ""

    def _call_ollama(self, system_prompt: str, query: str) -> str:
        if _ollama_lib is None:
            raise ImportError(
                "The 'ollama' package is required for the Ollama reasoning provider. "
                "Install it with: pip install 'clara-memory[ollama]'"
            )

        model = self._model_name()
        _ensure_ollama_model(self._ollama_base_url, model)

        client = _ollama_lib.Client(host=self._ollama_base_url)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.2, "num_predict": 2048},
        )
        message = getattr(response, "message", None)
        if message is not None:
            return getattr(message, "content", "") or ""
        if isinstance(response, dict):
            payload = response.get("message", {})
            if isinstance(payload, dict):
                return str(payload.get("content", "") or "")
        return ""
