"""CLARA - Reasoning engine.

Assembles memory context, calls an LLM responder, and feeds any facts from the
response back into the update engine.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TypeAlias

from sqlalchemy.ext.asyncio import AsyncSession

from clara.core import llm as _llm
from clara.core.ollama import ensure_model as _ensure_ollama_model
from clara.extraction.extractor import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    ENV_ANTHROPIC_KEY,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    ENV_OPENAI_KEY,
    FactExtractor,
    _anthropic,
    _openai,
)
from clara.extraction.heuristic import HeuristicExtractor
from clara.reasoning.context import ContextAssembler
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import RetrievalEngine, ScoredMemory
from clara.update.engine import MemoryUpdateEngine, UpdateResult

logger = logging.getLogger(__name__)
try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib = None  # type: ignore[assignment]

DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided memory context when it is relevant. "
    "If the context does not support a claim, say you do not know instead of inventing details."
)

ResponseGenerator: TypeAlias = Callable[[str, str, str], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ReasoningResponse:
    """Result of the reasoning loop.

    ``facts_considered`` is the number of facts the extractor pulled from the
    response (each yields one entry in ``facts_stored``); it lets callers see
    how much was processed without inspecting every result.
    """

    text: str
    memory_context: str
    facts_stored: list[UpdateResult]
    memories_used: list[ScoredMemory]
    facts_considered: int = 0


class ReasoningEngine:
    """Full reasoning loop over memory context + LLM response generation."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        extractor: FactExtractor | HeuristicExtractor,
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
        """Run retrieval, reasoning, and response-fact persistence.

        The three phases keep separate transaction boundaries: retrieval reads,
        then the read snapshot is released *before* the LLM call so the writer
        slot is not held across a network round-trip, then persistence runs in
        its own write transaction. The caller must NOT wrap this in an open
        transaction, or the release below is a no-op.
        """
        retrieval_result, memory_context = await self._assembler.assemble(
            query,
            user_id=user_id,
            top_k=top_k,
        )
        # Release any implicit read transaction before the network call.
        await self._session.rollback()

        response_text = await self._generate_response(
            query,
            memory_context,
            system_prompt=system_prompt,
        )

        stored: list[UpdateResult] = []
        async with self._session.begin():
            # Persist what the USER said first, at full trust. Previously only
            # the assistant's reply was extracted, so a fact stated by the user
            # ("I switched to Rust") was stored only if the model happened to
            # restate it — and then at the downgraded agent_inference weight.
            # Extract the user's own text at user_direct (the extractor's
            # default) so it can supersede stale beliefs like remember() does.
            if query.strip():
                for fact in await self._extractor.extract(query):
                    stored.append(await self._updater.process(fact, user_id=user_id))
            if response_text.strip():
                facts = await self._extractor.extract(response_text)
                for fact in facts:
                    # These facts come from the assistant's own generated reply,
                    # not from the user. The extractor's default source_type is
                    # "user_direct" (trust weight 1.0), which would let a
                    # hallucinated claim supersede a genuine user-stated belief
                    # at the >0.6 confidence threshold. Downgrade to
                    # agent_inference (0.5) so model output can never outrank
                    # what the user actually said.
                    fact = replace(fact, source_type="agent_inference")
                    stored.append(await self._updater.process(fact, user_id=user_id))

        return ReasoningResponse(
            text=response_text,
            memory_context=memory_context,
            facts_stored=stored,
            memories_used=retrieval_result.all,
            facts_considered=len(stored),
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
        _llm.require_sdk(_openai, "openai", "OpenAI reasoning provider")
        api_key = _llm.require_api_key(ENV_OPENAI_KEY, os.environ.get)
        return await _llm.openai_chat(
            _openai,
            system=system_prompt,
            user=query,
            model=self._model_name(),
            api_key=api_key,
            temperature=0.2,
        )

    async def _call_anthropic(self, system_prompt: str, query: str) -> str:
        _llm.require_sdk(_anthropic, "anthropic", "Anthropic reasoning provider")
        api_key = _llm.require_api_key(ENV_ANTHROPIC_KEY, os.environ.get)
        return await _llm.anthropic_chat(
            _anthropic,
            system=system_prompt,
            user=query,
            model=self._model_name(),
            api_key=api_key,
            temperature=0.2,
            max_tokens=2048,
        )

    def _call_ollama(self, system_prompt: str, query: str) -> str:
        _llm.require_sdk(
            _ollama_lib, "ollama", "Ollama reasoning provider",
            install="'clara-memory[ollama]'",
        )
        model = self._model_name()
        _ensure_ollama_model(self._ollama_base_url, model)

        response = _llm.ollama_chat(
            _ollama_lib,
            system=system_prompt,
            user=query,
            model=model,
            base_url=self._ollama_base_url,
            temperature=0.2,
            num_predict=2048,
        )
        return _llm.ollama_text(response) or ""
