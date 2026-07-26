"""Reflection pipeline for synthesizing higher-order beliefs from memory."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Sequence, TypeAlias

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory, MemoryStatus, MemoryType
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
    _anthropic,
    _openai,
)
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import RetrievalEngine
from clara.memory.belief import SourceType
from clara.reflection.prompts import (
    DEFAULT_REFLECTION_SYSTEM_PROMPT,
    build_reflection_prompt,
    fallback_reflection_text,
)
from clara.update.engine import MemoryUpdateEngine, UpdateResult

logger = logging.getLogger(__name__)
try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib: Any = None  # type: ignore[assignment]

from clara.core.ollama import ensure_model as _ensure_ollama_model

DEFAULT_REFLECTION_WINDOW_DAYS = 7
DEFAULT_MIN_OCCURRENCES = 3

InsightGenerator: TypeAlias = Callable[[str, str, str], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """A reflection candidate distilled from recent memories."""

    subject: str
    relation: str
    object: str
    pattern_type: str
    count: int
    supporting_memory_ids: tuple[str, ...] = ()


def _memory_subject(memory: Memory) -> str:
    content = memory.content if isinstance(memory.content, dict) else {}
    return str(content.get("subject", "") or "").strip()


def _memory_object(memory: Memory) -> str:
    content = memory.content if isinstance(memory.content, dict) else {}
    return str(content.get("object", "") or "").strip()


def _memory_relation(memory: Memory) -> str:
    content = memory.content if isinstance(memory.content, dict) else {}
    return str(content.get("relation", "") or "").strip()


def _memory_line(memory: Memory) -> str:
    subject = _memory_subject(memory) or "?"
    relation = _memory_relation(memory) or "?"
    object_ = _memory_object(memory) or "?"
    return f"{subject} {relation} {object_}"


def detect_patterns(
    memories: Sequence[Memory],
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[PatternCandidate]:
    """Detect recurring subjects, repeated events, and repeated skills."""
    patterns: list[PatternCandidate] = []

    subject_buckets: dict[str, list[Memory]] = {}
    event_buckets: dict[tuple[str, str, str], list[Memory]] = {}
    skill_buckets: dict[tuple[str, str], list[Memory]] = {}

    for memory in memories:
        subject = _memory_subject(memory)
        relation = _memory_relation(memory)
        object_ = _memory_object(memory)

        if subject:
            subject_buckets.setdefault(subject.lower(), []).append(memory)

        if memory.memory_type == MemoryType.event and subject and relation and object_:
            event_buckets.setdefault(
                (subject.lower(), relation.lower(), object_.lower()),
                [],
            ).append(memory)

        if memory.memory_type == MemoryType.skill and subject and object_:
            skill_buckets.setdefault(
                (subject.lower(), object_.lower()),
                [],
            ).append(memory)

    seen: set[tuple[str, str, str]] = set()

    for bucket in subject_buckets.values():
        if len(bucket) < min_occurrences:
            continue
        subject = _memory_subject(bucket[0])
        candidate = PatternCandidate(
            subject=subject,
            relation="appears_frequently_in",
            object="recent_activity",
            pattern_type="recurring_entity",
            count=len(bucket),
            supporting_memory_ids=tuple(str(mem.memory_id) for mem in bucket),
        )
        key = (candidate.subject.lower(), candidate.relation, candidate.object.lower())
        if key not in seen:
            patterns.append(candidate)
            seen.add(key)

    for bucket in event_buckets.values():
        if len(bucket) < min_occurrences:
            continue
        subject = _memory_subject(bucket[0])
        relation = _memory_relation(bucket[0])
        object_ = _memory_object(bucket[0])
        candidate = PatternCandidate(
            subject=subject,
            relation=f"repeatedly_{relation}",
            object=object_,
            pattern_type="repeated_event_type",
            count=len(bucket),
            supporting_memory_ids=tuple(str(mem.memory_id) for mem in bucket),
        )
        key = (candidate.subject.lower(), candidate.relation, candidate.object.lower())
        if key not in seen:
            patterns.append(candidate)
            seen.add(key)

    for bucket in skill_buckets.values():
        if len(bucket) < min_occurrences:
            continue
        subject = _memory_subject(bucket[0])
        object_ = _memory_object(bucket[0])
        candidate = PatternCandidate(
            subject=subject,
            relation="is_consistently_skilled_at",
            object=object_,
            pattern_type="skill_generalization",
            count=len(bucket),
            supporting_memory_ids=tuple(str(mem.memory_id) for mem in bucket),
        )
        key = (candidate.subject.lower(), candidate.relation, candidate.object.lower())
        if key not in seen:
            patterns.append(candidate)
            seen.add(key)

    return patterns


class ReflectionEngine:
    """Synthesizes higher-order beliefs from recent memory activity."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        *,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        ollama_base_url: str | None = None,
        insight_generator: InsightGenerator | None = None,
        window_days: int = DEFAULT_REFLECTION_WINDOW_DAYS,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        cache: MemoryCache | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedding_engine
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._ollama_base_url = (
            ollama_base_url
            or os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
        )
        self._insight_generator = insight_generator
        self._window_days = window_days
        self._min_occurrences = min_occurrences
        self._retriever = RetrievalEngine(session, embedding_engine, cache=cache)
        self._updater = MemoryUpdateEngine(
            session,
            embedding_engine,
            self._retriever,
            cache=cache,
        )

    async def recent_memories(self, *, user_id: str | None = None) -> list[Memory]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._window_days)
        filters = [
            Memory.status == MemoryStatus.active,
            Memory.created_at >= cutoff,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def run(self, user_id: str | None = None) -> list[UpdateResult]:
        """Detect patterns and persist one insight each.

        Must NOT be called inside an open transaction: each insight's LLM call
        is made with no transaction held, then that single insight is persisted
        in its own ``session.begin()``. Previously the whole per-user loop ran
        inside one transaction, pinning the SQLite writer slot across every
        pattern's network round-trip and rolling back all users' insights on a
        single late failure.
        """
        memories = await self.recent_memories(user_id=user_id)
        # Release the read snapshot before the first network call.
        await self._session.rollback()
        if not memories:
            return []

        patterns = detect_patterns(memories, min_occurrences=self._min_occurrences)
        if not patterns:
            return []

        memory_by_id = {str(memory.memory_id): memory for memory in memories}
        stored: list[UpdateResult] = []
        for pattern in patterns:
            supporting = [
                memory_by_id[memory_id]
                for memory_id in pattern.supporting_memory_ids
                if memory_id in memory_by_id
            ]
            insight_text = await self._generate_insight(pattern, supporting)
            fact = ExtractedFact(
                subject=pattern.subject,
                relation=pattern.relation,
                object=pattern.object,
                domain=None,
                source_type=SourceType.agent_reflection.value,
                confidence=0.5,
                is_negation=False,
                raw_text=insight_text,
            )
            async with self._session.begin():
                stored.append(await self._updater.process(fact, user_id=user_id))

        return stored

    async def _generate_insight(
        self,
        pattern: PatternCandidate,
        supporting_memories: Sequence[Memory],
    ) -> str:
        evidence_lines = [_memory_line(memory) for memory in supporting_memories[:5]]
        prompt = build_reflection_prompt(pattern, evidence_lines=evidence_lines)

        if self._insight_generator is not None:
            response = self._insight_generator(
                DEFAULT_REFLECTION_SYSTEM_PROMPT,
                prompt,
                self._model_name(),
            )
            if inspect.isawaitable(response):
                response = await response
            return str(response).strip() or fallback_reflection_text(pattern)

        provider = self._llm_provider.strip().lower()
        if provider == "openai":
            return await self._call_openai(prompt, pattern)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, pattern)
        if provider == "ollama":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._call_ollama,
                prompt,
                pattern,
            )
        return self._degraded(pattern, f"unknown provider {provider!r}")

    def _degraded(self, pattern: PatternCandidate, reason: str) -> str:
        """Return the templated fallback, but WARN so a misconfigured provider
        is visible. Previously this substitution was silent, so a missing API
        key produced permanently-stored canned "insights" indistinguishable
        from real ones."""
        logger.warning(
            "reflection degraded to template text (%s); insight for %r is not "
            "LLM-generated", reason, f"{pattern.subject} {pattern.relation}",
        )
        return fallback_reflection_text(pattern)

    def _model_name(self) -> str:
        if self._llm_model:
            return self._llm_model
        provider = self._llm_provider.strip().lower()
        if provider == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL
        if provider == "ollama":
            return os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL)
        return DEFAULT_OPENAI_MODEL

    async def _call_openai(self, prompt: str, pattern: PatternCandidate) -> str:
        if _openai is None:
            return self._degraded(pattern, "openai SDK not installed")
        api_key = os.environ.get(ENV_OPENAI_KEY)
        if not api_key:
            return self._degraded(pattern, f"{ENV_OPENAI_KEY} not set")

        client = _openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self._model_name(),
            messages=[
                {"role": "system", "content": DEFAULT_REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    async def _call_anthropic(self, prompt: str, pattern: PatternCandidate) -> str:
        if _anthropic is None:
            return self._degraded(pattern, "anthropic SDK not installed")
        api_key = os.environ.get(ENV_ANTHROPIC_KEY)
        if not api_key:
            return self._degraded(pattern, f"{ENV_ANTHROPIC_KEY} not set")

        client = _anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self._model_name(),
            max_tokens=512,
            system=DEFAULT_REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (response.content[0].text or "").strip()

    def _call_ollama(self, prompt: str, pattern: PatternCandidate) -> str:
        if _ollama_lib is None:
            raise ImportError(
                "The 'ollama' package is required for the Ollama reflection provider. "
                "Install it with: pip install 'clara-memory[ollama]'"
            )

        model = self._model_name()
        _ensure_ollama_model(self._ollama_base_url, model)

        client = _ollama_lib.Client(host=self._ollama_base_url)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 1024},
        )
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", "") or ""
            return str(content).strip() or self._degraded(pattern, "empty ollama response")
        if isinstance(response, dict):
            payload = response.get("message", {})
            if isinstance(payload, dict):
                content = str(payload.get("content", "") or "").strip()
                return content or self._degraded(pattern, "empty ollama response")
        return self._degraded(pattern, "unexpected ollama response shape")
