"""
CLARA — Lexical (keyword) Retrieval Engine

A zero-backend alternative to the vector :class:`RetrievalEngine`. It ranks
memories using plain keyword matching over the SQLite ``content`` payload and
reuses the same composite scoring (similarity + confidence + recency + usage)
as the vector path, so callers get the identical :class:`RetrievalResult`.

This exists for the "Claude is the brain" integration: Claude supplies the
query terms, so we need no embedding model, no LanceDB, and no LLM — only the
SQLite source of truth.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import String, and_, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.retrieval.engine import (
    RetrievalEngine,
    RetrievalResult,
    ScoredMemory,
    compute_final_score,
    compute_recency_score,
    compute_usage_frequency,
)

logger = logging.getLogger(__name__)

# Default ceiling on how many active rows we pull before re-ranking in Python.
# Fine at personal scale; a FTS5 index is the upgrade path past this.
DEFAULT_CANDIDATE_LIMIT = 1000

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tiny stop-word set so common words don't dominate the overlap score.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "what", "when", "which",
        "who", "with", "you", "your", "do", "does", "did", "my", "me", "we",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stop-words and 1-char tokens."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _memory_text(memory: Memory) -> str:
    """Flatten a memory's searchable fields into one lowercase string."""
    content = memory.content if isinstance(memory.content, dict) else {}
    parts: list[str] = []

    for key in ("subject", "relation", "object", "name", "description",
                "entity_type", "domain"):
        value = content.get(key)
        if isinstance(value, str) and value:
            parts.append(value)

    for key in ("trigger_conditions", "steps"):
        value = content.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)

    props = content.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            parts.append(f"{k} {v}")

    meta = memory.metadata_ if isinstance(memory.metadata_, dict) else {}
    tags = meta.get("tags")
    if isinstance(tags, (list, tuple)):
        parts.extend(str(tag) for tag in tags)

    return " ".join(parts).lower()


class LexicalRetriever:
    """Keyword retrieval over the Unified Memory Store. Session-bound, read-only."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self._session = session
        self._candidate_limit = candidate_limit

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """Return ranked, grouped memories matching *query* by keyword overlap.

        An empty/blank query falls back to a recency + confidence ranking, which
        makes this usable as a "most relevant recent memories" feed.
        """
        if memory_types is not None and len(memory_types) == 0:
            return RetrievalResult()

        query_tokens = tokenize(query or "")
        candidates = await self._fetch_candidates(
            query_tokens,
            memory_types=memory_types,
            user_id=user_id,
        )
        if not candidates:
            return RetrievalResult()

        now = datetime.now(timezone.utc)
        max_access = max(
            (RetrievalEngine._get_access_count(m) for m in candidates),
            default=0,
        )

        scored: list[ScoredMemory] = []
        for memory in candidates:
            similarity = self._similarity(query_tokens, memory)
            # When the user typed real terms, drop rows that match nothing.
            if query_tokens and similarity <= 0.0:
                continue
            access = RetrievalEngine._get_access_count(memory)
            recency = compute_recency_score(memory.updated_at, now)
            usage = compute_usage_frequency(access, max_access)
            final = compute_final_score(
                similarity=similarity,
                confidence=memory.confidence,
                recency_score=recency,
                usage_frequency=usage,
            )
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=final,
                    similarity=similarity,
                    confidence=memory.confidence,
                    recency_score=recency,
                    usage_frequency=usage,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = scored[:top_k]
        return RetrievalEngine._group_scored(scored)

    @staticmethod
    def _similarity(query_tokens: Sequence[str], memory: Memory) -> float:
        """Fraction of distinct query tokens present in the memory text (0..1)."""
        if not query_tokens:
            return 0.0
        memory_tokens = set(tokenize(_memory_text(memory)))
        if not memory_tokens:
            return 0.0
        unique_query = set(query_tokens)
        matched = sum(1 for tok in unique_query if tok in memory_tokens)
        return matched / len(unique_query)

    async def _fetch_candidates(
        self,
        query_tokens: Sequence[str],
        *,
        memory_types: Sequence[MemoryType] | None,
        user_id: str | None,
    ) -> list[Memory]:
        filters: list[Any] = [Memory.status == MemoryStatus.active]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if memory_types:
            filters.append(Memory.memory_type.in_(list(memory_types)))

        if query_tokens:
            content_text = cast(Memory.content, String)
            filters.append(
                or_(*[content_text.ilike(f"%{tok}%") for tok in set(query_tokens)])
            )

        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.updated_at.desc())
            .limit(self._candidate_limit)
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        if len(rows) >= self._candidate_limit:
            logger.warning(
                "LexicalRetriever hit candidate_limit=%d; results may be partial. "
                "Consider an FTS index for stores this large.",
                self._candidate_limit,
            )
        return rows
