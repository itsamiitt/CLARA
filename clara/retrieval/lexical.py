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
import math
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import String, and_, cast, or_, select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.fts import FTS_TABLE, build_match_expression
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.docs import TIER_MULTIPLIER
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
# Used by the ILIKE fallback; the FTS5 path is index-ranked and needs no cap.
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
        self._fts_available: bool | None = None

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        memory_types: Sequence[MemoryType] | None = None,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """Return ranked, grouped memories matching *query* lexically.

        Uses the porter-stemmed FTS5/BM25 index when present (see
        :mod:`clara.db.fts`), degrading to the token-overlap ILIKE scan when
        it is not. An empty/blank query falls back to a recency + confidence
        ranking, which makes this usable as a "most relevant recent
        memories" feed.
        """
        if memory_types is not None and len(memory_types) == 0:
            return RetrievalResult()

        query_tokens = tokenize(query or "")

        weighted: list[tuple[Memory, float]] | None = None
        if query_tokens and await self._has_fts():
            try:
                weighted = await self._fetch_fts_candidates(
                    query_tokens,
                    limit=max(top_k * 4, 32),
                    memory_types=memory_types,
                    user_id=user_id,
                )
            except Exception:  # noqa: BLE001 — degrade to the scan path
                logger.exception("FTS5 search failed; falling back to scan")
                self._fts_available = False
                weighted = None

        if weighted is None:
            candidates = await self._fetch_candidates(
                query_tokens,
                memory_types=memory_types,
                user_id=user_id,
            )
            weighted = [
                (memory, self._similarity(query_tokens, memory))
                for memory in candidates
            ]
            # When the user typed real terms, drop rows that match nothing.
            if query_tokens:
                weighted = [(m, sim) for m, sim in weighted if sim > 0.0]

        if not weighted:
            return RetrievalResult()

        now = datetime.now(timezone.utc)
        max_access = max(
            (RetrievalEngine._get_access_count(m) for m, _ in weighted),
            default=0,
        )

        scored: list[ScoredMemory] = []
        for memory, similarity in weighted:
            # Doc-derived provenance: tier multiplier on the composite score;
            # TX (quarantined) provenance is excluded from default context —
            # mirrored in clara/fastpath/context.py (parity test enforces it).
            meta = memory.metadata_ if isinstance(memory.metadata_, dict) else {}
            doc_tier = meta.get("doc_tier")
            if doc_tier == "TX":
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
            if isinstance(doc_tier, str):
                final *= TIER_MULTIPLIER.get(doc_tier, 1.0)
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

    async def _has_fts(self) -> bool:
        """Probe (once per retriever) whether the FTS5 index exists."""
        if self._fts_available is not None:
            return self._fts_available
        bind = self._session.get_bind()
        if bind is None or bind.dialect.name != "sqlite":
            self._fts_available = False
            return False
        try:
            result = await self._session.execute(
                sa_text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :n"
                ),
                {"n": FTS_TABLE},
            )
            self._fts_available = result.first() is not None
        except Exception:  # noqa: BLE001
            self._fts_available = False
        return self._fts_available

    async def _fetch_fts_candidates(
        self,
        query_tokens: Sequence[str],
        *,
        limit: int,
        memory_types: Sequence[MemoryType] | None,
        user_id: str | None,
    ) -> list[tuple[Memory, float]]:
        """BM25-ranked candidates from the FTS5 index, hydrated from SQLite.

        BM25 rank (lower = better, typically negative) is mapped to a (0, 1)
        similarity with a sigmoid — the same normalization trick Mem0 uses —
        so it slots into the shared composite score's 0.65 similarity term.
        """
        match_expr = build_match_expression(list(dict.fromkeys(query_tokens)))
        clauses = [f"{FTS_TABLE} MATCH :match", "status = 'active'"]
        params: dict[str, Any] = {"match": match_expr, "limit": limit}
        if user_id is not None:
            clauses.append("user_id = :user_id")
            params["user_id"] = user_id
        if memory_types:
            names = sorted({mt.value for mt in memory_types})
            placeholders = ", ".join(f":mt{i}" for i in range(len(names)))
            clauses.append(f"memory_type IN ({placeholders})")
            params.update({f"mt{i}": name for i, name in enumerate(names)})

        rows = (
            await self._session.execute(
                sa_text(
                    f"SELECT memory_id, bm25({FTS_TABLE}) AS rank "
                    f"FROM {FTS_TABLE} WHERE {' AND '.join(clauses)} "
                    f"ORDER BY rank LIMIT :limit"
                ),
                params,
            )
        ).all()
        if not rows:
            return []

        similarity_by_id: dict[str, float] = {}
        ordered_ids: list[uuid.UUID] = []
        for memory_id, rank in rows:
            try:
                parsed = uuid.UUID(str(memory_id))
            except (TypeError, ValueError):
                continue
            clamped = max(-50.0, min(50.0, float(rank if rank is not None else 0.0)))
            similarity_by_id[str(parsed)] = 1.0 / (1.0 + math.exp(clamped))
            ordered_ids.append(parsed)

        if not ordered_ids:
            return []

        # Hydrate from the memories table — the source of truth — re-checking
        # status/tenant/type in case the FTS row is momentarily stale.
        filters: list[Any] = [
            Memory.memory_id.in_(ordered_ids),
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if memory_types:
            filters.append(Memory.memory_type.in_(list(memory_types)))
        result = await self._session.execute(select(Memory).where(and_(*filters)))
        by_id = {str(m.memory_id): m for m in result.scalars().all()}

        return [
            (by_id[key], similarity_by_id[key])
            for key in (str(mid) for mid in ordered_ids)
            if key in by_id
        ]

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
