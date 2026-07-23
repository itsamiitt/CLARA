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
import unicodedata
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
from clara.retrieval.synonyms import expand_groups, flatten

logger = logging.getLogger(__name__)

# Default ceiling on how many active rows we pull before re-ranking in Python.
# Used by the ILIKE fallback; the FTS5 path is index-ranked and needs no cap.
DEFAULT_CANDIDATE_LIMIT = 1000

TRIGRAM_TABLE = "memories_fts_tri"

# Word characters minus underscore, any script — not just [a-z0-9]. CJK text
# has no word boundaries, so contiguous CJK runs are additionally split into
# character bigrams below.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compat
)

# Tiny stop-word set so common words don't dominate the overlap score.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "what", "when", "which",
        "who", "with", "you", "your", "do", "does", "did", "my", "me", "we",
    }
)


def _fold(text: str) -> str:
    """Casefold + strip combining marks — matches what FTS5 unicode61 does
    index-side (``remove_diacritics``), so "café" queries hit "cafe" rows."""
    nfkd = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _has_cjk(token: str) -> bool:
    return any(
        lo <= ord(ch) <= hi for ch in token for lo, hi in _CJK_RANGES
    )


def tokenize(text: str) -> list[str]:
    """Fold, split on non-word chars, drop stop-words and 1-char tokens.

    CJK runs (no word boundaries) are kept whole *and* split into character
    bigrams, so both prefix FTS matches and overlap scoring work on them.
    """
    out: list[str] = []
    for tok in _TOKEN_RE.findall(_fold(text)):
        if _has_cjk(tok):
            out.append(tok)
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        elif len(tok) > 1 and tok not in _STOPWORDS:
            out.append(tok)
    return out


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
        self._tri_available: bool | None = None

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

        # Clamp top_k: a negative value would become a Python slice
        # ``scored[:-1]`` and silently drop the lowest-ranked hit instead of
        # returning the requested count.
        top_k = max(0, int(top_k))
        if top_k == 0:
            return RetrievalResult()

        query_tokens = tokenize(query or "")
        token_groups = expand_groups(query_tokens)
        search_tokens = flatten(token_groups)

        weighted: list[tuple[Memory, float]] | None = None
        if search_tokens and await self._has_fts():
            try:
                weighted = await self._fetch_fts_candidates(
                    search_tokens,
                    limit=max(top_k * 4, 32),
                    memory_types=memory_types,
                    user_id=user_id,
                )
                # The porter index tokenizes a CJK run as one term, so only
                # prefix matches land; the trigram twin covers substrings.
                # It also rescues short result sets (rare/partial terms).
                if (
                    any(_has_cjk(tok) for tok in search_tokens)
                    or len(weighted) < top_k
                ) and await self._has_trigram():
                    tri = await self._fetch_fts_candidates(
                        [t for t in search_tokens if len(t) >= 3],
                        limit=max(top_k * 4, 32),
                        memory_types=memory_types,
                        user_id=user_id,
                        table=TRIGRAM_TABLE,
                        prefix=False,
                    )
                    weighted = self._merge_weighted(weighted, tri)
            except Exception:  # noqa: BLE001 — degrade to the scan path
                logger.exception("FTS5 search failed; falling back to scan")
                self._fts_available = False
                weighted = None
            else:
                # Zero FTS hits ≠ zero matches: rows written before the
                # raw-UTF-8 serializer carry \uXXXX escapes the index can't
                # match, and the scan path (which parses the JSON) can. The
                # sim>0 filter below keeps precision.
                if not weighted:
                    weighted = None

        if weighted is None:
            candidates = await self._fetch_candidates(
                search_tokens,
                memory_types=memory_types,
                user_id=user_id,
            )
            weighted = [
                (memory, self._similarity(token_groups, memory))
                for memory in candidates
            ]
            # When the user typed real terms, drop rows that match nothing.
            if search_tokens:
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
        self._fts_available = await self._table_exists(FTS_TABLE)
        return self._fts_available

    async def _has_trigram(self) -> bool:
        if self._tri_available is None:
            self._tri_available = await self._table_exists(TRIGRAM_TABLE)
        return self._tri_available

    async def _table_exists(self, name: str) -> bool:
        bind = self._session.get_bind()
        if bind is None or bind.dialect.name != "sqlite":
            return False
        try:
            result = await self._session.execute(
                sa_text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :n"
                ),
                {"n": name},
            )
            return result.first() is not None
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _merge_weighted(
        primary: list[tuple[Memory, float]],
        secondary: list[tuple[Memory, float]],
    ) -> list[tuple[Memory, float]]:
        """Union candidate lists, keeping the best similarity per memory."""
        best: dict[str, tuple[Memory, float]] = {}
        for memory, sim in [*primary, *secondary]:
            key = str(memory.memory_id)
            kept = best.get(key)
            if kept is None or sim > kept[1]:
                best[key] = (memory, sim)
        return list(best.values())

    async def _fetch_fts_candidates(
        self,
        query_tokens: Sequence[str],
        *,
        limit: int,
        memory_types: Sequence[MemoryType] | None,
        user_id: str | None,
        table: str = FTS_TABLE,
        prefix: bool = True,
    ) -> list[tuple[Memory, float]]:
        """BM25-ranked candidates from an FTS5 index, hydrated from SQLite.

        BM25 rank (lower = better, typically negative) is mapped to a (0, 1)
        similarity with a sigmoid — the same normalization trick Mem0 uses —
        so it slots into the shared composite score's 0.65 similarity term.
        ``table``/``prefix`` select between the porter index (prefix queries)
        and the trigram twin (bare quoted terms = substring semantics).
        """
        terms = list(dict.fromkeys(query_tokens))
        if not terms:
            return []
        if prefix:
            match_expr = build_match_expression(terms)
        else:
            match_expr = " OR ".join(f'"{term}"' for term in terms)
        clauses = [f"{table} MATCH :match", "status = 'active'"]
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
                    f"SELECT memory_id, bm25({table}) AS rank "
                    f"FROM {table} WHERE {' AND '.join(clauses)} "
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
    def _similarity(token_groups: Sequence[set[str]], memory: Memory) -> float:
        """Fraction of query concepts present in the memory text (0..1).

        Each group is one typed token plus its synonyms — a group counts as
        matched when *any* member appears, so expansion widens recall without
        deflating the score of literal matches.
        """
        if not token_groups:
            return 0.0
        memory_tokens = set(tokenize(_memory_text(memory)))
        if not memory_tokens:
            return 0.0
        matched = sum(1 for group in token_groups if group & memory_tokens)
        return matched / len(token_groups)

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
