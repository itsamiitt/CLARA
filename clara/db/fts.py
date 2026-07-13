"""
CLARA — SQLite FTS5 index for lexical retrieval.

Maintains a ``memories_fts`` FTS5 virtual table (porter-stemmed BM25) kept in
sync with the ``memories`` table by triggers. This is the upgrade path past
the ILIKE-over-JSON scan in :mod:`clara.retrieval.lexical`: real relevance
ranking, stemming ("deploying" matches "deployed"), and index-backed scale.

Everything here is SQLite-only and fail-soft: if FTS5 is unavailable (custom
SQLite builds) or the dialect is not SQLite, :func:`ensure_fts` returns
``False`` and the lexical retriever falls back to its scan path.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

FTS_TABLE = "memories_fts"

# The indexed text is the serialized content JSON. JSON keys appear in every
# row, so their inverse document frequency is ~0 and they contribute nothing
# to BM25 — while all searchable values (subject/relation/object/name/
# properties/…) are covered without fragile per-key extraction in triggers.
_TEXT_EXPR = "lower(CAST({row}.content AS TEXT))"

_DDL = [
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
        memory_id UNINDEXED,
        user_id UNINDEXED,
        memory_type UNINDEXED,
        status UNINDEXED,
        text,
        tokenize='porter unicode61'
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {FTS_TABLE}_ai AFTER INSERT ON memories BEGIN
        INSERT INTO {FTS_TABLE}(memory_id, user_id, memory_type, status, text)
        VALUES (
            new.memory_id,
            coalesce(new.user_id, ''),
            new.memory_type,
            new.status,
            {_TEXT_EXPR.format(row='new')}
        );
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {FTS_TABLE}_ad AFTER DELETE ON memories BEGIN
        DELETE FROM {FTS_TABLE} WHERE memory_id = old.memory_id;
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {FTS_TABLE}_au AFTER UPDATE ON memories BEGIN
        DELETE FROM {FTS_TABLE} WHERE memory_id = old.memory_id;
        INSERT INTO {FTS_TABLE}(memory_id, user_id, memory_type, status, text)
        VALUES (
            new.memory_id,
            coalesce(new.user_id, ''),
            new.memory_type,
            new.status,
            {_TEXT_EXPR.format(row='new')}
        );
    END
    """,
]

_BACKFILL = f"""
    INSERT INTO {FTS_TABLE}(memory_id, user_id, memory_type, status, text)
    SELECT memory_id, coalesce(user_id, ''), memory_type, status,
           {_TEXT_EXPR.format(row='memories')}
    FROM memories
    WHERE memory_id NOT IN (SELECT memory_id FROM {FTS_TABLE})
"""


async def ensure_fts(engine: AsyncEngine) -> bool:
    """Create the FTS5 table + sync triggers and backfill existing rows.

    Idempotent. Returns ``True`` when the index is ready, ``False`` when FTS
    is unavailable for this engine (non-SQLite dialect, or SQLite compiled
    without FTS5) — callers degrade to the scan path, never crash.
    """
    if engine.dialect.name != "sqlite":
        return False
    try:
        async with engine.begin() as conn:
            for statement in _DDL:
                await conn.execute(text(statement))
            await conn.execute(text(_BACKFILL))
        return True
    except Exception as exc:  # noqa: BLE001 — availability probe, fail soft
        logger.warning("FTS5 index unavailable (%s); lexical search will scan", exc)
        return False


def build_match_expression(tokens: list[str]) -> str:
    """Build a safe FTS5 MATCH expression from pre-tokenized query terms.

    Tokens come from :func:`clara.retrieval.lexical.tokenize` (``[a-z0-9]+``
    only), and are additionally quoted so nothing can be interpreted as FTS5
    query syntax. Each term is a prefix query (``"postgres"*`` matches
    "postgresql") — the ILIKE scan this replaces did substring matching, and
    losing "postgres" → "PostgreSQL" recall would be a regression. Terms are
    OR-ed: BM25 ranks rows matching more/rarer terms higher without
    requiring every term to be present.
    """
    return " OR ".join(f'"{token}"*' for token in tokens)
