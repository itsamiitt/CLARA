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

from clara.db.migrations import FTS_TEXT_EXPR, _fts_trigger_ddl

logger = logging.getLogger(__name__)

FTS_TABLE = "memories_fts"

# The indexed text is the serialized content JSON plus tags/origin metadata,
# shared with migration 5 (clara/db/migrations.py) so triggers written by
# either path index identical text. JSON keys appear in every row, so their
# inverse document frequency is ~0 and they contribute nothing to BM25 —
# while all searchable values are covered without per-key extraction.
_TEXT_EXPR = FTS_TEXT_EXPR

# Triggers come from the shared column-scoped builder in clara.db.migrations
# — a bare AFTER UPDATE trigger would reindex every row on bookkeeping
# writes (decay, access counts), which is a full-store FTS rebuild nightly.
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
    *_fts_trigger_ddl(FTS_TABLE),
]

_BACKFILL = f"""
    INSERT INTO {FTS_TABLE}(memory_id, user_id, memory_type, status, text)
    SELECT memory_id, coalesce(user_id, ''), memory_type, status,
           {_TEXT_EXPR.format(row='memories')}
    FROM memories
    WHERE status = 'active'
      AND memory_id NOT IN (SELECT memory_id FROM {FTS_TABLE})
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
            existed = (
                await conn.execute(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        f"WHERE type='table' AND name='{FTS_TABLE}'"
                    )
                )
            ).first() is not None
            for statement in _DDL:
                await conn.execute(text(statement))
            # The backfill is a full anti-join (memories vs FTS) that used to
            # run on every store open — expensive at scale and pointless when
            # the triggers already keep an existing table in sync. Only run it
            # when we just created the table, or when it is provably empty.
            if not existed:
                await conn.execute(text(_BACKFILL))
            else:
                empty = (
                    await conn.execute(text(f"SELECT 1 FROM {FTS_TABLE} LIMIT 1"))
                ).first() is None
                if empty:
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
