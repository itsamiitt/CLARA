"""
Async SQLAlchemy engine factory with CLARA's SQLite tuning.

Lives here rather than in clara.agent so that opening a store does not drag in
the LLM tier. clara.agent imports the extractor and the reasoning engine, which
import the OpenAI SDK (2.2 s), and LocalMemory needed nothing from agent except
this function -- so every `clara` command that touched the store paid for a
stack it never used.

``from clara.agent import _make_engine`` still works; agent re-exports it.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

# Matches _BUSY_TIMEOUT_MS in clara/db/migrations.py: the stdlib and ORM paths
# open the same file and must wait the same amount of time for a writer.
SQLITE_BUSY_TIMEOUT_MS = 30_000


def _is_in_memory_sqlite(db_url: str) -> bool:
    return db_url in {"sqlite://", "sqlite+aiosqlite://"} or ":memory:" in db_url


def _make_engine(db_url: str) -> AsyncEngine:
    """Create a database engine with SQLite-specific concurrency tuning."""
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {
        "echo": False,
        # Store JSON as raw UTF-8, not \uXXXX escapes: the FTS index reads
        # CAST(content AS TEXT), and escaped CJK/diacritics would never
        # match a query typed in the original script.
        "json_serializer": lambda obj: json.dumps(obj, ensure_ascii=False),
    }
    if db_url.startswith("sqlite"):
        connect_args["timeout"] = SQLITE_BUSY_TIMEOUT_MS / 1000
        if _is_in_memory_sqlite(db_url):
            engine_kwargs["poolclass"] = StaticPool
        else:
            # File-backed SQLite handles bursty concurrent reads better when
            # sessions don't block on a small QueuePool.
            engine_kwargs["poolclass"] = NullPool

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    engine = create_async_engine(db_url, **engine_kwargs)

    if db_url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout = 30000")
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
            finally:
                cursor.close()

    return engine
