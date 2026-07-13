"""Tests for the SQLite FTS5 lexical index (clara/db/fts.py)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from clara.db.fts import FTS_TABLE, build_match_expression, ensure_fts
from clara.integrations.local_memory import LocalMemory


@pytest.fixture
async def memory(tmp_path):
    mem = await LocalMemory.create(str(tmp_path / "clara.db"))
    yield mem
    await mem.close()


class TestEnsureFts:
    async def test_creates_table_and_triggers(self, memory: LocalMemory):
        async with memory._engine.connect() as conn:
            names = {
                row[0]
                for row in await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE name LIKE :p"),
                    {"p": f"{FTS_TABLE}%"},
                )
            }
        assert FTS_TABLE in names
        assert {f"{FTS_TABLE}_ai", f"{FTS_TABLE}_ad", f"{FTS_TABLE}_au"} <= names

    async def test_idempotent(self, memory: LocalMemory):
        assert await ensure_fts(memory._engine) is True
        assert await ensure_fts(memory._engine) is True

    async def test_insert_populates_index(self, memory: LocalMemory):
        await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Rust",
        )
        async with memory._engine.connect() as conn:
            count = (
                await conn.execute(text(f"SELECT count(*) FROM {FTS_TABLE}"))
            ).scalar_one()
        assert count == 1


class TestFtsSearch:
    async def test_stemmed_match(self, memory: LocalMemory):
        """Porter stemming: query 'deploying' must find 'deployed'."""
        await memory.save(
            mem_type="event", subject="user", event_type="deployed",
            description="deployed the payments service",
        )
        result = await memory.search("deploying payments")
        assert result["total"] == 1

    async def test_bm25_prefers_more_matching_terms(self, memory: LocalMemory):
        await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Rust",
            domain="systems programming",
        )
        await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Python",
        )
        result = await memory.search("rust systems programming")
        assert result["total"] >= 1
        assert "rust" in str(result["hits"][0]["content"]).lower()

    async def test_status_change_removes_from_results(self, memory: LocalMemory):
        saved = await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Fortran",
        )
        await memory.forget(saved["memory_id"])
        result = await memory.search("fortran")
        assert result["total"] == 0

    async def test_scan_fallback_still_works(self, memory: LocalMemory):
        """Dropping the FTS table degrades to the ILIKE scan, not to zero."""
        await memory.save(
            mem_type="belief", subject="user", relation="prefers", object="tabs",
        )
        async with memory._engine.begin() as conn:
            for suffix in ("_ai", "_ad", "_au"):
                await conn.execute(text(f"DROP TRIGGER {FTS_TABLE}{suffix}"))
            await conn.execute(text(f"DROP TABLE {FTS_TABLE}"))
        result = await memory.search("tabs")
        assert result["total"] == 1


class TestMatchExpression:
    def test_tokens_are_quoted(self):
        assert build_match_expression(["rust", "http2"]) == '"rust" OR "http2"'
