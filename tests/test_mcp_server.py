"""Tests for clara.integrations.mcp_server — clara-mcp wiring."""

from __future__ import annotations

import asyncio

import pytest

from clara.integrations.local_memory import LocalMemory
from clara.integrations.mcp_server import _recall, default_db_path

EXPECTED_TOOLS = {
    "memory_save",
    "memory_search",
    "memory_recent",
    "memory_update",
    "memory_forget",
    "memory_stats",
}


def test_default_db_path_honors_env(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "clara.db"
    monkeypatch.setenv("CLARA_DB_PATH", str(target))
    resolved = default_db_path()
    assert resolved == str(target)
    assert target.parent.exists()  # parent dir is created


def test_build_server_registers_all_tools():
    pytest.importorskip("mcp")
    from clara.integrations.mcp_server import build_server

    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert EXPECTED_TOOLS <= names


class TestRecallCLI:
    @pytest.mark.asyncio
    async def test_recall_empty_store_returns_blank(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
        assert await _recall("", top_k=8) == ""

    @pytest.mark.asyncio
    async def test_recall_returns_context_after_save(self, tmp_path, monkeypatch):
        db = str(tmp_path / "clara.db")
        monkeypatch.setenv("CLARA_DB_PATH", db)

        mem = await LocalMemory.create(db)
        try:
            await mem.save(
                mem_type="belief", subject="user", relation="prefers",
                object="dark mode",
            )
        finally:
            await mem.close()

        text = await _recall("dark mode", top_k=8)
        assert "MEMORY CONTEXT" in text
        assert "dark mode" in text
