"""Tests for clara.integrations.mcp_server — clara-mcp wiring."""

from __future__ import annotations

import asyncio

import pytest

from clara.integrations.local_memory import LocalMemory
from clara.integrations.mcp_server import _recall, default_db_path

# The complete tool surface. README and SKILL.md describe this as
# "6 memory + 5 docs + 4 graph (15 total)" — this set is the enforcing test
# for that number: adding or removing a tool must update both.
EXPECTED_TOOLS = {
    # memory (6)
    "memory_save",
    "memory_search",
    "memory_recent",
    "memory_update",
    "memory_forget",
    "memory_stats",
    # docs (5)
    "docs_status",
    "docs_classify",
    "docs_supersede",
    "docs_fulfill",
    "docs_report",
    # graph (4, counting the memory_link sugar)
    "graph_entity",
    "graph_neighbors",
    "graph_path",
    "memory_link",
    # code index (3)
    "code_deps",
    "code_impact",
    "code_health",
    # status bar (2) — a plugin cannot ship a statusLine, so these let Claude
    # configure it for users who never open a terminal
    "statusline_install",
    "statusline_status",
    # project (1) — what the repo *is*, read from its manifests
    "project_profile",
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
    assert names == EXPECTED_TOOLS, (
        f"tool surface changed: +{names - EXPECTED_TOOLS} -{EXPECTED_TOOLS - names} "
        "— update EXPECTED_TOOLS, README, and SKILL.md together"
    )
    assert len(EXPECTED_TOOLS) == 21


def test_docs_status_respects_kill_switch(monkeypatch):
    pytest.importorskip("mcp")
    from clara.flags import DOCS_DISABLED_HINT
    from clara.integrations.mcp_server import build_server

    monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
    server = build_server()

    async def _call():
        return await server.call_tool("docs_status", {"path_or_query": "x.md"})

    result = asyncio.run(_call())
    # FastMCP returns (content, structured) — normalize to the payload dict.
    payload = result[1] if isinstance(result, tuple) else result
    text = str(payload)
    assert "disabled" in text
    assert DOCS_DISABLED_HINT.split("(")[0].strip() in text or "CLARA_DOCS_ENABLED" in text


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
