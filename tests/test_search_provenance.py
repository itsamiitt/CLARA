"""
memory_search and memory_recent say whose fact each hit is.

Third surface of the same defect family (session-start block, per-prompt
recall, now the agent-facing tools): the store holds several projects'
memories, each stamped at save time, and search results gave no way to tell
another project's finding from this one's.

Deliberately different from the other two surfaces: **no reordering and no
suppression.** The caller asked a question; relevance won the ranking and
should keep it. What was missing is provenance — the label in the context
block and a ``foreign`` flag on each structured hit.
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from clara.integrations.local_memory import LocalMemory
from clara.repoid import repo_id


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Two repos, one global store, facts saved from repoA."""
    for name in ("repoA", "repoB"):
        (tmp_path / name).mkdir()
        subprocess.run(["git", "init", "-q", str(tmp_path / name)],
                       capture_output=True)
    monkeypatch.chdir(tmp_path / "repoA")

    async def seed():
        memory = await LocalMemory.create(str(tmp_path / "clara.db"))
        await memory.save(mem_type="belief", subject="payments service",
                          relation="uses", object="stripe webhooks")
        await memory.save(mem_type="belief", subject="user",
                          relation="prefers", object="pnpm")
        await memory.close()

    asyncio.run(seed())
    return tmp_path


def search(store, query, repo_dir):
    async def go():
        memory = await LocalMemory.create(str(store / "clara.db"))
        try:
            return await memory.search(
                query, current_repo=repo_id(str(store / repo_dir))
            )
        finally:
            await memory.close()

    return asyncio.run(go())


class TestProvenance:
    def test_own_repo_hits_carry_no_label(self, store) -> None:
        found = search(store, "stripe webhooks payments", "repoA")
        assert found["total"] >= 1
        assert "[from another project]" not in found["context"]
        assert all(hit["foreign"] is False for hit in found["hits"])

    def test_foreign_hits_are_labeled_not_hidden(self, store) -> None:
        found = search(store, "stripe webhooks payments", "repoB")
        assert found["total"] >= 1, "provenance must not suppress results"
        assert "[from another project]" in found["context"]
        foreign = [hit for hit in found["hits"] if hit["foreign"]]
        assert foreign, "the structured hits must carry the flag too"

    def test_user_facts_are_never_foreign(self, store) -> None:
        found = search(store, "pnpm package preferences", "repoB")
        pnpm = [h for h in found["hits"]
                if h["content"].get("object") == "pnpm"]
        assert pnpm and pnpm[0]["foreign"] is False

    def test_no_repo_given_means_no_flags_at_all(self, store) -> None:
        async def go():
            memory = await LocalMemory.create(str(store / "clara.db"))
            try:
                return await memory.search("stripe webhooks payments")
            finally:
                await memory.close()

        found = asyncio.run(go())
        assert "[from another project]" not in found["context"]
        assert all("foreign" not in hit for hit in found["hits"])

    def test_recent_labels_the_same_way(self, store) -> None:
        async def go():
            memory = await LocalMemory.create(str(store / "clara.db"))
            try:
                return await memory.recent(
                    n=10, current_repo=repo_id(str(store / "repoB"))
                )
            finally:
                await memory.close()

        found = asyncio.run(go())
        assert "[from another project]" in found["context"]


class TestThroughTheMcpTool:
    def test_search_labels_foreign_hits_for_the_session_repo(
        self, store, monkeypatch
    ) -> None:
        pytest.importorskip("mcp")
        from clara.integrations.mcp_server import build_server

        monkeypatch.setenv("CLARA_HOME", str(store))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(store / "repoB"))
        server = build_server()
        result = asyncio.run(server.call_tool(
            "memory_search", {"query": "stripe webhooks payments"}
        ))
        payload = result[1] if isinstance(result, tuple) else result
        assert "[from another project]" in payload["context"]
