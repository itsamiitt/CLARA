"""Tests for clara.integrations.local_memory — LocalMemory (zero backend)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from clara.integrations.local_memory import LocalMemory


@pytest_asyncio.fixture
async def memory(tmp_path):
    mem = await LocalMemory.create(str(tmp_path / "clara.db"))
    try:
        yield mem
    finally:
        await mem.close()


class TestSaveAndSearch:
    @pytest.mark.asyncio
    async def test_save_belief_and_recall(self, memory):
        saved = await memory.save(
            mem_type="belief", subject="user", relation="prefers",
            object="concise answers", tags=["style"],
        )
        assert saved["type"] == "belief"
        assert saved["memory_id"]

        found = await memory.search("concise answers")
        assert found["total"] == 1
        assert "MEMORY CONTEXT" in found["context"]
        assert found["hits"][0]["content"]["object"] == "concise answers"

    @pytest.mark.asyncio
    async def test_all_four_types(self, memory):
        await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Rust"
        )
        await memory.save(
            mem_type="event", subject="user", event_type="shipped",
            description="shipped release v2",
        )
        await memory.save(
            mem_type="skill", name="deploy-api",
            trigger_conditions=["deploy"], steps=["build", "push"],
        )
        await memory.save(
            mem_type="world_model", entity_type="service", name="api",
            properties={"port": 8000, "status": "running"},
        )
        stats = await memory.stats()
        assert stats["active_by_type"] == {
            "belief": 1, "event": 1, "skill": 1, "world_model": 1
        }

    @pytest.mark.asyncio
    async def test_missing_required_fields_raise(self, memory):
        with pytest.raises(ValueError):
            await memory.save(mem_type="belief", subject="user")  # no relation/object
        with pytest.raises(ValueError):
            await memory.save(mem_type="nonsense")

    @pytest.mark.asyncio
    async def test_recent(self, memory):
        await memory.save(
            mem_type="belief", subject="user", relation="likes", object="tabs"
        )
        recent = await memory.recent(n=5)
        assert recent["total"] == 1


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_forget_excludes_from_search(self, memory):
        saved = await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Vim"
        )
        assert (await memory.search("Vim"))["total"] == 1

        result = await memory.forget(saved["memory_id"])
        assert result["action"] == "deprecated"
        assert (await memory.search("Vim"))["total"] == 0

    @pytest.mark.asyncio
    async def test_update_confidence(self, memory):
        saved = await memory.save(
            mem_type="belief", subject="user", relation="uses", object="Emacs"
        )
        out = await memory.update(saved["memory_id"], confidence=0.99)
        assert out["action"] == "updated"
        hit = (await memory.search("Emacs"))["hits"][0]
        assert hit["confidence"] == pytest.approx(0.99)


class TestNoBackend:
    @pytest.mark.asyncio
    async def test_no_vector_store_created(self, tmp_path):
        mem = await LocalMemory.create(str(tmp_path / "clara.db"))
        try:
            await mem.save(
                mem_type="belief", subject="user", relation="uses", object="SQLite"
            )
            await mem.search("SQLite")
        finally:
            await mem.close()

        assert (tmp_path / "clara.db").exists()
        # Zero-backend: LanceDB must never have been touched.
        assert not (tmp_path / "clara_vectors").exists()
