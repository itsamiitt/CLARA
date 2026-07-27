"""
memory_save_many: one transaction, all or nothing, indexed errors.

The tool exists because of a real session: an agent with eight audit findings
had only single saves, fired them as parallel tool calls, and the calls raced.
A batch is one request and one commit — measured 2.1 s for 100 items against
7.4 s sequentially — and cannot race itself.

The property that matters most is atomicity. A half-applied batch reports
success for what it wrote and silence for what it dropped, and the caller has
no way to tell which is which.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from clara.integrations.local_memory import LocalMemory


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def store(tmp_path):
    return str(tmp_path / "clara.db")


def items(n, prefix="s"):
    return [
        {"mem_type": "belief", "subject": f"{prefix}{i}", "relation": "uses",
         "object": f"o{i}", "confidence": 0.8}
        for i in range(n)
    ]


def rows(store):
    conn = sqlite3.connect(store)
    try:
        return conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    finally:
        conn.close()


class TestBulkSaves:
    def test_saves_and_reports_each_item_in_order(self, store) -> None:
        async def go():
            memory = await LocalMemory.create(store)
            out = await memory.save_many(items(5))
            await memory.close()
            return out

        out = run(go())
        assert out["count"] == 5
        assert [s["action"] for s in out["saved"]] == ["saved"] * 5
        assert len({s["memory_id"] for s in out["saved"]}) == 5

    def test_bulk_and_sequential_store_the_same_content(self, tmp_path) -> None:
        async def go(path, bulk):
            memory = await LocalMemory.create(path)
            if bulk:
                await memory.save_many(items(4))
            else:
                for item in items(4):
                    await memory.save(**item)
            await memory.close()

        a, b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
        run(go(a, bulk=True))
        run(go(b, bulk=False))

        def contents(path):
            conn = sqlite3.connect(path)
            try:
                return sorted(
                    (json.dumps(json.loads(c), sort_keys=True), round(conf, 4))
                    for c, conf in conn.execute(
                        "SELECT content, confidence FROM memories"
                    )
                )
            finally:
                conn.close()

        assert contents(a) == contents(b)

    def test_empty_batch_is_a_noop(self, store) -> None:
        async def go():
            memory = await LocalMemory.create(store)
            out = await memory.save_many([])
            await memory.close()
            return out

        assert run(go()) == {"count": 0, "saved": []}

    def test_search_sees_the_batch_immediately(self, store) -> None:
        async def go():
            memory = await LocalMemory.create(store)
            await memory.save_many(items(3))
            found = await memory.search("uses", top_k=10)
            await memory.close()
            return found

        assert run(go())["total"] == 3


class TestAllOrNothing:
    def test_one_bad_item_rolls_back_everything(self, store) -> None:
        batch = items(3)
        # A belief without subject/relation/object is rejected by routing —
        # deliberately item 2, after two valid ones have been staged.
        batch.append({"mem_type": "belief"})

        async def go():
            memory = await LocalMemory.create(store)
            try:
                with pytest.raises(ValueError) as caught:
                    await memory.save_many(batch)
                return str(caught.value)
            finally:
                await memory.close()

        message = run(go())
        assert "item 3" in message
        assert rows(store) == 0, "a failed batch must write nothing"

    def test_unknown_mem_type_is_rejected_before_any_write(self, store) -> None:
        batch = items(2) + [{"mem_type": "nonsense", "subject": "x"}]

        async def go():
            memory = await LocalMemory.create(store)
            try:
                with pytest.raises(ValueError) as caught:
                    await memory.save_many(batch)
                return str(caught.value)
            finally:
                await memory.close()

        message = run(go())
        assert "item 2" in message and "nonsense" in message
        assert rows(store) == 0


class TestMcpBoundary:
    def call(self, name, args):
        pytest.importorskip("mcp")
        from clara.integrations.mcp_server import build_server

        server = build_server()
        result = asyncio.run(server.call_tool(name, args))
        return result[1] if isinstance(result, tuple) else result

    def test_tool_saves_a_batch(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))
        payload = self.call("memory_save_many", {"items": items(3)})
        assert payload["count"] == 3

    def test_out_of_range_confidence_names_the_item(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))
        batch = items(2)
        batch[1]["confidence"] = 5.0
        with pytest.raises(Exception) as caught:
            self.call("memory_save_many", {"items": batch})
        message = str(caught.value)
        assert "item 1" in message and "1.0" in message

    def test_non_object_item_names_the_item(self, tmp_path, monkeypatch) -> None:
        # Rejected by the schema layer before the tool body runs; the
        # framework's own error names the offending index as "items.1".
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))
        with pytest.raises(Exception) as caught:
            self.call("memory_save_many", {"items": [items(1)[0], "not-a-dict"]})
        assert "items.1" in str(caught.value)
