from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clara.integrations import OpenClawMemoryBridge


@pytest.mark.asyncio
async def test_bridge_remember_turn_serializes_session_role_and_text():
    memory = SimpleNamespace(remember=AsyncMock(return_value=[{"action": "created"}]))
    bridge = OpenClawMemoryBridge(memory)

    out = await bridge.remember_turn(
        session_id="s-1",
        role="user",
        text="I prefer Python.",
        metadata={"channel": "whatsapp"},
    )

    assert out == [{"action": "created"}]
    payload = memory.remember.await_args.args[0]
    assert "session:s-1" in payload
    assert "role:user" in payload
    assert "text: I prefer Python." in payload
    assert "metadata: channel=whatsapp" in payload


@pytest.mark.asyncio
async def test_bridge_recall_and_context_are_scoped_to_session():
    memory = SimpleNamespace(
        recall=AsyncMock(return_value=SimpleNamespace(total=0)),
        context_for=AsyncMock(return_value="ctx"),
    )
    bridge = OpenClawMemoryBridge(memory)

    result = await bridge.recall_for(session_id="abc", query="favorite language")
    ctx = await bridge.context_for(session_id="abc", query="favorite language")

    assert result.total == 0
    assert ctx == "ctx"
    memory.recall.assert_awaited_once_with("session:abc favorite language", top_k=8)
    memory.context_for.assert_awaited_once_with("session:abc favorite language", top_k=8)
