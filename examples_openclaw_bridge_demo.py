"""Quick demo: store/retrieve past memory through OpenClawMemoryBridge."""

from __future__ import annotations

import asyncio

from clara.agent import ClaraMemory
from clara.integrations import OpenClawMemoryBridge


async def main() -> None:
    agent = await ClaraMemory.create(
        db_url="sqlite+aiosqlite://",
        embedding_backend="local",
        llm_provider="openai",
        start_scheduler=False,
    )
    bridge = OpenClawMemoryBridge(agent)

    await bridge.remember_turn(
        session_id="demo-session",
        role="user",
        text="My favorite deployment target is AWS ECS.",
        metadata={"channel": "whatsapp"},
    )

    result = await bridge.recall_for(
        session_id="demo-session",
        query="deployment target",
        top_k=5,
    )
    print("retrieved:", result.total)
    print(await bridge.context_for(session_id="demo-session", query="deployment target", top_k=5))

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
