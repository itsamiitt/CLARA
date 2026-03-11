import asyncio
from unittest.mock import patch

from clara.agent import ClaraMemory
from clara.extraction.extractor import ExtractedFact
from clara.integrations import OpenClawMemoryBridge


async def main():
    agent = await ClaraMemory.create(
        db_url="sqlite+aiosqlite://",
        embedding_backend="local",
        llm_provider="openai",
        start_scheduler=False,
    )
    bridge = OpenClawMemoryBridge(agent)

    fake_facts = [
        ExtractedFact(
            subject="user",
            relation="prefers",
            object="aws ecs",
            domain="deployment",
            source_type="user_direct",
            confidence=0.92,
            is_negation=False,
            raw_text="I prefer AWS ECS for deployments.",
        ),
        ExtractedFact(
            subject="user",
            relation="asked",
            object="deploy payment service",
            domain="work",
            source_type="user_direct",
            confidence=0.88,
            is_negation=False,
            raw_text="Please help deploy payment service.",
        ),
    ]

    with patch.object(agent._extractor, "extract", return_value=fake_facts):
        await bridge.remember_turn(
            session_id="sess-001",
            role="user",
            text="I prefer AWS ECS for deployments.",
            metadata={"channel": "whatsapp"},
        )

    result = await bridge.recall_for(session_id="sess-001", query="deployment platform", top_k=5)
    context = await bridge.context_for(session_id="sess-001", query="deploy service", top_k=5)

    print("TOTAL_RETRIEVED", result.total)
    print("BELIEFS", len(result.beliefs), "EVENTS", len(result.events), "SKILLS", len(result.skills), "WORLD", len(result.world_model))
    print("--- CONTEXT PREVIEW ---")
    print(context[:1200])

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
