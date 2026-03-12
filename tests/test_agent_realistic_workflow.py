from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from clara.agent import ClaraMemory
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact
from clara.reasoning.engine import ReasoningEngine


TOKEN_RE = re.compile(r"[a-z0-9_+-]+")
USE_RE = re.compile(
    r"^(?P<subject>[a-z][a-z0-9_-]*) uses (?P<object>[a-z0-9_+-]+) for (?P<domain>[a-z0-9 _-]+)$",
    re.IGNORECASE,
)
NEGATED_USE_RE = re.compile(
    r"^(?P<subject>[a-z][a-z0-9_-]*) no longer uses (?P<object>[a-z0-9_+-]+) for (?P<domain>[a-z0-9 _-]+)$",
    re.IGNORECASE,
)
DEPLOY_RE = re.compile(
    r"^(?:yesterday )?(?P<subject>[a-z][a-z0-9_-]*) deployed (?:the )?(?P<object>[a-z0-9_ -]+?)(?: to [a-z0-9_ -]+)?$",
    re.IGNORECASE,
)
KNOWS_RE = re.compile(
    r"^(?P<subject>[a-z][a-z0-9_-]*) knows (?P<objects>[a-z0-9_+,\- ]+)$",
    re.IGNORECASE,
)
HAS_NODES_RE = re.compile(
    r"^(?:the )?(?P<subject>[a-z0-9_-]+(?: [a-z0-9_-]+)*) has (?P<count>[0-9]+) nodes$",
    re.IGNORECASE,
)

REALISTIC_TURNS = [
    ("alice", "Alicia uses Rust for payments systems."),
    ("alice", "Alicia knows Kubernetes and Terraform."),
    ("alice", "Yesterday Alicia deployed the payments-api service to production."),
    ("alice", "The payments cluster has 6 nodes."),
    ("alice", "Alicia uses Python for internal tooling."),
    ("alice", "Alicia no longer uses Python for internal tooling."),
    ("bob", "Ben uses Go for backend services."),
    ("bob", "Ben knows Docker."),
    ("bob", "Ben deployed the inventory-api service."),
    ("bob", "The inventory cluster has 4 nodes."),
    ("carol", "Carol uses Java for billing systems."),
    ("carol", "Carol knows Kafka."),
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(".!?")).lower()


def _canonical_token(token: str) -> str:
    mapping = {
        "uses": "use",
        "using": "use",
        "deployed": "deploy",
        "deploys": "deploy",
        "deployment": "deploy",
        "knows": "know",
        "nodes": "node",
        "systems": "system",
        "services": "service",
        "payments": "payment",
    }
    return mapping.get(token, token)


class _SemanticConversationBackend:
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = [_canonical_token(token) for token in TOKEN_RE.findall(text.lower())]
        if not tokens:
            return vector

        for token in tokens:
            for salt in ("a", "b"):
                digest = hashlib.blake2b(
                    f"{salt}:{token}".encode("utf-8"),
                    digest_size=8,
                ).digest()
                idx = int.from_bytes(digest[:4], "big") % self._dimensions
                vector[idx] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _RealisticConversationExtractor:
    def extract(self, text: str) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for sentence in self._sentences(text):
            match = NEGATED_USE_RE.match(sentence)
            if match:
                facts.append(self._fact(
                    sentence,
                    subject=match.group("subject"),
                    relation="uses",
                    object_=match.group("object"),
                    domain=match.group("domain"),
                    is_negation=True,
                ))
                continue

            match = USE_RE.match(sentence)
            if match:
                facts.append(self._fact(
                    sentence,
                    subject=match.group("subject"),
                    relation="uses",
                    object_=match.group("object"),
                    domain=match.group("domain"),
                ))
                continue

            match = DEPLOY_RE.match(sentence)
            if match:
                facts.append(self._fact(
                    sentence,
                    subject=match.group("subject"),
                    relation="deployed",
                    object_=match.group("object"),
                    domain=None,
                ))
                continue

            match = KNOWS_RE.match(sentence)
            if match:
                for item in self._split_list(match.group("objects")):
                    facts.append(self._fact(
                        sentence,
                        subject=match.group("subject"),
                        relation="knows",
                        object_=item,
                        domain=None,
                    ))
                continue

            match = HAS_NODES_RE.match(sentence)
            if match:
                facts.append(self._fact(
                    sentence,
                    subject=match.group("subject"),
                    relation="has",
                    object_=f"{match.group('count')} nodes",
                    domain=None,
                    source_type="tool_api",
                ))

        return facts

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [
            _normalize_text(sentence)
            for sentence in re.split(r"[.!?]+", text)
            if sentence.strip()
        ]

    @staticmethod
    def _split_list(items: str) -> Iterable[str]:
        for item in re.split(r",| and ", items):
            cleaned = _normalize_text(item)
            if cleaned:
                yield cleaned

    @staticmethod
    def _fact(
        raw_text: str,
        *,
        subject: str,
        relation: str,
        object_: str,
        domain: str | None,
        is_negation: bool = False,
        source_type: str = "user_direct",
    ) -> ExtractedFact:
        return ExtractedFact(
            subject=_normalize_text(subject),
            relation=relation,
            object=_normalize_text(object_),
            domain=_normalize_text(domain) if domain is not None else None,
            source_type=source_type,
            confidence=0.94,
            is_negation=is_negation,
            raw_text=raw_text,
        )


def _has_match(
    result,
    *,
    group_name: str,
    subject: str,
    relation: str,
    object_: str,
    domain: str | None = None,
    is_negation: bool | None = None,
) -> bool:
    for scored in getattr(result, group_name):
        content = scored.memory.content if isinstance(scored.memory.content, dict) else {}
        if _normalize_text(content.get("subject", "")) != _normalize_text(subject):
            continue
        if _normalize_text(content.get("relation", "")) != _normalize_text(relation):
            continue
        if _normalize_text(content.get("object", "")) != _normalize_text(object_):
            continue
        if domain is not None and _normalize_text(content.get("domain", "")) != _normalize_text(domain):
            continue
        if is_negation is not None and bool(content.get("is_negation", False)) != is_negation:
            continue
        return True
    return False


@pytest_asyncio.fixture
async def realistic_agent() -> ClaraMemory:
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(type_, compiler, **kw):
        return "TEXT"

    try:
        from pgvector.sqlalchemy import Vector

        @compiles(Vector, "sqlite")
        def _compile_vector_sqlite(type_, compiler, **kw):
            return "TEXT"
    except Exception:
        pass

    with patch("clara.agent._create_backend", return_value=_SemanticConversationBackend()):
        agent = await ClaraMemory.create(
            db_url="sqlite+aiosqlite://",
            embedding_backend="openai",
            llm_provider="openai",
            start_scheduler=False,
        )

    agent._extractor = _RealisticConversationExtractor()  # type: ignore[assignment]
    yield agent
    await agent.close()


class TestRealisticAgentWorkflow:
    @pytest.mark.asyncio
    async def test_multi_turn_realistic_bulk_workflow(self, realistic_agent: ClaraMemory):
        action_counts: Counter[str] = Counter()

        for user_id, message in REALISTIC_TURNS:
            results = await realistic_agent.remember(message, user_id=user_id)
            assert results
            for item in results:
                action_counts[item["action"]] += 1

        assert action_counts["created"] >= 10
        assert action_counts["superseded"] == 1

        rust_result = await realistic_agent.recall(
            "Which language does Alicia use for payments systems?",
            top_k=6,
            user_id="alice",
        )
        assert _has_match(
            rust_result,
            group_name="beliefs",
            subject="alicia",
            relation="uses",
            object_="rust",
            domain="payments systems",
            is_negation=False,
        )
        assert all(scored.memory.user_id == "alice" for scored in rust_result.all)

        python_result = await realistic_agent.recall(
            "Does Alicia still use Python for internal tooling?",
            top_k=6,
            user_id="alice",
        )
        assert _has_match(
            python_result,
            group_name="beliefs",
            subject="alicia",
            relation="uses",
            object_="python",
            domain="internal tooling",
            is_negation=True,
        )
        assert not _has_match(
            python_result,
            group_name="beliefs",
            subject="alicia",
            relation="uses",
            object_="python",
            domain="internal tooling",
            is_negation=False,
        )

        cluster_result = await realistic_agent.recall(
            "How many nodes are in the payments cluster?",
            top_k=6,
            user_id="alice",
        )
        assert _has_match(
            cluster_result,
            group_name="world_model",
            subject="payments cluster",
            relation="has",
            object_="6 nodes",
        )

        context = await realistic_agent.context_for(
            "Summarize Alicia's payments stack, deployment work, and infrastructure.",
            top_k=8,
            user_id="alice",
        )
        context_lower = context.lower()
        assert "=== memory context ===" in context_lower
        assert "alicia uses rust" in context_lower
        assert "payments-api service" in context_lower
        assert "kubernetes" in context_lower
        assert "payments cluster" in context_lower
        assert "ben uses go" not in context_lower

        async def _generate_response(
            self,
            query: str,
            memory_context: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            lowered = memory_context.lower()
            assert "alicia uses rust" in lowered
            assert "payments cluster" in lowered
            return "Alicia uses Rust for payments systems."

        with patch.object(ReasoningEngine, "_generate_response", _generate_response):
            reply = await realistic_agent.interact(
                "Give me Alicia's current payments stack.",
                user_id="alice",
                top_k=8,
            )

        assert reply["response"] == "Alicia uses Rust for payments systems."
        assert reply["memories_used"]
        assert "alicia uses rust" in reply["memory_context"].lower()
        assert len(reply["facts_stored"]) == 1
        assert reply["facts_stored"][0]["action"] == "reinforced"

        async with realistic_agent._session_factory() as session:
            rows = (
                await session.execute(select(Memory).where(Memory.user_id == "alice"))
            ).scalars().all()

        active_counts = Counter(
            row.memory_type for row in rows if row.status == MemoryStatus.active
        )
        superseded_rows = [row for row in rows if row.status == MemoryStatus.superseded]

        assert active_counts == Counter({
            MemoryType.belief: 2,
            MemoryType.skill: 2,
            MemoryType.event: 1,
            MemoryType.world_model: 1,
        })
        assert len(superseded_rows) == 1
        assert _normalize_text(superseded_rows[0].content.get("object", "")) == "python"
