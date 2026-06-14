from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from clara.agent import ClaraMemory
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact


TOKEN_RE = re.compile(r"[a-z0-9_]+")
USE_RE = re.compile(
    r"^(?P<subject>[a-z0-9_]+) uses (?P<object>[a-z0-9_]+) for (?P<domain>[a-z0-9_]+)$",
    re.IGNORECASE,
)
NEGATED_USE_RE = re.compile(
    r"^(?P<subject>[a-z0-9_]+) no longer uses (?P<object>[a-z0-9_]+) for (?P<domain>[a-z0-9_]+)$",
    re.IGNORECASE,
)
DEPLOY_RE = re.compile(
    r"^(?P<subject>[a-z0-9_]+) deployed (?P<object>[a-z0-9_]+)$",
    re.IGNORECASE,
)
KNOWS_RE = re.compile(
    r"^(?P<subject>[a-z0-9_]+) knows (?P<object>[a-z0-9_]+)$",
    re.IGNORECASE,
)
HAS_NODES_RE = re.compile(
    r"^(?P<subject>[a-z0-9_]+) has (?P<count>[0-9]+) nodes$",
    re.IGNORECASE,
)

BASE_USERS = 60
DUPLICATE_USERS = 12
CONFLICT_USERS = 12
DOMAIN_USERS = 12
NEGATION_USERS = 12
CONCURRENT_QUERY_USERS = 30
MAX_CONCURRENT_RECALLS = 10


def _canonical_token(token: str) -> str:
    mapping = {
        "uses": "use",
        "using": "use",
        "deployed": "deploy",
        "deploys": "deploy",
        "deployment": "deploy",
        "knows": "know",
        "nodes": "node",
        "systems": "systems",
    }
    return mapping.get(token, token)


class _SemanticFakeBackend:
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = [_canonical_token(t) for t in TOKEN_RE.findall(text.lower())]
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

        magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / magnitude for v in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _StressExtractor:
    async def extract(self, text: str) -> list[ExtractedFact]:
        cleaned = text.strip().lower()
        match = NEGATED_USE_RE.match(cleaned)
        if match:
            return [ExtractedFact(
                subject=match.group("subject"),
                relation="uses",
                object=match.group("object"),
                domain=match.group("domain"),
                source_type="user_direct",
                confidence=0.96,
                is_negation=True,
                raw_text=text,
            )]

        match = USE_RE.match(cleaned)
        if match:
            return [ExtractedFact(
                subject=match.group("subject"),
                relation="uses",
                object=match.group("object"),
                domain=match.group("domain"),
                source_type="user_direct",
                confidence=0.94,
                is_negation=False,
                raw_text=text,
            )]

        match = DEPLOY_RE.match(cleaned)
        if match:
            return [ExtractedFact(
                subject=match.group("subject"),
                relation="deployed",
                object=match.group("object"),
                domain=None,
                source_type="user_direct",
                confidence=0.9,
                is_negation=False,
                raw_text=text,
            )]

        match = KNOWS_RE.match(cleaned)
        if match:
            return [ExtractedFact(
                subject=match.group("subject"),
                relation="knows",
                object=match.group("object"),
                domain=None,
                source_type="user_direct",
                confidence=0.91,
                is_negation=False,
                raw_text=text,
            )]

        match = HAS_NODES_RE.match(cleaned)
        if match:
            return [ExtractedFact(
                subject=match.group("subject"),
                relation="has",
                object=f"{match.group('count')} nodes",
                domain=None,
                source_type="tool_api",
                confidence=0.93,
                is_negation=False,
                raw_text=text,
            )]

        return []


def _db_url_for() -> tuple[str, Path]:
    results_dir = Path.cwd() / "test-results"
    results_dir.mkdir(exist_ok=True)
    db_path = results_dir / f"stress-agent-{uuid.uuid4().hex}.sqlite3"
    return f"sqlite+aiosqlite:///{db_path.as_posix()}", db_path


def _build_workload() -> list[str]:
    payloads: list[str] = []

    for idx in range(BASE_USERS):
        payloads.extend([
            f"user_{idx:03d} uses rust for systems",
            f"user_{idx:03d} deployed service_{idx:03d}",
            f"user_{idx:03d} knows kubernetes",
            f"cluster_{idx:03d} has {idx % 5 + 1} nodes",
        ])

    for idx in range(DUPLICATE_USERS):
        text = f"user_dup_{idx:03d} uses rust for systems"
        payloads.extend([text, text])

    for idx in range(CONFLICT_USERS):
        payloads.extend([
            f"user_conflict_{idx:03d} uses rust for systems",
            f"user_conflict_{idx:03d} uses go for systems",
        ])

    for idx in range(DOMAIN_USERS):
        payloads.extend([
            f"user_domain_{idx:03d} uses rust for systems",
            f"user_domain_{idx:03d} uses python for web",
        ])

    for idx in range(NEGATION_USERS):
        payloads.extend([
            f"user_neg_{idx:03d} uses python for web",
            f"user_neg_{idx:03d} no longer uses python for web",
        ])

    return payloads


def _has_match(
    result,
    *,
    group_name: str,
    subject: str,
    relation: str,
    object_: str,
    is_negation: bool | None = None,
    domain: str | None = None,
) -> bool:
    group = getattr(result, group_name)
    for scored in group:
        content = scored.memory.content
        if not isinstance(content, dict):
            continue
        if content.get("subject") != subject:
            continue
        if content.get("relation") != relation:
            continue
        if content.get("object") != object_:
            continue
        if is_negation is not None and bool(content.get("is_negation", False)) != is_negation:
            continue
        if domain is not None and content.get("domain") != domain:
            continue
        return True
    return False


async def _recall_guarded(
    agent: ClaraMemory,
    query: str,
    *,
    top_k: int,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        return await agent.recall(query, top_k=top_k)


@pytest.mark.stress
@pytest.mark.asyncio
async def test_bulk_agent_round_trip_with_heavy_retrievals():
    workload = _build_workload()
    db_url, db_path = _db_url_for()
    recall_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RECALLS)

    with patch("clara.agent._create_backend", return_value=_SemanticFakeBackend()):
        agent = await ClaraMemory.create(
            db_url=db_url,
            embedding_backend="openai",
            llm_provider="openai",
            start_scheduler=False,
        )

    agent._extractor = _StressExtractor()

    try:
        action_counts: Counter[str] = Counter()
        for text in workload:
            result = await agent.remember(text)
            assert len(result) == 1
            action_counts[result[0]["action"]] += 1

        assert action_counts == Counter({
            "created": 288,
            "superseded": 24,
            "retained_both": 12,
            "reinforced": 12,
        })

        async with agent._session_factory() as session:
            total_rows = await session.scalar(select(func.count()).select_from(Memory))
            active_rows = await session.scalar(
                select(func.count())
                .select_from(Memory)
                .where(Memory.status == MemoryStatus.active)
            )
            superseded_rows = await session.scalar(
                select(func.count())
                .select_from(Memory)
                .where(Memory.status == MemoryStatus.superseded)
            )
            missing_embeddings = await session.scalar(
                select(func.count())
                .select_from(Memory)
                .where(Memory.embedding.is_(None))
            )

        assert total_rows == 324
        assert active_rows == 300
        assert superseded_rows == 24
        assert missing_embeddings == 0

        for idx in range(BASE_USERS):
            result = await agent.recall(
                f"user_{idx:03d} uses rust for systems",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="beliefs",
                subject=f"user_{idx:03d}",
                relation="uses",
                object_="rust",
                domain="systems",
                is_negation=False,
            )

            result = await agent.recall(
                f"user_{idx:03d} deployed service_{idx:03d}",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="events",
                subject=f"user_{idx:03d}",
                relation="deployed",
                object_=f"service_{idx:03d}",
            )

            result = await agent.recall(
                f"user_{idx:03d} knows kubernetes",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="skills",
                subject=f"user_{idx:03d}",
                relation="knows",
                object_="kubernetes",
            )

            result = await agent.recall(
                f"cluster_{idx:03d} has {idx % 5 + 1} nodes",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="world_model",
                subject=f"cluster_{idx:03d}",
                relation="has",
                object_=f"{idx % 5 + 1} nodes",
            )

        for idx in range(CONFLICT_USERS):
            result = await agent.recall(
                f"user_conflict_{idx:03d} uses go for systems",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="beliefs",
                subject=f"user_conflict_{idx:03d}",
                relation="uses",
                object_="go",
                domain="systems",
                is_negation=False,
            )
            assert not _has_match(
                result,
                group_name="beliefs",
                subject=f"user_conflict_{idx:03d}",
                relation="uses",
                object_="rust",
                domain="systems",
                is_negation=False,
            )

        for idx in range(DOMAIN_USERS):
            systems_result = await agent.recall(
                f"user_domain_{idx:03d} uses rust for systems",
                top_k=6,
            )
            web_result = await agent.recall(
                f"user_domain_{idx:03d} uses python for web",
                top_k=6,
            )
            assert _has_match(
                systems_result,
                group_name="beliefs",
                subject=f"user_domain_{idx:03d}",
                relation="uses",
                object_="rust",
                domain="systems",
                is_negation=False,
            )
            assert _has_match(
                web_result,
                group_name="beliefs",
                subject=f"user_domain_{idx:03d}",
                relation="uses",
                object_="python",
                domain="web",
                is_negation=False,
            )

        for idx in range(NEGATION_USERS):
            result = await agent.recall(
                f"user_neg_{idx:03d} no longer uses python for web",
                top_k=6,
            )
            assert _has_match(
                result,
                group_name="beliefs",
                subject=f"user_neg_{idx:03d}",
                relation="uses",
                object_="python",
                domain="web",
                is_negation=True,
            )
            assert not _has_match(
                result,
                group_name="beliefs",
                subject=f"user_neg_{idx:03d}",
                relation="uses",
                object_="python",
                domain="web",
                is_negation=False,
            )

        concurrent_queries = [
            f"user_{idx:03d} uses rust for systems" for idx in range(CONCURRENT_QUERY_USERS)
        ] + [
            f"user_{idx:03d} deployed service_{idx:03d}" for idx in range(CONCURRENT_QUERY_USERS)
        ] + [
            f"user_{idx:03d} knows kubernetes" for idx in range(CONCURRENT_QUERY_USERS)
        ] + [
            f"cluster_{idx:03d} has {idx % 5 + 1} nodes" for idx in range(CONCURRENT_QUERY_USERS)
        ]

        concurrent_results = await asyncio.gather(*(
            _recall_guarded(
                agent,
                query,
                top_k=6,
                semaphore=recall_semaphore,
            )
            for query in concurrent_queries
        ))
        assert len(concurrent_results) == CONCURRENT_QUERY_USERS * 4
        assert all(result.total >= 1 for result in concurrent_results)

        context = await agent.context_for(
            "user_042 uses rust for systems and user_042 deployed service_042 and "
            "user_042 knows kubernetes and cluster_042 has 3 nodes",
            top_k=12,
        )
        assert "=== MEMORY CONTEXT ===" in context
        assert "user_042 uses rust" in context
        assert "user_042 deployed service_042" in context
        assert "kubernetes" in context
        assert "cluster_042" in context

        async with agent._session_factory() as session:
            belief = await session.scalar(
                select(Memory)
                .where(Memory.status == MemoryStatus.active)
                .where(Memory.memory_type == MemoryType.belief)
                .where(Memory.content["subject"].as_string() == "user_042")
                .where(Memory.content["object"].as_string() == "rust")
            )

        assert belief is not None
        assert int((belief.metadata_ or {}).get("access_count", 0)) > 0
    finally:
        await agent.close()
        for suffix in ("", "-shm", "-wal"):
            candidate = Path(f"{db_path}{suffix}")
            try:
                if candidate.exists():
                    candidate.unlink()
            except PermissionError:
                pass
