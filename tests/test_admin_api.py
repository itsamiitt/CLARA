from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

import pytest
import pytest_asyncio
pytest.importorskip(
    "fastapi",
    reason="fastapi not installed - run: pip install 'clara-memory[api]'",
)
pytest.importorskip(
    "httpx",
    reason="httpx not installed - run: pip install 'clara-memory[api]'",
)
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clara.agent import ClaraMemory
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine


FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        magnitude = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / magnitude for value in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _EmptyExtractor:
    def extract(self, text: str):
        return []


@pytest_asyncio.fixture
async def admin_client():
    from clara.api import create_app

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

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    agent = ClaraMemory(
        engine=engine,
        session_factory=factory,
        embedding_engine=EmbeddingEngine(_FakeBackend()),
        extractor=_EmptyExtractor(),  # type: ignore[arg-type]
        decay_scheduler=None,
        cache=MemoryCache("memory://"),
    )

    now = datetime.now(timezone.utc)
    async with factory() as session:
        session.add_all([
            Memory(
                memory_type=MemoryType.belief,
                user_id="alice",
                content={"subject": "user", "relation": "uses", "object": "Python"},
                embedding=None,
                confidence=0.2,
                status=MemoryStatus.active,
                decay_rate=0.02,
                created_at=now,
                updated_at=now,
                metadata_={},
            ),
            Memory(
                memory_type=MemoryType.belief,
                user_id="alice",
                content={"subject": "user", "relation": "uses", "object": "Rust"},
                embedding=None,
                confidence=0.9,
                status=MemoryStatus.superseded,
                decay_rate=0.02,
                created_at=now,
                updated_at=now,
                metadata_={"superseded_by": "new-memory-id"},
            ),
            Memory(
                memory_type=MemoryType.skill,
                user_id="alice",
                content={"subject": "user", "relation": "knows", "object": "Kubernetes"},
                embedding=None,
                confidence=0.8,
                status=MemoryStatus.active,
                decay_rate=0.01,
                created_at=now,
                updated_at=now,
                metadata_={"attempt_count": 10, "success_count": 9},
            ),
            Memory(
                memory_type=MemoryType.skill,
                user_id="alice",
                content={"subject": "user", "relation": "knows", "object": "Docker"},
                embedding=None,
                confidence=0.7,
                status=MemoryStatus.active,
                decay_rate=0.01,
                created_at=now,
                updated_at=now,
                metadata_={"attempt_count": 10, "success_count": 5},
            ),
        ])
        await session.commit()

    app = create_app(agent=agent)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as http:
            yield http

    await agent.close()


class TestAdminRoutes:
    @pytest.mark.asyncio
    async def test_stats_and_conflicts(self, admin_client):
        stats = await admin_client.get("/admin/stats")
        conflicts = await admin_client.get("/admin/conflicts")

        assert stats.status_code == 200
        assert any(item["memory_type"] == "belief" for item in stats.json()["counts"])

        assert conflicts.status_code == 200
        payload = conflicts.json()["conflicts"]
        assert len(payload) == 1
        assert payload[0]["superseded_by"] == "new-memory-id"

    @pytest.mark.asyncio
    async def test_decay_report_and_skill_leaderboard(self, admin_client):
        decay = await admin_client.get("/admin/decay-report")
        skills = await admin_client.get("/admin/skills/leaderboard")

        assert decay.status_code == 200
        assert decay.json()["beliefs"][0]["confidence"] == pytest.approx(0.2, abs=1e-6)

        assert skills.status_code == 200
        leaderboard = skills.json()["skills"]
        assert leaderboard[0]["content"]["object"] == "Kubernetes"
        assert leaderboard[0]["success_rate"] > leaderboard[1]["success_rate"]

    @pytest.mark.asyncio
    async def test_health(self, admin_client):
        response = await admin_client.get("/admin/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["cache"]["backend"] == "memory"


class TestAdminAuth:
    @pytest.mark.asyncio
    async def test_data_routes_require_header_when_auth_required(self):
        from clara.api import create_app
        from clara.config import ClaraConfig

        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        agent = ClaraMemory(
            engine=engine,
            session_factory=factory,
            embedding_engine=EmbeddingEngine(_FakeBackend()),
            extractor=_EmptyExtractor(),  # type: ignore[arg-type]
            decay_scheduler=None,
        )

        app = create_app(config=ClaraConfig(auth_required=True), agent=agent)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as http:
                # Data route blocked without the header...
                stats = await http.get("/admin/stats")
                # ...but liveness probe stays open.
                health = await http.get("/admin/health")

        assert stats.status_code == 401
        assert health.status_code == 200

        await agent.close()
