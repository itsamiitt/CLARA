from __future__ import annotations

import hashlib
import math
from unittest.mock import AsyncMock

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
from clara.config import ClaraConfig
from clara.db.models import Base, Memory
from clara.extraction.extractor import ExtractedFact
from clara.retrieval.embeddings import EmbeddingEngine

FAKE_DIM = 8


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return FAKE_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=FAKE_DIM).digest()
        raw = [byte / 255.0 for byte in digest]
        mag = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / mag for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _FakeExtractor:
    async def extract(self, text: str) -> list[ExtractedFact]:
        text_lower = text.lower()
        facts: list[ExtractedFact] = []
        if "rust" in text_lower and "systems" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="uses",
                object="Rust",
                domain="systems",
                source_type="user_direct",
                confidence=0.9,
                is_negation=False,
                raw_text=text,
            ))
        if "deploy" in text_lower:
            facts.append(ExtractedFact(
                subject="user",
                relation="deployed",
                object="service",
                domain=None,
                source_type="user_direct",
                confidence=0.85,
                is_negation=False,
                raw_text=text,
            ))
        return facts


@pytest_asyncio.fixture
async def api_agent() -> ClaraMemory:

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    agent = ClaraMemory(
        engine=engine,
        session_factory=factory,
        embedding_engine=EmbeddingEngine(_FakeBackend()),
        extractor=_FakeExtractor(),  # type: ignore[arg-type]
        decay_scheduler=None,
    )
    yield agent
    await agent.close()


@pytest_asyncio.fixture
async def client(api_agent: ClaraMemory):
    from clara.api import create_app

    app = create_app(agent=api_agent)
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as http:
        yield http, api_agent


class TestInteractionRoutes:
    @pytest.mark.asyncio
    async def test_memory_learn_route(self, client):
        http, _agent = client

        response = await http.post("/memory/learn", json={
            "text": "I use Rust for systems work.",
            "user_id": "alice",
        })

        assert response.status_code == 200
        payload = response.json()
        assert payload["stored"] == 1
        assert payload["results"][0]["action"] == "created"

    @pytest.mark.asyncio
    async def test_interact_route(self, client):
        http, agent = client
        agent.interact = AsyncMock(return_value={
            "response": "Rust is the user's systems language.",
            "memory_context": "=== MEMORY CONTEXT ===",
            "facts_stored": [],
            "memories_used": [],
        })

        response = await http.post("/interact", json={
            "message": "What language does the user use?",
            "user_id": "alice",
        })

        assert response.status_code == 200
        payload = response.json()
        assert "Rust" in payload["response"]
        assert agent.interact.await_args.kwargs["user_id"] == "alice"


class TestMemoryRoutes:
    @pytest.mark.asyncio
    async def test_search_route(self, client):
        http, agent = client
        await agent.remember("I use Rust for systems work.", user_id="alice")
        agent._lance_engine.flush_pending_sync()  # search no longer flushes itself

        response = await http.get("/memory/search", params={
            "q": "What language for systems?",
            "user_id": "alice",
        })

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] >= 1
        assert all(item["user_id"] == "alice" for item in payload["beliefs"])

    @pytest.mark.asyncio
    async def test_get_memory_and_beliefs_routes(self, client):
        http, agent = client
        await agent.remember("I use Rust for systems work.", user_id="alice")

        async with agent._session_factory() as session:
            memory = (await session.execute(
                Memory.__table__.select().where(Memory.user_id == "alice")
            )).mappings().first()

        response = await http.get(f"/memory/{memory['memory_id']}", params={"user_id": "alice"})
        assert response.status_code == 200
        assert response.json()["user_id"] == "alice"

        beliefs = await http.get("/memory/beliefs", params={"user_id": "alice", "subject": "user"})
        assert beliefs.status_code == 200
        assert len(beliefs.json()["beliefs"]) == 1

    @pytest.mark.asyncio
    async def test_timeline_route(self, client):
        http, agent = client
        await agent.remember("I deployed the service.", user_id="alice")

        response = await http.get("/memory/timeline", params={"user_id": "alice"})

        assert response.status_code == 200
        payload = response.json()
        assert len(payload["events"]) == 1
        assert payload["events"][0]["memory_type"] == "event"

    @pytest.mark.asyncio
    async def test_search_uses_authenticated_header_scope(self, client):
        http, agent = client
        await agent.remember("I use Rust for systems work.", user_id="alice")
        await agent.remember("I use Rust for systems work.", user_id="bob")
        agent._lance_engine.flush_pending_sync()  # search no longer flushes itself

        response = await http.get(
            "/memory/search",
            params={"q": "systems language"},
            headers={"X-User-ID": "alice"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] >= 1
        assert all(item["user_id"] == "alice" for item in payload["beliefs"])

    @pytest.mark.asyncio
    async def test_search_rejects_header_and_query_user_mismatch(self, client):
        http, agent = client
        await agent.remember("I use Rust for systems work.", user_id="alice")

        response = await http.get(
            "/memory/search",
            params={"q": "systems language", "user_id": "bob"},
            headers={"X-User-ID": "alice"},
        )

        assert response.status_code == 403


class TestAuthenticatedApi:
    @pytest.mark.asyncio
    async def test_auth_required_rejects_missing_header(self, api_agent: ClaraMemory):
        from clara.api import create_app

        app = create_app(
            config=ClaraConfig(auth_required=True),
            agent=api_agent,
        )
        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as http:
            response = await http.get("/memory/search", params={"q": "Rust"})

        assert response.status_code == 401


class TestCors:
    @pytest.mark.asyncio
    async def test_cors_header_emitted_when_origin_configured(
        self, api_agent: ClaraMemory, monkeypatch
    ):
        monkeypatch.setenv("CLARA_CORS_ORIGINS", "https://example.com")
        from clara.api import create_app

        app = create_app(agent=api_agent)
        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as http:
            response = await http.get(
                "/memory/search",
                params={"q": "Rust", "user_id": "alice"},
                headers={"Origin": "https://example.com"},
            )

        assert response.headers.get("access-control-allow-origin") == "https://example.com"
