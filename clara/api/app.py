"""FastAPI application factory for CLARA."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from clara.agent import ClaraMemory
from clara.api.routes_admin import router as admin_router
from clara.api.routes_interaction import router as interaction_router
from clara.api.routes_memory import router as memory_router
from clara.config import ClaraConfig


def create_app(
    config: ClaraConfig | None = None,
    *,
    agent: ClaraMemory | None = None,
) -> FastAPI:
    config = config or ClaraConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        managed_agent = agent
        if managed_agent is None:
            managed_agent = await ClaraMemory.create(
                db_url=config.db_url,
                embedding_backend=config.embedding_backend,
                llm_provider=config.llm_provider,
                start_scheduler=config.start_scheduler,
                cache_url=config.cache_url,
            )
        app.state.agent = managed_agent
        app.state.config = config
        try:
            yield
        finally:
            if agent is None and managed_agent is not None:
                await managed_agent.close()

    app = FastAPI(
        title="CLARA Memory API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(interaction_router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    return app
