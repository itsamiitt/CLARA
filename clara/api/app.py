"""FastAPI application factory for CLARA."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clara.agent import ClaraMemory
from clara.api.routes_admin import router as admin_router
from clara.api.routes_interaction import router as interaction_router
from clara.api.routes_memory import router as memory_router
from clara.config import ClaraConfig

logger = logging.getLogger(__name__)

# Warn at most once per process so test suites that build many apps stay quiet.
_AUTH_WARNED = False


def create_app(
    config: ClaraConfig | None = None,
    *,
    agent: ClaraMemory | None = None,
) -> FastAPI:
    config = config or ClaraConfig.from_env()

    global _AUTH_WARNED
    if not config.auth_required and not _AUTH_WARNED:
        _AUTH_WARNED = True
        logger.warning(
            "CLARA API is starting with auth_required=False: requests may act on "
            "any user_id without an X-User-ID header. Set CLARA_AUTH_REQUIRED=true "
            "(or put the API behind an authenticating gateway) before exposing it."
        )

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
                similarity_threshold=config.similarity_threshold,
                retrieval_top_k=config.retrieval_top_k,
                archival_threshold=config.archival_threshold,
                event_stale_days=config.event_stale_days,
                skill_unused_days=config.skill_unused_days,
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

    # Opt-in CORS: set CLARA_CORS_ORIGINS to a comma-separated origin list
    # (use "*" for any origin). Off by default to stay safe for local use.
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("CLARA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(interaction_router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    return app
