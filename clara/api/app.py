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
    if config.auth_required and not config.api_tokens:
        # Fail closed. Without tokens the only "identity" would be a
        # client-supplied header, which is indistinguishable from no auth —
        # starting anyway would present security that does not exist.
        raise RuntimeError(
            "CLARA_AUTH_REQUIRED is set but no API tokens are configured. Set "
            "CLARA_API_TOKENS='user:token' (token at least 16 characters), or "
            "unset CLARA_AUTH_REQUIRED to run in local single-user mode."
        )
    if not config.auth_required and not _AUTH_WARNED:
        _AUTH_WARNED = True
        logger.warning(
            "CLARA API is starting with auth_required=False: any client that can "
            "reach the port may read and write every user's memories. Bind to "
            "localhost only. Set CLARA_AUTH_REQUIRED=true with CLARA_API_TOKENS "
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

    # Opt-in CORS: set CLARA_CORS_ORIGINS to a comma-separated origin list.
    # Off by default to stay safe for local use. "*" is accepted but then
    # credentials are refused: Starlette reflects the caller's origin when a
    # wildcard is combined with allow_credentials, which would hand every
    # website on the internet credentialed access to the memory store.
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("CLARA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        wildcard = "*" in cors_origins
        if wildcard and len(cors_origins) > 1:
            logger.warning(
                "CLARA_CORS_ORIGINS contains '*' alongside explicit origins; "
                "the wildcard makes the others redundant."
            )
        if wildcard:
            logger.warning(
                "CLARA_CORS_ORIGINS='*' allows any origin, so credentialed "
                "cross-origin requests are disabled. List explicit origins to "
                "allow credentials."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if wildcard else cors_origins,
            allow_credentials=not wildcard,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-User-ID"],
        )

    app.include_router(interaction_router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    return app
