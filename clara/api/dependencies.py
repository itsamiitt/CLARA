"""Dependency helpers for the CLARA FastAPI app."""

from __future__ import annotations

import os
from typing import AsyncIterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from clara.agent import ClaraMemory


def get_agent(request: Request) -> ClaraMemory:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=500, detail="CLARA agent is not initialized.")
    return agent


async def get_session(
    agent: ClaraMemory = Depends(get_agent),
) -> AsyncIterator[AsyncSession]:
    async with agent._session_factory() as session:
        yield session


def get_current_user(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> str | None:
    user_id = x_user_id.strip() if x_user_id else None
    config = getattr(request.app.state, "config", None)
    auth_required = getattr(config, "auth_required", None)
    if auth_required is None:
        auth_required = os.environ.get("CLARA_AUTH_REQUIRED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if auth_required and not user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-ID header.")
    return user_id or None


def resolve_user_scope(
    *,
    requested_user_id: str | None,
    current_user: str | None,
) -> str | None:
    if current_user is None:
        return requested_user_id
    if requested_user_id is not None and requested_user_id != current_user:
        raise HTTPException(status_code=403, detail="Authenticated user does not match requested user_id.")
    return current_user
