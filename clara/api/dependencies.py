"""Dependency helpers for the CLARA FastAPI app.

Authentication model: when ``auth_required`` is on, the caller presents
``Authorization: Bearer <token>`` and the user identity is derived from the
matching entry in ``config.api_tokens``. ``X-User-ID`` is *not* an identity —
it may only narrow a request to the token's own user — because a header the
client sets cannot authenticate the client.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from clara.agent import ClaraMemory

_UNAUTHENTICATED = {"WWW-Authenticate": "Bearer"}


def get_agent(request: Request) -> ClaraMemory:
    agent: ClaraMemory | None = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=500, detail="CLARA agent is not initialized.")
    return agent


async def get_session(
    agent: ClaraMemory = Depends(get_agent),
) -> AsyncIterator[AsyncSession]:
    async with agent.session_factory() as session:
        yield session


def _auth_required(config: object | None) -> bool:
    auth_required = getattr(config, "auth_required", None)
    if auth_required is None:
        raw = os.environ.get("CLARA_AUTH_REQUIRED", "").strip().lower()
        return bool(raw) and raw not in {"0", "false", "no", "off"}
    return bool(auth_required)


def _user_for_token(config: object | None, presented: str) -> str | None:
    """User whose token matches *presented*, compared in constant time.

    Every configured token is checked even after a match so the comparison
    count does not depend on which token was presented.
    """
    matched: str | None = None
    for user_id, token in getattr(config, "api_tokens", ()) or ():
        if secrets.compare_digest(token, presented):
            matched = user_id
    return matched


def get_current_user(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str | None:
    requested = x_user_id.strip() if x_user_id else None
    config = getattr(request.app.state, "config", None)

    if not _auth_required(config):
        # Local single-user mode: no credential, so X-User-ID is a scoping
        # hint, not an identity. create_app() warns when this mode is active.
        return requested or None

    scheme, _, presented = (authorization or "").partition(" ")
    presented = presented.strip()
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token. Send 'Authorization: Bearer <token>'.",
            headers=_UNAUTHENTICATED,
        )
    user_id = _user_for_token(config, presented)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token.",
            headers=_UNAUTHENTICATED,
        )
    if requested is not None and requested != user_id:
        raise HTTPException(
            status_code=403,
            detail="Authenticated user does not match requested user_id.",
        )
    return user_id


def resolve_user_scope(
    *,
    requested_user_id: str | None,
    current_user: str | None,
) -> str | None:
    if current_user is None:
        return requested_user_id
    if requested_user_id is not None and requested_user_id != current_user:
        raise HTTPException(
            status_code=403,
            detail="Authenticated user does not match requested user_id.",
        )
    return current_user
