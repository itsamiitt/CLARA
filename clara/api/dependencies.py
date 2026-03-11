"""Dependency helpers for the CLARA FastAPI app."""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Depends, HTTPException, Request
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
