"""Interaction-oriented API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from clara.agent import ClaraMemory
from clara.api.dependencies import get_agent

router = APIRouter()


class InteractRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str | None = None
    system_prompt: str | None = None
    top_k: int = 8


class LearnRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str | None = None


@router.post("/interact")
async def interact(
    payload: InteractRequest,
    agent: ClaraMemory = Depends(get_agent),
) -> dict[str, Any]:
    try:
        return await agent.interact(
            payload.message,
            user_id=payload.user_id,
            system_prompt=payload.system_prompt,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/memory/learn")
async def learn(
    payload: LearnRequest,
    agent: ClaraMemory = Depends(get_agent),
) -> dict[str, Any]:
    results = await agent.remember(payload.text, user_id=payload.user_id)
    return {"stored": len(results), "results": results}
