"""Memory query routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clara.agent import ClaraMemory
from clara.api.dependencies import (
    get_agent,
    get_current_user,
    get_session,
    resolve_user_scope,
)
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.memory.belief import BeliefMemory
from clara.retrieval.engine import RetrievalResult, ScoredMemory

router = APIRouter()


def _memory_summary(memory: Memory) -> dict[str, object]:
    return {
        "memory_id": str(memory.memory_id),
        "memory_type": memory.memory_type.value,
        "content": memory.content,
        "confidence": memory.confidence,
        "status": memory.status.value,
        "user_id": memory.user_id,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


def _scored_memory(sm: ScoredMemory) -> dict[str, object]:
    return {
        **_memory_summary(sm.memory),
        "score": sm.score,
        "similarity": sm.similarity,
        "recency_score": sm.recency_score,
        "usage_frequency": sm.usage_frequency,
    }


def _retrieval_payload(result: RetrievalResult) -> dict[str, object]:
    return {
        "total": result.total,
        "beliefs": [_scored_memory(sm) for sm in result.beliefs],
        "events": [_scored_memory(sm) for sm in result.events],
        "skills": [_scored_memory(sm) for sm in result.skills],
        "world_model": [_scored_memory(sm) for sm in result.world_model],
    }


@router.get("/memory/search")
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(8, ge=1, le=100),
    user_id: str | None = None,
    current_user: str | None = Depends(get_current_user),
    agent: ClaraMemory = Depends(get_agent),
) -> dict[str, object]:
    effective_user_id = resolve_user_scope(
        requested_user_id=user_id,
        current_user=current_user,
    )
    result = await agent.recall(q, top_k=top_k, user_id=effective_user_id)
    return _retrieval_payload(result)


@router.get("/memory/timeline")
async def timeline(
    user_id: str | None = None,
    subject: str | None = None,
    limit: int = Query(20, ge=1, le=200),
    current_user: str | None = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    effective_user_id = resolve_user_scope(
        requested_user_id=user_id,
        current_user=current_user,
    )
    filters = [
        Memory.memory_type == MemoryType.event,
        Memory.status == MemoryStatus.active,
    ]
    if effective_user_id is not None:
        filters.append(Memory.user_id == effective_user_id)
    if subject is not None:
        filters.append(Memory.content["subject"].as_string() == subject)

    rows = (
        await session.execute(
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    return {"events": [_memory_summary(row) for row in rows]}


@router.get("/memory/beliefs")
async def beliefs(
    user_id: str | None = None,
    subject: str | None = None,
    relation: str | None = None,
    domain: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    current_user: str | None = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    effective_user_id = resolve_user_scope(
        requested_user_id=user_id,
        current_user=current_user,
    )
    rows = await BeliefMemory(session).get_active_beliefs(
        subject=subject,
        relation=relation,
        domain=domain,
        user_id=effective_user_id,
        limit=limit,
    )
    return {"beliefs": [_memory_summary(row) for row in rows]}


@router.get("/memory/{memory_id}")
async def get_memory(
    memory_id: UUID,
    user_id: str | None = None,
    current_user: str | None = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    effective_user_id = resolve_user_scope(
        requested_user_id=user_id,
        current_user=current_user,
    )
    filters = [Memory.memory_id == memory_id]
    if effective_user_id is not None:
        filters.append(Memory.user_id == effective_user_id)

    row = await session.scalar(select(Memory).where(and_(*filters)))
    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return _memory_summary(row)
