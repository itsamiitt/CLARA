"""Administrative and dashboard-style API routes.

Every route here is scoped to the authenticated user, exactly like
``routes_memory``. These endpoints return full memory ``content``, so an
unscoped query would hand any authenticated caller every other user's
memories — they are no more privileged than ``/memory/search`` and must not
read as a cross-tenant view.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from clara.agent import ClaraMemory
from clara.api.dependencies import (
    get_agent,
    get_current_user,
    get_session,
    resolve_user_scope,
)
from clara.db.models import Memory, MemoryStatus, MemoryType

router = APIRouter()


def _scope_filters(current_user: str | None) -> list[ColumnElement[bool]]:
    """Row filter for the caller. Empty in local unauthenticated mode."""
    effective_user_id = resolve_user_scope(
        requested_user_id=None,
        current_user=current_user,
    )
    if effective_user_id is None:
        return []
    return [Memory.user_id == effective_user_id]


def _memory_payload(memory: Memory) -> dict[str, object]:
    return {
        "memory_id": str(memory.memory_id),
        "memory_type": memory.memory_type.value,
        "status": memory.status.value,
        "user_id": memory.user_id,
        "confidence": memory.confidence,
        "content": memory.content,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


@router.get("/admin/stats")
async def stats(
    session: AsyncSession = Depends(get_session),
    current_user: str | None = Depends(get_current_user),
) -> dict[str, object]:
    rows = (
        await session.execute(
            select(Memory.memory_type, Memory.status, func.count())
            .where(*_scope_filters(current_user))
            .group_by(Memory.memory_type, Memory.status)
            .order_by(Memory.memory_type, Memory.status)
        )
    ).all()
    return {
        "counts": [
            {"memory_type": memory_type.value, "status": status.value, "count": count}
            for memory_type, status, count in rows
        ]
    }


@router.get("/admin/conflicts")
async def conflicts(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    current_user: str | None = Depends(get_current_user),
) -> dict[str, object]:
    rows = (
        await session.execute(
            select(Memory)
            .where(Memory.status == MemoryStatus.superseded, *_scope_filters(current_user))
            .order_by(Memory.updated_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return {
        "conflicts": [
            {
                **_memory_payload(row),
                "superseded_by": (row.metadata_ or {}).get("superseded_by"),
            }
            for row in rows
        ]
    }


@router.get("/admin/decay-report")
async def decay_report(
    threshold: float = Query(0.25, gt=0.0, lt=1.0),
    limit: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
    current_user: str | None = Depends(get_current_user),
) -> dict[str, object]:
    rows = (
        await session.execute(
            select(Memory)
            .where(Memory.memory_type == MemoryType.belief)
            .where(Memory.status == MemoryStatus.active)
            .where(Memory.confidence < threshold)
            .where(*_scope_filters(current_user))
            .order_by(Memory.confidence.asc(), Memory.updated_at.asc())
            .limit(limit)
        )
    ).scalars().all()
    return {"beliefs": [_memory_payload(row) for row in rows], "threshold": threshold}


@router.get("/admin/skills/leaderboard")
async def skills_leaderboard(
    limit: int = Query(20, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    current_user: str | None = Depends(get_current_user),
) -> dict[str, object]:
    rows = (
        await session.execute(
            select(Memory)
            .where(Memory.memory_type == MemoryType.skill)
            .where(Memory.status == MemoryStatus.active)
            .where(*_scope_filters(current_user))
            .order_by(Memory.confidence.desc())
            .limit(limit * 10)
        )
    ).scalars().all()

    leaderboard = []
    for row in rows:
        meta = row.metadata_ or {}
        attempts = int(meta.get("attempt_count", 0))
        successes = int(meta.get("success_count", 0))
        success_rate = (successes / attempts) if attempts > 0 else 0.0
        leaderboard.append({
            **_memory_payload(row),
            "attempt_count": attempts,
            "success_count": successes,
            "success_rate": success_rate,
        })

    leaderboard.sort(
        key=lambda item: (item["success_rate"], item["attempt_count"], item["confidence"]),
        reverse=True,
    )
    return {"skills": leaderboard[:limit]}


@router.get("/admin/health")
async def health(
    agent: ClaraMemory = Depends(get_agent),
    session: AsyncSession = Depends(get_session),
    current_user: str | None = Depends(get_current_user),
) -> dict[str, object]:
    await session.execute(select(1))
    return {
        "status": "ok",
        "database": "ok",
        "scheduler_running": bool(getattr(agent, "_decay_scheduler", None)),
        "cache": await agent.cache_health(),
    }
