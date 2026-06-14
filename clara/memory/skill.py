"""
CLARA — Skill Memory Store

Manages procedural skill memories: creation, outcome feedback (success/failure),
trigger-condition matching, and auto-deprecation.

Skills adjust confidence based on execution outcomes:
    - Success → Bayesian reinforcement (confidence ↑)
    - Failure → confidence × 0.85 penalty

Skills are auto-deprecated when confidence drops below
``DEPRECATION_THRESHOLD`` (0.15).

Usage::

    async with async_session() as session:
        skills = SkillStore(session)
        record = await skills.create(
            name="deploy-api",
            trigger_conditions=["deploy", "push to prod"],
            steps=["build image", "push to registry", "run kubectl apply"],
            user_id="alice",
        )
        await skills.record_outcome(record.memory_id, SkillOutcome.success)
        await session.commit()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clara.core.enums import SkillOutcome
from clara.core.exceptions import MemoryNotFoundError
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import LanceRetrievalEngine, RetrievalEngine
from clara.retrieval.lexical import LexicalRetriever

logger = logging.getLogger(__name__)

# Skills use the standard volatile decay rate
SKILL_DECAY_RATE = 0.02

# Deprecation threshold — below this, skills are auto-deprecated
DEPRECATION_THRESHOLD = 0.15

# Outcome adjustment constants
SUCCESS_REINFORCEMENT = 0.10   # additive boost on success
FAILURE_PENALTY_FACTOR = 0.85  # multiplicative penalty on failure


class SkillStore:
    """Async interface for creating, tracking outcomes, and querying skills.

    Each instance is bound to an ``AsyncSession``.  The caller is responsible
    for committing or rolling back the session.
    """

    def __init__(
        self,
        session: AsyncSession,
        retrieval_engine: RetrievalEngine | None = None,
    ) -> None:
        self._session = session
        self._retrieval_engine = retrieval_engine

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        name: str,
        trigger_conditions: list[str] | None = None,
        steps: list[str] | None = None,
        description: str = "",
        domain: str | None = None,
        user_id: str | None = None,
        confidence: float = 0.5,
        source_type: str = "user_direct",
        raw_text: str | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Create a new skill memory.

        Args:
            name:                 Human-readable skill name.
            trigger_conditions:   Keywords/phrases that trigger this skill.
            steps:                Ordered list of procedure steps.
            description:          Longer description of what the skill does.
            domain:               Optional domain tag.
            user_id:              Tenant partition key.
            confidence:           Initial confidence (default 0.5).
            source_type:          Evidence source classification.
            raw_text:             Original text that produced this skill.
            embedding:            Precomputed embedding vector.

        Returns:
            The newly created :class:`Memory` row (not yet committed).
        """
        now = datetime.now(timezone.utc)

        content: dict[str, Any] = {
            "name": name,
            "subject": "skill",
            "relation": "can_do",
            "object": name,
            "trigger_conditions": trigger_conditions or [],
            "steps": steps or [],
            "description": description,
        }
        if domain:
            content["domain"] = domain

        metadata: dict[str, Any] = {
            "source_type": source_type,
            "raw_text": raw_text,
            "outcomes": [],
            "success_count": 0,
            "failure_count": 0,
            "last_used": None,
        }

        record = Memory(
            memory_type=MemoryType.skill,
            user_id=user_id,
            content=content,
            embedding=embedding,
            confidence=confidence,
            status=MemoryStatus.active,
            decay_rate=SKILL_DECAY_RATE,
            created_at=now,
            updated_at=now,
            metadata_=metadata,
        )
        self._session.add(record)
        await self._session.flush()

        logger.debug("Created skill %s: %s", record.memory_id, name)
        return record

    # ------------------------------------------------------------------
    # record_outcome — feedback loop
    # ------------------------------------------------------------------

    async def record_outcome(
        self,
        skill_id: uuid.UUID,
        outcome: SkillOutcome,
        *,
        detail: str | None = None,
    ) -> Memory:
        """Record a success or failure and adjust confidence.

        - **Success:** confidence += ``SUCCESS_REINFORCEMENT``, capped at 1.0
        - **Failure:** confidence *= ``FAILURE_PENALTY_FACTOR``

        If confidence drops below ``DEPRECATION_THRESHOLD``, the skill is
        auto-deprecated.

        Args:
            skill_id:  UUID of the skill record.
            outcome:   ``SkillOutcome.success`` or ``SkillOutcome.failure``.
            detail:    Optional human-readable outcome description.

        Returns:
            The updated :class:`Memory` row.

        Raises:
            MemoryNotFoundError: If the skill doesn't exist.
        """
        record = await self._get_skill(skill_id)
        now = datetime.now(timezone.utc)

        # Adjust confidence
        if outcome == SkillOutcome.success:
            record.confidence = min(1.0, record.confidence + SUCCESS_REINFORCEMENT)
        else:
            record.confidence = record.confidence * FAILURE_PENALTY_FACTOR

        # Update metadata
        meta = dict(record.metadata_) if record.metadata_ else {}
        outcomes = list(meta.get("outcomes", []))
        outcome_entry: dict[str, Any] = {
            "result": outcome.value,
            "timestamp": now.isoformat(),
        }
        if detail:
            outcome_entry["detail"] = detail
        outcomes.append(outcome_entry)
        meta["outcomes"] = outcomes
        meta["last_used"] = now.isoformat()

        if outcome == SkillOutcome.success:
            meta["success_count"] = meta.get("success_count", 0) + 1
        else:
            meta["failure_count"] = meta.get("failure_count", 0) + 1

        record.metadata_ = meta
        record.updated_at = now

        # Auto-deprecation check (inclusive: a skill that lands exactly on the
        # threshold is deprecated, matching the documented "< 0.15" intent).
        if record.confidence <= DEPRECATION_THRESHOLD:
            record.status = MemoryStatus.deprecated
            logger.info(
                "Skill %s auto-deprecated (confidence=%.3f < %.3f)",
                skill_id, record.confidence, DEPRECATION_THRESHOLD,
            )

        await self._session.flush()

        logger.debug(
            "Skill %s outcome=%s, confidence=%.3f",
            skill_id, outcome.value, record.confidence,
        )
        return record

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get(self, skill_id: uuid.UUID) -> Memory | None:
        """Retrieve a single skill by UUID. Returns ``None`` if not found."""
        return await self._get_skill_or_none(skill_id)

    async def match(
        self,
        context: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
    ) -> Sequence[Memory]:
        """Return active skills whose trigger conditions match *context*.

        Performs a simple substring match against each skill's
        ``trigger_conditions`` list.  For production use, this should be
        augmented with embedding-based similarity search.

        Args:
            context:  Text to match against trigger conditions.
            user_id:  Filter to a specific tenant.
            limit:    Maximum number of results.

        Returns:
            Matching skills ordered by confidence (highest first).
        """
        retriever = self._get_retrieval_engine()
        if retriever is not None:
            semantic = await retriever.search(
                context,
                top_k=limit,
                memory_types=[MemoryType.skill],
                user_id=user_id,
                track_access=False,
            )
            if semantic.skills:
                return [sm.memory for sm in semantic.skills[:limit]]

        # No embedding engine bound (e.g. the zero-backend LocalMemory path):
        # fall back to keyword search over SQLite rather than substring-only.
        lexical = await LexicalRetriever(self._session).search(
            context,
            top_k=limit,
            memory_types=[MemoryType.skill],
            user_id=user_id,
        )
        if lexical.skills:
            return [sm.memory for sm in lexical.skills[:limit]]

        all_skills = await self.get_active_skills(user_id=user_id, limit=200)
        context_lower = context.lower()

        matched = []
        for skill in all_skills:
            content = skill.content if isinstance(skill.content, dict) else {}
            triggers = content.get("trigger_conditions", [])
            if any(
                trigger.lower() in context_lower
                for trigger in triggers
                if isinstance(trigger, str)
            ):
                matched.append(skill)

        # Sort by confidence descending
        matched.sort(key=lambda s: s.confidence, reverse=True)
        return matched[:limit]

    async def get_active_skills(
        self,
        *,
        user_id: str | None = None,
        domain: str | None = None,
        limit: int = 50,
    ) -> Sequence[Memory]:
        """Return all active skills, optionally filtered by tenant/domain."""
        filters = [
            Memory.memory_type == MemoryType.skill,
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if domain is not None:
            filters.append(Memory.content["domain"].as_string() == domain)

        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.confidence.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    def _get_retrieval_engine(self) -> RetrievalEngine | None:
        if self._retrieval_engine is not None:
            return self._retrieval_engine

        session_info = getattr(getattr(self._session, "sync_session", self._session), "info", None)
        if not isinstance(session_info, dict):
            return None
        embedding_engine = session_info.get("_clara_embedding_engine")
        if not isinstance(embedding_engine, EmbeddingEngine):
            return None

        cache = session_info.get("_clara_cache")
        lance_engine = session_info.get("_lance_engine")
        self._retrieval_engine = RetrievalEngine(
            self._session,
            embedding_engine,
            cache=cache if isinstance(cache, MemoryCache) else None,
            lance_engine=lance_engine if isinstance(lance_engine, LanceRetrievalEngine) else None,
        )
        return self._retrieval_engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_skill_or_none(self, skill_id: uuid.UUID) -> Memory | None:
        stmt = select(Memory).where(
            and_(
                Memory.memory_id == skill_id,
                Memory.memory_type == MemoryType.skill,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_skill(self, skill_id: uuid.UUID) -> Memory:
        record = await self._get_skill_or_none(skill_id)
        if record is None:
            raise MemoryNotFoundError(str(skill_id))
        return record
