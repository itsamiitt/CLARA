"""
CLARA — Belief Memory Module

Manages atomic subject–relation–object beliefs in the Unified Memory Store.
Beliefs are never deleted; they are versioned and superseded when contradicted.

Confidence is computed via source-weighted Bayesian updating with exponential
time decay, as described in CONTEXT.md §4 (Belief Memory → Confidence Scoring).
"""

from __future__ import annotations

import enum
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory, MemoryStatus, MemoryType


# ---------------------------------------------------------------------------
# Source weights (from CONTEXT.md)
# ---------------------------------------------------------------------------

class SourceType(str, enum.Enum):
    """Classification of the evidence source that produced a belief."""

    user_direct = "user_direct"
    user_indirect = "user_indirect"
    tool_api = "tool_api"
    system = "system"
    agent_inference = "agent_inference"
    agent_reflection = "agent_reflection"


SOURCE_WEIGHTS: dict[SourceType, float] = {
    SourceType.user_direct: 1.0,
    SourceType.user_indirect: 0.7,
    SourceType.tool_api: 0.85,
    SourceType.system: 0.75,
    SourceType.agent_inference: 0.5,
    SourceType.agent_reflection: 0.5,
}

# Default decay rates for belief sub-types (per day)
STABLE_DECAY_RATE = 0.005
VOLATILE_DECAY_RATE = 0.02
DEFAULT_DECAY_RATE = VOLATILE_DECAY_RATE

# Observation strength for a single new piece of evidence
DEFAULT_OBSERVATION_STRENGTH = 1.0


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _ensure_aware(dt: datetime) -> datetime:
    """Normalize naive datetimes to UTC for SQLite-backed test environments."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def compute_confidence(
    *,
    prior_confidence: float,
    prior_weight: float,
    decay_rate: float,
    days_since_last_seen: float,
    source_weight: float,
    observation_strength: float = DEFAULT_OBSERVATION_STRENGTH,
) -> float:
    """Source-weighted Bayesian confidence update with exponential decay.

        decay_factor    = e^(−decay_rate × days_since_last_seen)
        confidence_new  = (prior × prior_weight × decay_factor
                           + source_weight × observation_strength)
                          / (prior_weight + 1)

    The prior is scaled by ``prior_weight`` (the number of prior
    observations) so the update is a proper weighted mean: confirming an
    already-confident belief keeps confidence high instead of diluting it
    toward zero as the observation count grows.

    Returns:
        The updated confidence, clamped to [0.0, 1.0].
    """
    decay_factor = math.exp(-decay_rate * days_since_last_seen)
    numerator = (
        prior_confidence * prior_weight * decay_factor
        + source_weight * observation_strength
    )
    denominator = prior_weight + 1.0
    return max(0.0, min(1.0, numerator / denominator))


# ---------------------------------------------------------------------------
# BeliefMemory
# ---------------------------------------------------------------------------

class BeliefMemory:
    """Async interface for creating, reading, updating, and superseding beliefs.

    Each instance is bound to an ``AsyncSession``.  The caller is responsible
    for committing or rolling back the session (``BeliefMemory`` only flushes).

    Usage::

        async with async_session() as session:
            beliefs = BeliefMemory(session)
            record = await beliefs.store(
                subject="user",
                relation="uses",
                object_="Rust",
                domain="systems",
                source=SourceType.user_direct,
                raw_text="I switched from Python to Rust for my systems work.",
            )
            await session.commit()
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------

    async def store(
        self,
        *,
        subject: str,
        relation: str,
        object_: str,
        domain: str | None = None,
        is_negation: bool = False,
        source: SourceType = SourceType.user_direct,
        raw_text: str | None = None,
        user_id: str | None = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
        observation_strength: float = DEFAULT_OBSERVATION_STRENGTH,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Create a new belief record in the Unified Memory Store.

        The initial confidence is set to the source weight of the evidence
        (equivalent to a Bayesian update with prior_confidence=0,
        prior_weight=0).

        Args:
            subject:   Entity the belief is about (e.g. ``"user"``).
            relation:  Relation predicate (e.g. ``"uses"``).
            object_:   Target of the relation (e.g. ``"Rust"``).
            domain:    Optional domain tag for scoped beliefs.
            source:    Evidence source classification.
            raw_text:  Original text that produced this belief.
            decay_rate: Per-day exponential decay rate.
            observation_strength: Strength of this observation (default 1.0).

        Returns:
            The newly created :class:`Memory` row (not yet committed).
        """
        source_weight = SOURCE_WEIGHTS[source]

        # First observation → confidence = source_weight * obs_strength / 1
        initial_confidence = min(1.0, source_weight * observation_strength)

        now = datetime.now(timezone.utc)

        evidence_entry: dict[str, Any] = {
            "text": raw_text,
            "source": source.value,
            "timestamp": now.isoformat(),
        }

        content: dict[str, Any] = {
            "subject": subject,
            "relation": relation,
            "object": object_,
        }
        if domain is not None:
            content["domain"] = domain
        if is_negation:
            content["is_negation"] = True

        metadata: dict[str, Any] = {
            "evidence": [evidence_entry],
            "source_weight": source_weight,
            "prior_weight": 1.0,  # one observation so far
        }

        record = Memory(
            memory_type=MemoryType.belief,
            user_id=user_id,
            content=content,
            embedding=embedding,
            confidence=initial_confidence,
            status=MemoryStatus.active,
            decay_rate=decay_rate,
            created_at=now,
            updated_at=now,
            metadata_=metadata,
        )

        self._session.add(record)
        await self._session.flush()
        return record

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    async def get(
        self,
        memory_id: uuid.UUID,
    ) -> Memory | None:
        """Retrieve a single belief by its UUID.

        Returns ``None`` if the record does not exist or is not a belief.
        """
        stmt = select(Memory).where(
            and_(
                Memory.memory_id == memory_id,
                Memory.memory_type == MemoryType.belief,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------

    async def get_active_beliefs(
        self,
        subject: str | None = None,
        relation: str | None = None,
        domain: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[Memory]:
        """Return active beliefs, optionally filtered by subject / relation / domain."""
        filters = [
            Memory.memory_type == MemoryType.belief,
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if subject is not None:
            filters.append(Memory.content["subject"].as_string() == subject)
        if relation is not None:
            filters.append(Memory.content["relation"].as_string() == relation)
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

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    async def update(
        self,
        memory_id: uuid.UUID,
        *,
        source: SourceType,
        raw_text: str | None = None,
        observation_strength: float = DEFAULT_OBSERVATION_STRENGTH,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Reinforce an existing belief with a new observation.

        Applies the Bayesian confidence update formula from CONTEXT.md.
        Appends the new evidence entry to the evidence trail.

        Args:
            memory_id: UUID of the belief to reinforce.
            source:    Source classification of the new evidence.
            raw_text:  Original text that produced this evidence.
            observation_strength: Strength of the observation (default 1.0).

        Returns:
            The updated :class:`Memory` row.

        Raises:
            ValueError: If the record does not exist or is not an active belief.
        """
        record = await self.get(memory_id)
        if record is None:
            raise ValueError(f"Belief {memory_id} not found.")
        if record.status != MemoryStatus.active:
            raise ValueError(
                f"Cannot update belief {memory_id} — status is "
                f"{record.status.value!r}, expected 'active'."
            )

        now = datetime.now(timezone.utc)
        source_weight = SOURCE_WEIGHTS[source]

        # Compute days since last update
        updated_at = _ensure_aware(record.updated_at)
        days_elapsed = max(0.0, (now - updated_at).total_seconds() / 86_400.0)

        meta = dict(record.metadata_) if record.metadata_ else {}
        prior_weight = meta.get("prior_weight", 1.0)

        new_confidence = compute_confidence(
            prior_confidence=record.confidence,
            prior_weight=prior_weight,
            decay_rate=record.decay_rate,
            days_since_last_seen=days_elapsed,
            source_weight=source_weight,
            observation_strength=observation_strength,
        )

        # Append evidence
        evidence: list[dict[str, Any]] = list(meta.get("evidence", []))
        evidence.append({
            "text": raw_text,
            "source": source.value,
            "timestamp": now.isoformat(),
        })

        meta["evidence"] = evidence
        meta["source_weight"] = source_weight
        meta["prior_weight"] = prior_weight + 1.0

        record.confidence = new_confidence
        if embedding is not None:
            record.embedding = embedding
        record.updated_at = now
        record.metadata_ = meta

        await self._session.flush()
        return record

    # ------------------------------------------------------------------
    # supersede
    # ------------------------------------------------------------------

    async def supersede(
        self,
        old_memory_id: uuid.UUID,
        *,
        subject: str,
        relation: str,
        object_: str,
        domain: str | None = None,
        is_negation: bool = False,
        source: SourceType = SourceType.user_direct,
        raw_text: str | None = None,
        user_id: str | None = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
        embedding: list[float] | None = None,
    ) -> tuple[Memory, Memory]:
        """Replace an existing belief with a new one.

        The old belief is marked ``superseded`` and linked to the new record
        via ``metadata.superseded_by``.  The new belief is stored as an
        independent active record with ``metadata.supersedes`` pointing back.

        Args:
            old_memory_id: UUID of the belief to supersede.
            subject, relation, object_, domain, source, raw_text, decay_rate:
                Fields for the *new* belief (same semantics as :meth:`store`).

        Returns:
            A ``(old_record, new_record)`` tuple.

        Raises:
            ValueError: If the old record does not exist or is not an active
            belief.
        """
        old_record = await self.get(old_memory_id)
        if old_record is None:
            raise ValueError(f"Belief {old_memory_id} not found.")
        if old_record.status != MemoryStatus.active:
            raise ValueError(
                f"Cannot supersede belief {old_memory_id} — status is "
                f"{old_record.status.value!r}, expected 'active'."
            )

        # Create the replacement belief first so we have its UUID
        new_record = await self.store(
            subject=subject,
            relation=relation,
            object_=object_,
            domain=domain,
            is_negation=is_negation,
            source=source,
            raw_text=raw_text,
            user_id=old_record.user_id if user_id is None else user_id,
            decay_rate=decay_rate,
            embedding=embedding,
        )

        # Mark old record as superseded
        now = datetime.now(timezone.utc)
        old_record.status = MemoryStatus.superseded
        old_record.updated_at = now

        old_meta = dict(old_record.metadata_) if old_record.metadata_ else {}
        old_meta["superseded_by"] = str(new_record.memory_id)
        old_record.metadata_ = old_meta

        # Back-link on the new record
        new_meta = dict(new_record.metadata_) if new_record.metadata_ else {}
        new_meta["supersedes"] = str(old_record.memory_id)
        new_record.metadata_ = new_meta

        await self._session.flush()
        return old_record, new_record
