"""
CLARA — Memory Update Engine

Orchestrates the full pipeline from an extracted fact to a write in the
Unified Memory Store.  Implements the processing pipeline and conflict
resolution protocol described in CONTEXT.md §3 (Memory Update Engine).

Pipeline:
    1. Embed the fact text via :class:`~clara.retrieval.embeddings.EmbeddingEngine`
    2. Search top-10 similar memories (cosine threshold ≥ 0.82)
    3. Classify the fact into a memory type (belief / event / skill / world_model)
    4. Detect conflicts against retrieved candidates
    5. Resolve conflicts per the CONTEXT.md protocol
    6. Write the result: create new or supersede existing
"""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from clara.db.models import Memory, MemoryStatus, MemoryType, VECTOR_DIMENSIONS
from clara.extraction.extractor import ExtractedFact
from clara.memory.belief import BeliefMemory, SourceType
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine, normalize_embedding_dimensions
from clara.retrieval.engine import RetrievalEngine, ScoredMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: float = 0.82
SEARCH_TOP_K: int = 10
CONFLICT_CONFIDENCE_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class ActionTaken(str, enum.Enum):
    """What the update engine did with a given fact."""

    created = "created"
    superseded = "superseded"
    reinforced = "reinforced"
    retained_both = "retained_both"
    skipped = "skipped"


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """Outcome of processing a single :class:`ExtractedFact`.

    Attributes:
        action_taken:      What happened (created / superseded / etc.).
        memory_id:         UUID of the new or updated memory record.
        conflict_detected: Whether a conflicting candidate was found.
        superseded_id:     UUID of the record that was superseded, if any.
    """

    action_taken: ActionTaken
    memory_id: uuid.UUID | None
    conflict_detected: bool
    superseded_id: uuid.UUID | None = None


# ---------------------------------------------------------------------------
# Helpers — type classification & conflict detection
# ---------------------------------------------------------------------------

# Keywords used as heuristics for classifying the fact into a memory type.
# Relations are normalised (lowercased, spaces/hyphens → underscores) before
# matching, so "proficient in" and "proficient-in" both match "proficient_in".
_EVENT_RELATIONS: frozenset[str] = frozenset({
    "started", "completed", "failed", "updated", "finished",
    "began", "launched", "deployed", "shipped", "released",
    "created", "deleted", "migrated", "installed", "upgraded",
    "ran", "fixed", "merged", "pushed", "removed", "abandoned",
})

_SKILL_RELATIONS: frozenset[str] = frozenset({
    "knows", "learned", "studies", "proficient_in", "certified_in",
    "can_do", "trained_in", "experienced_with", "mastered",
    "skilled_in", "specializes_in", "able_to", "good_at",
})

_WORLD_MODEL_RELATIONS: frozenset[str] = frozenset({
    "is", "has", "contains", "belongs_to", "located_in",
    "consists_of", "type_of", "instance_of", "part_of",
    "runs_on", "depends_on", "configured_with",
})


def _normalize_relation(relation: str) -> str:
    """Canonicalise a relation for keyword matching."""
    return relation.lower().strip().replace(" ", "_").replace("-", "_")


def classify_memory_type(fact: ExtractedFact) -> MemoryType:
    """Classify a fact into one of the four memory types.

    Uses the ``relation`` field as the primary signal, with a fixed precedence
    of **event > skill > world_model > belief**: the first keyword set that the
    (normalised) relation matches wins, and anything unmatched falls back to
    :attr:`MemoryType.belief` — the default for general preference/usage triples.
    """
    rel = _normalize_relation(fact.relation)

    if rel in _EVENT_RELATIONS:
        return MemoryType.event
    if rel in _SKILL_RELATIONS:
        return MemoryType.skill
    if rel in _WORLD_MODEL_RELATIONS:
        return MemoryType.world_model

    # Default — most facts about user preferences, usage, etc. are beliefs
    return MemoryType.belief


def _is_conflicting(fact: ExtractedFact, candidate: ScoredMemory) -> bool:
    """Determine whether *fact* directly contradicts *candidate*.

    A conflict exists when:
    * Same subject and relation, but different object
    * The fact is a negation of the candidate's content
    """
    content = candidate.memory.content
    if not isinstance(content, dict):
        return False

    same_subject = (
        content.get("subject", "").lower() == fact.subject.lower()
    )
    same_relation = (
        content.get("relation", "").lower() == fact.relation.lower()
    )
    candidate_negation = bool(content.get("is_negation", False))
    same_object = (
        content.get("object", "").lower() == fact.object.lower()
    )
    different_object = (
        not same_object
    )

    if not (same_subject and same_relation):
        return False

    # Same proposition, opposite polarity.
    if same_object and candidate_negation != fact.is_negation:
        return True

    # Direct contradiction between competing positive beliefs.
    if not candidate_negation and not fact.is_negation and different_object:
        return True

    return False


def _same_belief(fact: ExtractedFact, candidate: ScoredMemory) -> bool:
    """Return True when the candidate represents the same belief proposition."""
    content = candidate.memory.content
    if not isinstance(content, dict):
        return False

    fact_domain = (fact.domain or "").strip().lower()
    candidate_domain = (content.get("domain", "") or "").strip().lower()

    return (
        content.get("subject", "").lower() == fact.subject.lower()
        and content.get("relation", "").lower() == fact.relation.lower()
        and content.get("object", "").lower() == fact.object.lower()
        and bool(content.get("is_negation", False)) == fact.is_negation
        and fact_domain == candidate_domain
    )


def _conflict_priority(fact: ExtractedFact, candidate: ScoredMemory) -> tuple[int, float, float]:
    """Rank conflicts so polarity conflicts beat generic object conflicts."""
    content = candidate.memory.content if isinstance(candidate.memory.content, dict) else {}
    candidate_negation = bool(content.get("is_negation", False))
    same_object = content.get("object", "").lower() == fact.object.lower()

    if same_object and candidate_negation != fact.is_negation:
        return (2, candidate.similarity, candidate.memory.confidence)
    if not candidate_negation and not fact.is_negation and not same_object:
        return (1, candidate.similarity, candidate.memory.confidence)
    return (0, candidate.similarity, candidate.memory.confidence)


def _domains_differ(fact: ExtractedFact, candidate: ScoredMemory) -> bool:
    """Return ``True`` if the fact and candidate have different non-null domains."""
    fact_domain = (fact.domain or "").strip().lower()
    candidate_domain = (
        candidate.memory.content.get("domain", "") or ""
    ).strip().lower()

    # Both have domains and they differ → domain-scoped retention
    if fact_domain and candidate_domain and fact_domain != candidate_domain:
        return True
    return False


def _map_source_type(source_type_str: str) -> SourceType:
    """Map a string source_type from ExtractedFact to the SourceType enum."""
    try:
        return SourceType(source_type_str)
    except ValueError:
        return SourceType.agent_inference


# ---------------------------------------------------------------------------
# MemoryUpdateEngine
# ---------------------------------------------------------------------------

class MemoryUpdateEngine:
    """Orchestrates fact → memory writes with conflict resolution.

    Usage::

        engine = MemoryUpdateEngine(session, embedder, retrieval_engine)
        result = await engine.process(extracted_fact)

    The engine never commits — the caller manages the transaction boundary.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine,
        retrieval_engine: RetrievalEngine,
        cache: MemoryCache | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedding_engine
        self._retriever = retrieval_engine
        self._belief_memory = BeliefMemory(session)
        self._cache = cache

    # ------------------------------------------------------------------
    # process — main entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        fact: ExtractedFact,
        *,
        user_id: str | None = None,
    ) -> UpdateResult:
        """Process a single extracted fact through the full update pipeline.

        Args:
            fact: A structured fact from :class:`FactExtractor`.

        Returns:
            An :class:`UpdateResult` describing the action taken.
        """
        # [1] Similarity search
        # [2] Classify memory type
        memory_type = classify_memory_type(fact)

        # [1] Similarity search
        candidates = await self._find_similar(fact, user_id=user_id)
        if memory_type == MemoryType.belief:
            candidates = self._merge_candidates(
                candidates,
                await self._find_exact_belief_candidates(fact, user_id=user_id),
            )

        # [3] Filter candidates to same memory type and above threshold
        typed_candidates = [
            c for c in candidates
            if c.memory.memory_type == memory_type
            and c.similarity >= SIMILARITY_THRESHOLD
        ]

        # [4] Conflict detection
        conflicts = [
            c for c in typed_candidates
            if _is_conflicting(fact, c)
        ]

        # [5] Resolve
        if conflicts:
            result = await self._resolve_conflict(
                fact,
                conflicts,
                memory_type,
                user_id=user_id,
            )
            await self._invalidate_cache(user_id)
            return result

        # [6] No conflict — check if this reinforces an existing memory
        if typed_candidates:
            result = await self._reinforce_or_create(
                fact,
                typed_candidates,
                memory_type,
                user_id=user_id,
            )
            await self._invalidate_cache(user_id)
            return result

        # Novel fact — create new
        result = await self._create_new(fact, memory_type, user_id=user_id)
        await self._invalidate_cache(user_id)
        return result

    # ------------------------------------------------------------------
    # Pipeline step helpers
    # ------------------------------------------------------------------

    async def _find_similar(
        self,
        fact: ExtractedFact,
        *,
        user_id: str | None = None,
    ) -> list[ScoredMemory]:
        """Embed the fact and search for similar memories."""
        result = await self._retriever.search(
            self._fact_text(fact),
            top_k=SEARCH_TOP_K,
            user_id=user_id,
        )
        return result.all

    @staticmethod
    def _fact_text(fact: ExtractedFact) -> str:
        """Canonical text form used for both retrieval and stored embeddings."""
        text = f"{fact.subject} {fact.relation} {fact.object}"
        if fact.domain:
            text += f" ({fact.domain})"
        if fact.is_negation:
            text = f"not {text}"
        return text

    def _fact_embedding(self, fact: ExtractedFact) -> list[float]:
        return normalize_embedding_dimensions(
            self._embedder.embed(self._fact_text(fact)),
            target_dimensions=VECTOR_DIMENSIONS,
        )

    async def _find_exact_belief_candidates(
        self,
        fact: ExtractedFact,
        *,
        user_id: str | None = None,
    ) -> list[ScoredMemory]:
        """Fetch active beliefs with the same subject + relation exactly."""
        try:
            records = await self._belief_memory.get_active_beliefs(
                subject=fact.subject,
                relation=fact.relation,
                user_id=user_id,
                limit=100,
            )
        except Exception:
            logger.debug(
                "Exact belief candidate lookup failed; falling back to similarity-only search",
                exc_info=True,
            )
            return []
        return [
            ScoredMemory(
                memory=record,
                score=1.0,
                similarity=1.0,
                confidence=record.confidence,
                recency_score=1.0,
                usage_frequency=0.0,
            )
            for record in records
        ]

    @staticmethod
    def _merge_candidates(
        *candidate_lists: list[ScoredMemory],
    ) -> list[ScoredMemory]:
        merged: dict[uuid.UUID, ScoredMemory] = {}
        for candidates in candidate_lists:
            for candidate in candidates:
                existing = merged.get(candidate.memory.memory_id)
                if existing is None or candidate.similarity > existing.similarity:
                    merged[candidate.memory.memory_id] = candidate
        return list(merged.values())

    async def _resolve_conflict(
        self,
        fact: ExtractedFact,
        conflicts: list[ScoredMemory],
        memory_type: MemoryType,
        *,
        user_id: str | None = None,
    ) -> UpdateResult:
        """Apply the CONTEXT.md conflict resolution protocol."""
        primary_conflict = max(conflicts, key=lambda c: _conflict_priority(fact, c))
        old_memory = primary_conflict.memory

        # Case 1: Domain-scoped — retain both with domain tags
        if _domains_differ(fact, primary_conflict):
            logger.info(
                "Domain-scoped conflict: retaining both (fact domain=%r, "
                "existing domain=%r)",
                fact.domain,
                old_memory.content.get("domain"),
            )
            new_record = await self._store_fact(fact, memory_type, user_id=user_id)
            return UpdateResult(
                action_taken=ActionTaken.retained_both,
                memory_id=new_record.memory_id,
                conflict_detected=True,
            )

        # Case 2: Newer + high confidence → supersede
        if fact.confidence > CONFLICT_CONFIDENCE_THRESHOLD:
            logger.info(
                "Superseding memory %s (confidence=%.2f → %.2f)",
                old_memory.memory_id, old_memory.confidence, fact.confidence,
            )
            if memory_type == MemoryType.belief:
                old_record, new_record = await self._belief_memory.supersede(
                    old_memory.memory_id,
                    subject=fact.subject,
                    relation=fact.relation,
                    object_=fact.object,
                    domain=fact.domain,
                    is_negation=fact.is_negation,
                    source=_map_source_type(fact.source_type),
                    raw_text=fact.raw_text,
                    user_id=user_id,
                    embedding=self._fact_embedding(fact),
                )
                return UpdateResult(
                    action_taken=ActionTaken.superseded,
                    memory_id=new_record.memory_id,
                    conflict_detected=True,
                    superseded_id=old_record.memory_id,
                )
            else:
                # For non-belief types, create new and mark old as superseded
                new_record = await self._store_fact(fact, memory_type, user_id=user_id)
                await self._mark_superseded(old_memory, new_record.memory_id)
                return UpdateResult(
                    action_taken=ActionTaken.superseded,
                    memory_id=new_record.memory_id,
                    conflict_detected=True,
                    superseded_id=old_memory.memory_id,
                )

        # Case 3: Ambiguous — store both
        logger.info(
            "Ambiguous conflict (confidence=%.2f ≤ %.2f): storing both",
            fact.confidence, CONFLICT_CONFIDENCE_THRESHOLD,
        )
        new_record = await self._store_fact(fact, memory_type, user_id=user_id)
        return UpdateResult(
            action_taken=ActionTaken.retained_both,
            memory_id=new_record.memory_id,
            conflict_detected=True,
        )

    async def _reinforce_or_create(
        self,
        fact: ExtractedFact,
        candidates: list[ScoredMemory],
        memory_type: MemoryType,
        *,
        user_id: str | None = None,
    ) -> UpdateResult:
        """When a similar non-conflicting memory exists, reinforce it."""
        best = max(candidates, key=lambda c: c.similarity)

        # Only reinforce beliefs — events, skills, etc. are always new entries
        if (
            memory_type == MemoryType.belief
            and best.similarity >= SIMILARITY_THRESHOLD
            and _same_belief(fact, best)
        ):
            updated = await self._belief_memory.update(
                best.memory.memory_id,
                source=_map_source_type(fact.source_type),
                raw_text=fact.raw_text,
                embedding=self._fact_embedding(fact),
            )
            return UpdateResult(
                action_taken=ActionTaken.reinforced,
                memory_id=updated.memory_id,
                conflict_detected=False,
            )

        # Non-belief or below threshold → create new
        return await self._create_new(fact, memory_type, user_id=user_id)

    async def _create_new(
        self,
        fact: ExtractedFact,
        memory_type: MemoryType,
        *,
        user_id: str | None = None,
    ) -> UpdateResult:
        """Create a brand-new memory record."""
        record = await self._store_fact(fact, memory_type, user_id=user_id)
        return UpdateResult(
            action_taken=ActionTaken.created,
            memory_id=record.memory_id,
            conflict_detected=False,
        )

    # ------------------------------------------------------------------
    # Low-level storage
    # ------------------------------------------------------------------

    async def _store_fact(
        self,
        fact: ExtractedFact,
        memory_type: MemoryType,
        *,
        user_id: str | None = None,
    ) -> Memory:
        """Persist a fact as a new Memory record.

        Uses :class:`BeliefMemory.store` for beliefs; for other types,
        creates the record directly.
        """
        embedding = self._fact_embedding(fact)

        if memory_type == MemoryType.belief:
            return await self._belief_memory.store(
                subject=fact.subject,
                relation=fact.relation,
                object_=fact.object,
                domain=fact.domain,
                is_negation=fact.is_negation,
                source=_map_source_type(fact.source_type),
                raw_text=fact.raw_text,
                user_id=user_id,
                embedding=embedding,
            )

        # Non-belief types: create directly
        now = datetime.now(timezone.utc)
        content: dict[str, Any] = {
            "subject": fact.subject,
            "relation": fact.relation,
            "object": fact.object,
        }
        if fact.domain:
            content["domain"] = fact.domain
        if fact.is_negation:
            content["is_negation"] = True

        record = Memory(
            memory_type=memory_type,
            user_id=user_id,
            content=content,
            embedding=embedding,
            confidence=fact.confidence,
            status=MemoryStatus.active,
            decay_rate=0.0 if memory_type == MemoryType.event else 0.02,
            created_at=now,
            updated_at=now,
            metadata_={
                "source_type": fact.source_type,
                "raw_text": fact.raw_text,
            },
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def _mark_superseded(
        self,
        old_memory: Memory,
        new_memory_id: uuid.UUID,
    ) -> None:
        """Mark a non-belief memory as superseded."""
        old_memory.status = MemoryStatus.superseded
        old_memory.updated_at = datetime.now(timezone.utc)

        meta = dict(old_memory.metadata_) if old_memory.metadata_ else {}
        meta["superseded_by"] = str(new_memory_id)
        old_memory.metadata_ = meta

        await self._session.flush()

    async def _invalidate_cache(self, user_id: str | None) -> None:
        if self._cache is not None:
            await self._cache.invalidate(user_id)
