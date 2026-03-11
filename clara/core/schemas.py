"""
CLARA — Core Schemas

Cross-module data structures used throughout the CLARA pipeline.
These are plain dataclasses (no Pydantic dependency) that serve as
the *lingua franca* between subsystems.

All schemas are frozen and slotted for performance and safety.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from clara.core.enums import SourceType


# ---------------------------------------------------------------------------
# InteractionRecord — Interaction Layer → Extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """A normalized input record produced by the Interaction Layer.

    Carries the raw text together with provenance metadata (source, session,
    user, timestamp) so that downstream stages can make trust and scoping
    decisions without re-parsing the input.

    Attributes:
        interaction_id: Unique identifier for this interaction.
        raw_text:       The cleaned (whitespace-normalized) input text.
        source:         How this input was received.
        session_id:     Optional session identifier for grouping turns.
        user_id:        Optional tenant / user identifier.
        timestamp:      When the interaction was received (UTC).
        metadata:       Arbitrary key-value bag for extensions.
    """

    interaction_id: uuid.UUID
    raw_text: str
    source: SourceType
    session_id: str | None = None
    user_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExtractionCandidate — Extractor → Update Engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExtractionCandidate:
    """A structured fact candidate produced by the Extraction stage.

    This schema is the bridge between the LLM-based extractor and the
    Update Engine.  It carries enough information for the engine to
    classify, embed, conflict-detect, and store the fact.

    Attributes:
        candidate_type: Target memory type (``belief``, ``event``, etc.).
        subject:        Entity the fact is about.
        relation:       Relation or verb (e.g. ``"uses"``, ``"deployed"``).
        object:         Target of the relation.
        domain:         Optional domain tag for scoped facts.
        event_type:     For events: the event verb (e.g. ``"deployed"``).
        description:    Optional longer description.
        is_negation:    Whether this fact negates a prior belief.
        negates:        Free-text description of what is being invalidated.
        confidence:     Extraction confidence in [0.0, 1.0].
        source_type:    Evidence classification.
        raw_evidence:   The original text fragment that produced this fact.
    """

    candidate_type: str  # "belief" | "event" | "skill" | "world_model"
    subject: str
    relation: str
    object: str
    confidence: float
    raw_evidence: str
    source_type: str = "user_direct"
    domain: str | None = None
    event_type: str | None = None
    description: str | None = None
    is_negation: bool = False
    negates: str | None = None


# ---------------------------------------------------------------------------
# MemorySummary — lightweight representation for API responses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MemorySummary:
    """Lightweight representation of a Memory record for API responses.

    Avoids exposing SQLAlchemy model internals over the wire.

    Attributes:
        memory_id:   UUID of the memory record.
        memory_type: Cognitive category.
        content:     The JSONB content payload.
        confidence:  Current confidence score.
        status:      Lifecycle status.
        user_id:     Tenant identifier (may be ``None`` for legacy records).
        created_at:  Record creation timestamp (ISO format string).
        updated_at:  Last update timestamp (ISO format string).
    """

    memory_id: str
    memory_type: str
    content: dict[str, Any]
    confidence: float
    status: str
    user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
