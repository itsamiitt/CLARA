"""
CLARA — Unified Memory Store: SQLAlchemy 2.0 Models

Defines the single-table schema for all memory types (belief, event, skill,
world_model) stored in PostgreSQL with pgvector embeddings.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    Index,
    MetaData,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(str, enum.Enum):
    """Classifies memory records into one of four cognitive categories."""

    belief = "belief"
    event = "event"
    skill = "skill"
    world_model = "world_model"


class MemoryStatus(str, enum.Enum):
    """Lifecycle status of a memory record."""

    active = "active"
    superseded = "superseded"
    deprecated = "deprecated"
    archived = "archived"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Shared declarative base for all CLARA models."""

    metadata = metadata_obj


# ---------------------------------------------------------------------------
# Unified Memory Table
# ---------------------------------------------------------------------------

VECTOR_DIMENSIONS = 1536  # OpenAI text-embedding-3-small / compatible models


class Memory(Base):
    """
    Unified Memory Store — single table for all memory types.

    Every record represents one atomic unit of agent memory: a belief, an
    event, a skill, or a world-model snapshot.  Records are *never deleted*;
    instead they transition through status values (active → superseded →
    deprecated → archived).

    The ``content`` JSONB column holds the type-specific payload (e.g.
    subject/relation/object for beliefs, steps for skills).

    The ``metadata`` JSONB column holds auxiliary information such as
    ``superseded_by``, ``related_beliefs``, ``tags``, ``source``, etc.
    """

    __tablename__ = "memories"

    # --- Primary key ---
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the memory record.",
    )

    # --- Classification ---
    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(MemoryType, name="memory_type_enum", native_enum=True),
        nullable=False,
        comment="Cognitive category: belief | event | skill | world_model.",
    )

    # --- Payload ---
    content: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Type-specific structured content (subject/relation/object, steps, properties, etc.).",
    )

    # --- Embedding ---
    embedding = mapped_column(
        Vector(VECTOR_DIMENSIONS),
        nullable=True,
        comment="Dense vector embedding for semantic similarity search.",
    )

    # --- Scoring ---
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        server_default=text("0.5"),
        comment="Current confidence score (0.0–1.0), subject to Bayesian updates and decay.",
    )

    # --- Lifecycle ---
    status: Mapped[MemoryStatus] = mapped_column(
        Enum(MemoryStatus, name="memory_status_enum", native_enum=True),
        nullable=False,
        default=MemoryStatus.active,
        server_default=text("'active'"),
        comment="Lifecycle status: active | superseded | deprecated | archived.",
    )

    # --- Decay ---
    decay_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.02,
        server_default=text("0.02"),
        comment="Exponential decay rate (per day). 0.0 = no decay (events).",
    )

    # --- Timestamps ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
        comment="Timestamp when this record was first created.",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the most recent update.",
    )

    # --- Auxiliary metadata ---
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
        comment="Auxiliary metadata: superseded_by, tags, source, related IDs, etc.",
    )

    # --- Table-level indexes ---
    __table_args__ = (
        Index("ix_memories_memory_type", "memory_type"),
        Index("ix_memories_status", "status"),
        Index("ix_memories_created_at", "created_at"),
        Index(
            "ix_memories_type_status",
            "memory_type",
            "status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Memory(memory_id={self.memory_id!s}, "
            f"type={self.memory_type.value}, "
            f"status={self.status.value}, "
            f"confidence={self.confidence})>"
        )
