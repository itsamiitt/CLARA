"""
CLARA - Unified Memory Store: SQLAlchemy 2.0 Models

Defines the single-table schema for all memory types (belief, event, skill,
world_model) stored in SQLite, with embeddings kept in LanceDB.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Text,
    DateTime,
    Enum,
    event,
    Float,
    func,
    Index,
    MetaData,
    Uuid,
    literal,
    text,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, reconstructor


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
JSON_STORAGE_TYPE = JSON


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
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the memory record.",
    )

    # --- Tenant ---
    user_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Tenant partition key. NULL for backwards-compat with pre-tenant data.",
    )

    # --- Classification ---
    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(MemoryType, name="memory_type_enum", native_enum=True),
        nullable=False,
        comment="Cognitive category: belief | event | skill | world_model.",
    )

    # --- Payload ---
    content: Mapped[dict] = mapped_column(
        JSON_STORAGE_TYPE,
        nullable=False,
        comment="Type-specific structured content (subject/relation/object, steps, properties, etc.).",
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
    # server_default uses func.now() (compiles to CURRENT_TIMESTAMP on
    # SQLite) rather than raw "now()", which SQLite rejects at runtime for
    # non-ORM inserts — external tools writing rows directly would fail.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        comment="Timestamp when this record was first created.",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the most recent update.",
    )

    # --- Auxiliary metadata ---
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON_STORAGE_TYPE,
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
        Index("ix_memories_user_id", "user_id"),
        Index(
            "ix_memories_user_type_status",
            "user_id",
            "memory_type",
            "status",
        ),
        # Enforces one active world-model record per (user, entity_type, name)
        # so concurrent upserts collide with an IntegrityError instead of
        # duplicating. NOTE: SQLite-only — the json_extract() key expressions
        # do not exist in PostgreSQL (which needs content->>'entity_type');
        # only the WHERE predicate is dual-dialect. PostgreSQL is not a
        # supported target yet (see README).
        Index(
            "uq_memories_world_model_identity",
            text("coalesce(user_id, '')"),
            text("json_extract(content, '$.entity_type')"),
            text("json_extract(content, '$.name')"),
            unique=True,
            sqlite_where=text("memory_type = 'world_model' AND status = 'active'"),
            postgresql_where=text("memory_type = 'world_model' AND status = 'active'"),
        ),
    )

    @reconstructor
    def _init_on_load(self) -> None:
        self._embedding_cache = None

    @hybrid_property
    def embedding(self) -> list[float] | None:
        """Compatibility shim for code paths that still set ``record.embedding``.

        Embeddings are stored in LanceDB after the PostgreSQL/pgvector removal,
        but several existing write paths still pass ``embedding=...`` into the
        ORM constructor or assign ``record.embedding`` during updates. Keep an
        instance-local cache so those callers remain unchanged.
        """
        cached = getattr(self, "_embedding_cache", None)
        if cached is None:
            return None
        return [float(value) for value in cached]

    @embedding.setter
    def embedding(self, value: list[float] | tuple[float, ...] | None) -> None:
        if value is None:
            self._embedding_cache = None
            return
        self._embedding_cache = [float(item) for item in value]

    @embedding.expression
    def embedding(cls):
        # There is no persisted SQL column anymore. Returning a non-null literal
        # keeps legacy ``Memory.embedding.is_(None)`` checks deterministic in
        # tests without implying any real database storage.
        return literal(False)

    def __repr__(self) -> str:
        return (
            f"<Memory(memory_id={self.memory_id!s}, "
            f"user_id={self.user_id!r}, "
            f"type={self.memory_type.value}, "
            f"status={self.status.value}, "
            f"confidence={self.confidence})>"
        )


@event.listens_for(Memory, "expire")
def _clear_embedding_cache_on_expire(target: Memory, attrs) -> None:
    target._embedding_cache = None


@event.listens_for(Memory, "refresh")
def _clear_embedding_cache_on_refresh(target: Memory, context, attrs) -> None:
    target._embedding_cache = None
