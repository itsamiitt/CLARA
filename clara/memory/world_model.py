"""
CLARA — World Model Memory Store

Manages persistent world-state memories: entity properties, upsert with
mutation auditing, and state queries.

World model records represent *what CLARA knows about the world* — entities,
their properties, and how those properties change over time.  Every property
mutation is recorded in a ``mutation_history`` audit log inside the record's
metadata.

Usage::

    async with async_session() as session:
        world = WorldModelStore(session)
        record = await world.upsert(
            entity_type="tool",
            name="PostgreSQL",
            properties={"version": "16", "role": "primary_db"},
            user_id="alice",
        )
        await world.update_property(record.memory_id, "version", "17")
        await session.commit()
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from clara.core.exceptions import MemoryNotFoundError
from clara.db.models import Memory, MemoryStatus, MemoryType
from clara.graph import project as graph_project

logger = logging.getLogger(__name__)

# World model entries are relatively stable
WORLD_MODEL_DECAY_RATE = 0.005


class WorldModelStore:
    """Async interface for upserting, mutating, and querying world-model entries.

    Each instance is bound to an ``AsyncSession``.  The caller is responsible
    for committing or rolling back the session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # upsert — create or update an entity
    # ------------------------------------------------------------------

    async def upsert(
        self,
        *,
        entity_type: str,
        name: str,
        properties: dict[str, Any] | None = None,
        domain: str | None = None,
        user_id: str | None = None,
        confidence: float = 0.9,
        source_type: str = "user_direct",
        raw_text: str | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Create or update a world-model entity.

        If an active world-model record with the same ``entity_type`` and
        ``name`` exists (scoped to ``user_id``), its properties are merged
        with the provided ones and a mutation is logged.  Otherwise a new
        record is created.

        Args:
            entity_type:   Category of entity (e.g. ``"tool"``, ``"person"``).
            name:          Entity name (e.g. ``"PostgreSQL"``).
            properties:    Key-value properties to set or update.
            domain:        Optional domain tag.
            user_id:       Tenant partition key.
            confidence:    Confidence for this entry (default 0.9).
            source_type:   Evidence source classification.
            raw_text:      Original text that produced this entry.
            embedding:     Precomputed embedding vector.

        Returns:
            The created or updated :class:`Memory` row.
        """
        properties = properties or {}

        # Try to find an existing matching record
        existing = await self._find_existing(entity_type, name, user_id=user_id)

        if existing is not None:
            record = await self._merge_properties(
                existing,
                properties,
                source_type=source_type,
                raw_text=raw_text,
                embedding=embedding,
            )
            return await self._with_graph_projection(record)

        create_error: IntegrityError | None = None
        try:
            async with self._session.begin_nested():
                record = await self._create_new(
                    entity_type=entity_type,
                    name=name,
                    properties=properties,
                    domain=domain,
                    user_id=user_id,
                    confidence=confidence,
                    source_type=source_type,
                    raw_text=raw_text,
                    embedding=embedding,
                )
            return await self._with_graph_projection(record)
        except IntegrityError as exc:
            create_error = exc
            logger.debug(
                "World-model upsert raced for %s/%s (user_id=%r); retrying merge",
                entity_type,
                name,
                user_id,
            )

        existing = await self._find_existing(entity_type, name, user_id=user_id)
        if existing is not None:
            record = await self._merge_properties(
                existing,
                properties,
                source_type=source_type,
                raw_text=raw_text,
                embedding=embedding,
            )
            return await self._with_graph_projection(record)
        if create_error is None:
            raise RuntimeError("World-model upsert failed without an IntegrityError cause")
        raise create_error

    async def _with_graph_projection(self, record: Memory) -> Memory:
        """Fail-soft: mirror the world-model row onto its graph node."""
        await graph_project.project_world_model_upserted(self._session, record)
        return record

    # ------------------------------------------------------------------
    # update_property — single property mutation
    # ------------------------------------------------------------------

    async def update_property(
        self,
        model_id: uuid.UUID,
        key: str,
        value: Any,
        *,
        source_type: str = "user_direct",
        raw_text: str | None = None,
    ) -> Memory:
        """Update a single property on a world-model record.

        Logs the old → new value transition in the mutation history.

        Args:
            model_id:     UUID of the world-model record.
            key:          Property key to update.
            value:        New value for the property.
            source_type:  Evidence source classification.
            raw_text:     Original text that produced this update.

        Returns:
            The updated :class:`Memory` row.

        Raises:
            MemoryNotFoundError: If the record doesn't exist.
        """
        record = await self._get_model(model_id)
        now = datetime.now(timezone.utc)

        content = dict(record.content) if isinstance(record.content, dict) else {}
        props = dict(content.get("properties", {}))
        old_value = props.get(key)

        # Apply the change
        props[key] = value
        content["properties"] = props
        record.content = content

        # Log mutation
        meta = dict(record.metadata_) if record.metadata_ else {}
        history = list(meta.get("mutation_history", []))
        history.append({
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "timestamp": now.isoformat(),
            "source_type": source_type,
            "raw_text": raw_text,
        })
        meta["mutation_history"] = history
        record.metadata_ = meta

        record.updated_at = now
        await self._session.flush()

        logger.debug(
            "World model %s: %s = %r → %r",
            model_id, key, old_value, value,
        )
        return record

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get(self, model_id: uuid.UUID) -> Memory | None:
        """Retrieve a single world-model record by UUID."""
        return await self._get_model_or_none(model_id)

    async def get_state(
        self,
        *,
        user_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> Sequence[Memory]:
        """Return active world-model entries, optionally filtered.

        Args:
            user_id:      Filter to a specific tenant.
            entity_type:  Filter to a specific entity type.
            limit:        Maximum number of records to return.

        Returns:
            A list of :class:`Memory` rows ordered by last update.
        """
        filters = [
            Memory.memory_type == MemoryType.world_model,
            Memory.status == MemoryStatus.active,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)
        if entity_type is not None:
            filters.append(
                Memory.content["entity_type"].as_string() == entity_type
            )

        stmt = (
            select(Memory)
            .where(and_(*filters))
            .order_by(Memory.updated_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_name(
        self,
        name: str,
        *,
        user_id: str | None = None,
    ) -> Memory | None:
        """Find an active world-model record by entity name."""
        filters = [
            Memory.memory_type == MemoryType.world_model,
            Memory.status == MemoryStatus.active,
            Memory.content["name"].as_string() == name,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)

        stmt = select(Memory).where(and_(*filters)).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_history(
        self,
        model_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Return the mutation history for a world-model record.

        Returns:
            A list of mutation entries, each with ``key``, ``old_value``,
            ``new_value``, ``timestamp``, ``source_type``.

        Raises:
            MemoryNotFoundError: If the record doesn't exist.
        """
        record = await self._get_model(model_id)
        meta = record.metadata_ if isinstance(record.metadata_, dict) else {}
        return list(meta.get("mutation_history", []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_existing(
        self,
        entity_type: str,
        name: str,
        *,
        user_id: str | None = None,
    ) -> Memory | None:
        """Find an existing active world-model record by type + name."""
        filters = [
            Memory.memory_type == MemoryType.world_model,
            Memory.status == MemoryStatus.active,
            Memory.content["entity_type"].as_string() == entity_type,
            Memory.content["name"].as_string() == name,
        ]
        if user_id is not None:
            filters.append(Memory.user_id == user_id)

        stmt = select(Memory).where(and_(*filters)).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _create_new(
        self,
        *,
        entity_type: str,
        name: str,
        properties: dict[str, Any],
        domain: str | None,
        user_id: str | None,
        confidence: float,
        source_type: str,
        raw_text: str | None,
        embedding: list[float] | None,
    ) -> Memory:
        """Create a brand-new world-model record."""
        now = datetime.now(timezone.utc)

        content: dict[str, Any] = {
            "entity_type": entity_type,
            "name": name,
            "subject": name,
            "relation": "is_a",
            "object": entity_type,
            "properties": properties,
        }
        if domain:
            content["domain"] = domain

        metadata: dict[str, Any] = {
            "source_type": source_type,
            "raw_text": raw_text,
            "mutation_history": [],
        }

        record = Memory(
            memory_type=MemoryType.world_model,
            user_id=user_id,
            content=content,
            embedding=embedding,
            confidence=confidence,
            status=MemoryStatus.active,
            decay_rate=WORLD_MODEL_DECAY_RATE,
            created_at=now,
            updated_at=now,
            metadata_=metadata,
        )
        self._session.add(record)
        await self._session.flush()

        logger.debug("Created world model %s: %s/%s", record.memory_id, entity_type, name)
        return record

    async def _merge_properties(
        self,
        record: Memory,
        new_properties: dict[str, Any],
        *,
        source_type: str,
        raw_text: str | None,
        embedding: list[float] | None,
    ) -> Memory:
        """Merge new properties into an existing record with mutation logging."""
        now = datetime.now(timezone.utc)

        content = dict(record.content) if isinstance(record.content, dict) else {}
        existing_props = dict(content.get("properties", {}))

        meta = dict(record.metadata_) if record.metadata_ else {}
        history = list(meta.get("mutation_history", []))

        for key, new_val in new_properties.items():
            old_val = existing_props.get(key)
            if old_val != new_val:
                history.append({
                    "key": key,
                    "old_value": old_val,
                    "new_value": new_val,
                    "timestamp": now.isoformat(),
                    "source_type": source_type,
                    "raw_text": raw_text,
                })
            existing_props[key] = new_val

        content["properties"] = existing_props
        record.content = content

        meta["mutation_history"] = history
        record.metadata_ = meta

        if embedding is not None:
            # hybrid_property setter; mypy reads the getter as a plain method.
            record.embedding = embedding  # type: ignore[method-assign]

        record.updated_at = now
        await self._session.flush()

        logger.debug(
            "Updated world model %s: merged %d properties",
            record.memory_id,
            len(new_properties),
        )
        return record

    async def _get_model_or_none(self, model_id: uuid.UUID) -> Memory | None:
        stmt = select(Memory).where(
            and_(
                Memory.memory_id == model_id,
                Memory.memory_type == MemoryType.world_model,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_model(self, model_id: uuid.UUID) -> Memory:
        record = await self._get_model_or_none(model_id)
        if record is None:
            raise MemoryNotFoundError(str(model_id))
        return record
