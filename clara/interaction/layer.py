"""CLARA - Interaction Layer.

Normalizes raw inbound text into a structured interaction record before it
enters extraction or memory update flows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from clara.memory.belief import SourceType

DEFAULT_CONFIDENCE_FLOOR = 0.4


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """Structured representation of a normalized inbound interaction."""

    interaction_id: uuid.UUID
    raw_text: str
    source: SourceType
    session_id: str | None
    user_id: str | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionLayer:
    """Stateless input normalization for the public agent facade."""

    def receive(
        self,
        raw_input: str,
        *,
        source: SourceType | str = SourceType.user_direct,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> InteractionRecord:
        """Normalize raw text into an :class:`InteractionRecord`.

        Raises:
            ValueError: If the input is empty after whitespace normalization or
                the source cannot be parsed into ``SourceType``.
        """
        normalized = " ".join(raw_input.split())
        if not normalized:
            raise ValueError("Interaction input cannot be empty.")

        if isinstance(source, str):
            source = SourceType(source)

        return InteractionRecord(
            interaction_id=uuid.uuid4(),
            raw_text=normalized,
            source=source,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            metadata={"confidence_floor": DEFAULT_CONFIDENCE_FLOOR},
        )
