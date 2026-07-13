"""OpenClaw-friendly memory bridge on top of ClaraMemory.

.. deprecated::
    This adapter hard-partitions memory per session
    (``user_id="session:<id>"``), so nothing learned in one session is
    recallable in another — the opposite of long-term memory. The MCP
    server (``clara-mcp``) + :class:`~clara.integrations.local_memory.LocalMemory`
    superseded it for agent integrations. Scheduled for removal.

This adapter gives a thin, ergonomic layer for chat/session workflows:
- remember_turn(session, role, text, metadata)
- recall_for(session, query)

It does not change CLARA internals; it standardizes how chat turns are
serialized into memory text so they can be recalled later.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from clara.agent import ClaraMemory
from clara.retrieval.engine import RetrievalResult


@dataclass(slots=True)
class BridgeConfig:
    """Configuration for bridge serialization and retrieval defaults."""

    default_top_k: int = 8
    include_metadata: bool = True


class OpenClawMemoryBridge:
    """Session-oriented adapter for storing and retrieving past memory."""

    def __init__(self, memory: ClaraMemory, config: BridgeConfig | None = None) -> None:
        warnings.warn(
            "OpenClawMemoryBridge is deprecated: its per-session user_id "
            "partitioning defeats cross-session memory. Use the clara-mcp "
            "server / LocalMemory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.memory = memory
        self.config = config or BridgeConfig()

    async def remember_turn(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Store one chat turn into CLARA memory."""
        if not text or not text.strip():
            return []

        payload = self._serialize_turn(
            session_id=session_id,
            role=role,
            text=text,
            metadata=metadata or {},
        )
        return await self.memory.remember(
            payload,
            user_id=self._session_user_id(session_id),
        )

    async def recall_for(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Recall memories biased to a given session by query prefixing."""
        k = top_k if top_k is not None else self.config.default_top_k
        scoped_query = f"session:{session_id} {query}".strip()
        return await self.memory.recall(
            scoped_query,
            top_k=k,
            user_id=self._session_user_id(session_id),
        )

    async def context_for(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> str:
        """Return formatted memory context for a session/query."""
        k = top_k if top_k is not None else self.config.default_top_k
        scoped_query = f"session:{session_id} {query}".strip()
        return await self.memory.context_for(
            scoped_query,
            top_k=k,
            user_id=self._session_user_id(session_id),
        )

    @staticmethod
    def _session_user_id(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def _serialize_turn(
        *,
        session_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any],
    ) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        header = [
            f"session:{session_id}",
            f"role:{role}",
            f"timestamp:{ts}",
        ]
        meta_text = ""
        if metadata:
            pairs = [f"{k}={v}" for k, v in metadata.items()]
            meta_text = "\nmetadata: " + ", ".join(pairs)

        return "\n".join(header) + "\ntext: " + text.strip() + meta_text
