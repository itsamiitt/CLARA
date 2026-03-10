"""
CLARA — Main Agent Interface

Provides :class:`ClaraMemory`, the top-level façade that wires every CLARA
subsystem together into three ergonomic async methods:

* :meth:`remember` — extract facts from text and store them in memory.
* :meth:`recall`   — retrieve relevant memories for a query.
* :meth:`context_for` — build a formatted context block for LLM injection.

Usage::

    agent = await ClaraMemory.create(
        db_url="postgresql+asyncpg://...",
        embedding_backend="openai",
        llm_provider="openai",
    )
    await agent.remember("I switched from Python to Rust for systems work.")
    results = await agent.recall("What language does the user prefer?")
    ctx     = await agent.context_for("Help me deploy my service.")
    await agent.close()
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import ExtractedFact, FactExtractor
from clara.memory.belief import BeliefMemory
from clara.retrieval.embeddings import (
    EmbeddingEngine,
    _EmbeddingBackend,
    _LocalBackend,
    _OpenAIBackend,
    _create_backend,
)
from clara.retrieval.engine import RetrievalEngine, RetrievalResult, ScoredMemory
from clara.scheduler.decay import DecayScheduler
from clara.update.engine import MemoryUpdateEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_belief(sm: ScoredMemory) -> str:
    """One-line summary of a belief for the context block."""
    c = sm.memory.content
    domain = c.get("domain")
    line = f"- {c.get('subject', '?')} {c.get('relation', '?')} {c.get('object', '?')}"
    line += f" (confidence: {sm.memory.confidence:.2f}"
    if domain:
        line += f", domain: {domain}"
    line += ")"
    return line


def _format_event(sm: ScoredMemory) -> str:
    c = sm.memory.content
    ts = sm.memory.created_at.strftime("%Y-%m-%d") if sm.memory.created_at else "?"
    desc = c.get("object", c.get("description", ""))
    subj = c.get("subject", "")
    rel = c.get("relation", "")
    return f"- {ts}: {subj} {rel} {desc}"


def _format_skill(sm: ScoredMemory) -> str:
    c = sm.memory.content
    name = c.get("name", c.get("object", "unnamed skill"))
    return f"- {name} (confidence: {sm.memory.confidence:.2f})"


def _format_world_model(sm: ScoredMemory) -> str:
    c = sm.memory.content
    parts = []
    for key in ("name", "subject", "object"):
        if key in c and c[key]:
            parts.append(c[key])
            break
    # Show properties if available
    props = c.get("properties", {})
    if props and isinstance(props, dict):
        prop_strs = [f"{k}: {v}" for k, v in props.items()]
        parts.append(" | ".join(prop_strs))
    elif c.get("relation") and c.get("object"):
        parts.append(f"{c.get('relation', '')} {c.get('object', '')}")
    return f"- {' | '.join(parts)}" if parts else "- (world model entry)"


def format_context(result: RetrievalResult) -> str:
    """Build the ``=== MEMORY CONTEXT ===`` block from a retrieval result.

    Follows the CONTEXT.md §Reasoning Engine context injection format.
    """
    sections: list[str] = ["=== MEMORY CONTEXT ===", ""]

    # [BELIEFS]
    sections.append("[BELIEFS]")
    if result.beliefs:
        for sm in result.beliefs:
            sections.append(_format_belief(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    # [WORLD MODEL]
    sections.append("[WORLD MODEL]")
    if result.world_model:
        for sm in result.world_model:
            sections.append(_format_world_model(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    # [RECENT EVENTS]
    sections.append("[RECENT EVENTS]")
    if result.events:
        for sm in result.events:
            sections.append(_format_event(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    # [RELEVANT SKILLS]
    sections.append("[RELEVANT SKILLS]")
    if result.skills:
        for sm in result.skills:
            sections.append(_format_skill(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("=== END MEMORY CONTEXT ===")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# ClaraMemory
# ---------------------------------------------------------------------------

class ClaraMemory:
    """Top-level CLARA agent interface.

    Wires together:

    * **FactExtractor** — LLM-based extraction
    * **EmbeddingEngine** — vector embeddings
    * **MemoryUpdateEngine** — similarity search + conflict resolution + writes
    * **RetrievalEngine** — ranked semantic retrieval
    * **DecayScheduler** — background confidence decay + pruning

    Use the async :meth:`create` class method to build an instance (it needs
    to ``await`` table creation for fresh databases).

    Parameters:
        db_url:
            SQLAlchemy async connection URL, e.g.
            ``"postgresql+asyncpg://user:pass@host/db"`` or
            ``"sqlite+aiosqlite://"`` for in-memory testing.
        embedding_backend:
            ``"openai"`` or ``"local"`` (sentence-transformers).
        llm_provider:
            ``"openai"`` or ``"anthropic"`` — used by the fact extractor.
        start_scheduler:
            If ``True`` (default), start the ``DecayScheduler`` on init.
            Set to ``False`` in tests when an event loop may not be running.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_engine: EmbeddingEngine,
        extractor: FactExtractor,
        decay_scheduler: DecayScheduler | None,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._embedding_engine = embedding_engine
        self._extractor = extractor
        self._decay_scheduler = decay_scheduler

    @classmethod
    async def create(
        cls,
        db_url: str,
        embedding_backend: str = "openai",
        llm_provider: str = "openai",
        *,
        start_scheduler: bool = True,
    ) -> ClaraMemory:
        """Async factory — creates the engine, tables, and all subsystems.

        Args:
            db_url: SQLAlchemy async DB URL.
            embedding_backend: ``"openai"`` or ``"local"``.
            llm_provider: ``"openai"`` or ``"anthropic"``.
            start_scheduler: Whether to start the decay scheduler.

        Returns:
            A fully-initialised :class:`ClaraMemory` instance.
        """
        # --- Database ---
        engine = create_async_engine(db_url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # --- Embedding ---
        backend = _create_backend(embedding_backend)
        embedding_engine = EmbeddingEngine(backend)

        # --- Extraction ---
        extractor = FactExtractor(provider=llm_provider)

        # --- Decay Scheduler ---
        decay_scheduler: DecayScheduler | None = None
        if start_scheduler:
            decay_scheduler = DecayScheduler(session_factory)
            decay_scheduler.start()

        instance = cls(
            engine=engine,
            session_factory=session_factory,
            embedding_engine=embedding_engine,
            extractor=extractor,
            decay_scheduler=decay_scheduler,
        )
        logger.info(
            "ClaraMemory initialised (db=%s, embeddings=%s, llm=%s, scheduler=%s)",
            db_url,
            embedding_backend,
            llm_provider,
            "on" if start_scheduler else "off",
        )
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def remember(self, text: str) -> list[dict[str, Any]]:
        """Extract facts from *text* and store them in the memory store.

        Args:
            text: Raw natural-language input.

        Returns:
            A list of dicts summarising each stored fact::

                [{"action": "created", "memory_id": "...", "conflict": False}, ...]
        """
        if not text or not text.strip():
            return []

        facts = self._extractor.extract(text)
        if not facts:
            logger.debug("No facts extracted from text: %r", text[:120])
            return []

        results: list[dict[str, Any]] = []

        async with self._session_factory() as session:
            async with session.begin():
                update_engine = MemoryUpdateEngine(
                    session,
                    self._embedding_engine,
                    RetrievalEngine(session, self._embedding_engine),
                )

                for fact in facts:
                    outcome = await update_engine.process(fact)
                    results.append({
                        "action": outcome.action_taken.value,
                        "memory_id": str(outcome.memory_id) if outcome.memory_id else None,
                        "conflict": outcome.conflict_detected,
                        "superseded_id": (
                            str(outcome.superseded_id) if outcome.superseded_id else None
                        ),
                    })

        logger.info("Remembered %d fact(s) from text (%d chars)", len(results), len(text))
        return results

    async def recall(
        self,
        query: str,
        top_k: int = 8,
    ) -> RetrievalResult:
        """Retrieve the most relevant memories for *query*.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.

        Returns:
            A :class:`RetrievalResult` grouped by memory type.
        """
        async with self._session_factory() as session:
            retriever = RetrievalEngine(session, self._embedding_engine)
            result = await retriever.search(query, top_k=top_k)
            await session.commit()
        return result

    async def context_for(self, query: str, *, top_k: int = 8) -> str:
        """Build a formatted context string for LLM injection.

        Performs a :meth:`recall`, then formats the results into the
        ``=== MEMORY CONTEXT ===`` block described in CONTEXT.md.

        Args:
            query: Natural language query.
            top_k: Maximum number of results to include.

        Returns:
            A multi-line string ready for inclusion in an LLM system prompt.
        """
        result = await self.recall(query, top_k=top_k)
        return format_context(result)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the scheduler and dispose of the database engine."""
        if self._decay_scheduler is not None:
            self._decay_scheduler.shutdown(wait=False)
        await self._engine.dispose()
        logger.info("ClaraMemory closed")
