"""
CLARA — Main Agent Interface

Provides :class:`ClaraMemory`, the top-level façade that wires every CLARA
subsystem together into three ergonomic async methods:

* :meth:`remember` — extract facts from text and store them in memory.
* :meth:`recall`   — retrieve relevant memories for a query.
* :meth:`context_for` — build a formatted context block for LLM injection.

Usage::

    agent = await ClaraMemory.create(
        db_url="sqlite+aiosqlite:///clara.db",
        lance_path="./clara_vectors",
        embedding_backend="local",
        llm_provider="openai",
    )
    await agent.remember("I switched from Python to Rust for systems work.")
    results = await agent.recall("What language does the user prefer?")
    ctx     = await agent.context_for("Help me deploy my service.")
    await agent.close()
"""

from __future__ import annotations

import logging
import os
from sqlalchemy.exc import OperationalError
from typing import Any, Sequence

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.extraction.extractor import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    ExtractedFact,
    FactExtractor,
)
from clara.interaction import InteractionLayer
from clara.memory.belief import BeliefMemory
from clara.reasoning.engine import ReasoningEngine
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import (
    EmbeddingEngine,
    ENV_OLLAMA_EMBED_MODEL,
    _EmbeddingBackend,
    _LocalBackend,
    _OpenAIBackend,
    _create_backend,
)
from clara.retrieval.engine import (
    DEFAULT_LANCE_PATH,
    LanceRetrievalEngine,
    RetrievalEngine,
    RetrievalResult,
    ScoredMemory,
)
from clara.scheduler.decay import DecayScheduler
from clara.update.background import BackgroundWriter
from clara.update.engine import MemoryUpdateEngine

logger = logging.getLogger(__name__)

SQLITE_BUSY_TIMEOUT_MS = 30_000


def _is_in_memory_sqlite(db_url: str) -> bool:
    return db_url in {"sqlite://", "sqlite+aiosqlite://"} or ":memory:" in db_url


def _make_engine(db_url: str) -> AsyncEngine:
    """Create a database engine with SQLite-specific concurrency tuning."""
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {"echo": False}
    if db_url.startswith("sqlite"):
        connect_args["timeout"] = SQLITE_BUSY_TIMEOUT_MS / 1000
        if _is_in_memory_sqlite(db_url):
            engine_kwargs["poolclass"] = StaticPool
        else:
            # File-backed SQLite handles bursty concurrent reads better when
            # sessions don't block on a small QueuePool.
            engine_kwargs["poolclass"] = NullPool

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    engine = create_async_engine(db_url, **engine_kwargs)

    if db_url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout = 30000")
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
            finally:
                cursor.close()

    return engine


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_belief(sm: ScoredMemory) -> str:
    """One-line summary of a belief for the context block."""
    c = sm.memory.content
    domain = c.get("domain")
    core = f"{c.get('subject', '?')} {c.get('relation', '?')} {c.get('object', '?')}"
    if c.get("is_negation"):
        core = f"not ({core})"
    line = f"- {core}"
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
            ``"sqlite+aiosqlite:///clara.db"`` or
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
        lance_engine: LanceRetrievalEngine | None = None,
        interaction_layer: InteractionLayer | None = None,
        cache: MemoryCache | None = None,
        background_writer: BackgroundWriter | None = None,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._embedding_engine = embedding_engine
        self._extractor = extractor
        self._decay_scheduler = decay_scheduler
        self._lance_engine = lance_engine or LanceRetrievalEngine.get_default()
        self._interaction_layer = interaction_layer or InteractionLayer()
        self._llm_provider = getattr(extractor, "_provider", "openai")
        self._llm_model = getattr(extractor, "_model", None)
        self._cache = cache
        self._background_writer = background_writer

    @classmethod
    async def create(
        cls,
        db_url: str = "sqlite+aiosqlite:///clara.db",
        embedding_backend: str = "openai",
        llm_provider: str = "openai",
        *,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_llm_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_embed_model: str = "nomic-embed-text",
        lance_path: str = "./clara_vectors",
        start_scheduler: bool = True,
        cache_url: str | None = None,
    ) -> ClaraMemory:
        """Async factory — creates the engine, tables, and all subsystems.

        Args:
            db_url: SQLAlchemy async DB URL.
            embedding_backend: ``"openai"``, ``"local"``, or ``"ollama"``.
            llm_provider: ``"openai"``, ``"anthropic"``, or ``"ollama"``.
            start_scheduler: Whether to start the decay scheduler.

        Returns:
            A fully-initialised :class:`ClaraMemory` instance.
        """
        # --- Database ---
        engine = _make_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        resolved_lance_path = lance_path
        if lance_path == DEFAULT_LANCE_PATH:
            resolved_lance_path = os.environ.get("CLARA_LANCE_PATH", lance_path)
        LanceRetrievalEngine.configure_default_path(resolved_lance_path)
        lance_engine = LanceRetrievalEngine.get_default()

        resolved_ollama_base_url = ollama_base_url
        if ollama_base_url == DEFAULT_OLLAMA_BASE_URL:
            resolved_ollama_base_url = os.environ.get(ENV_OLLAMA_BASE_URL, ollama_base_url)

        resolved_ollama_llm_model = ollama_llm_model
        if ollama_llm_model == DEFAULT_OLLAMA_MODEL:
            resolved_ollama_llm_model = os.environ.get(ENV_OLLAMA_MODEL, ollama_llm_model)

        resolved_ollama_embed_model = ollama_embed_model
        if ollama_embed_model == "nomic-embed-text":
            resolved_ollama_embed_model = os.environ.get(
                ENV_OLLAMA_EMBED_MODEL,
                ollama_embed_model,
            )

        os.environ[ENV_OLLAMA_BASE_URL] = resolved_ollama_base_url
        os.environ[ENV_OLLAMA_MODEL] = resolved_ollama_llm_model
        os.environ[ENV_OLLAMA_EMBED_MODEL] = resolved_ollama_embed_model

        # --- Embedding ---
        backend = _create_backend(
            embedding_backend,
            ollama_base_url=resolved_ollama_base_url,
            ollama_model=resolved_ollama_embed_model,
        )
        embedding_engine = EmbeddingEngine(backend)

        # --- Extraction ---
        extractor = FactExtractor(
            provider=llm_provider,
            model=resolved_ollama_llm_model if llm_provider == "ollama" else None,
            ollama_base_url=resolved_ollama_base_url,
        )
        cache = MemoryCache(cache_url) if cache_url else None
        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
            info={
                "_lance_engine": lance_engine,
                "_clara_embedding_engine": embedding_engine,
                "_clara_cache": cache,
            },
        )

        # --- Decay Scheduler ---
        decay_scheduler: DecayScheduler | None = None
        if start_scheduler:
            decay_scheduler = DecayScheduler(
                session_factory,
                embedding_engine=embedding_engine,
                llm_provider=llm_provider,
                llm_model=getattr(extractor, "_model", None),
            )
            decay_scheduler.start()

        instance = cls(
            engine=engine,
            session_factory=session_factory,
            embedding_engine=embedding_engine,
            extractor=extractor,
            decay_scheduler=decay_scheduler,
            lance_engine=lance_engine,
            interaction_layer=InteractionLayer(),
            cache=cache,
            background_writer=BackgroundWriter(
                session_factory,
                embedding_engine,
                cache=cache,
                lance_engine=lance_engine,
            ),
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

    async def remember(
        self,
        text: str,
        *,
        user_id: str | None = None,
        wait: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract facts from *text* and store them in the memory store.

        Args:
            text: Raw natural-language input.

        Returns:
            A list of dicts summarising each stored fact::

                [{"action": "created", "memory_id": "...", "conflict": False}, ...]
        """
        if not text or not text.strip():
            return []

        interaction = self._interaction_layer.receive(text, user_id=user_id)

        facts = self._extractor.extract(interaction.raw_text)
        if not facts:
            logger.debug("No facts extracted from text: %r", text[:120])
            return []

        if not wait:
            if self._background_writer is None:
                raise RuntimeError("Background writer is not configured.")
            for fact in facts:
                await self._background_writer.enqueue(fact, user_id=interaction.user_id)
            logger.info("Queued %d fact(s) for background processing", len(facts))
            return [
                {
                    "action": "queued",
                    "memory_id": None,
                    "conflict": False,
                    "superseded_id": None,
                }
                for _ in facts
            ]

        results: list[dict[str, Any]] = []

        async with self._session_factory() as session:
            self._bind_session_context(session)
            async with session.begin():
                update_engine = MemoryUpdateEngine(
                    session,
                    self._embedding_engine,
                    RetrievalEngine(
                        session,
                        self._embedding_engine,
                        cache=self._cache,
                        lance_engine=self._lance_engine,
                    ),
                    cache=self._cache,
                )

                for fact in facts:
                    outcome = await update_engine.process(
                        fact,
                        user_id=interaction.user_id,
                    )
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
        *,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """Retrieve the most relevant memories for *query*.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.

        Returns:
            A :class:`RetrievalResult` grouped by memory type.
        """
        async with self._session_factory() as session:
            self._bind_session_context(session)
            retriever = RetrievalEngine(
                session,
                self._embedding_engine,
                cache=self._cache,
                lance_engine=self._lance_engine,
            )
            result = await retriever.search(
                query,
                top_k=top_k,
                user_id=user_id,
                track_access=False,
            )

        await self._record_accesses_best_effort(result.all)
        return result

    async def context_for(
        self,
        query: str,
        *,
        top_k: int = 8,
        user_id: str | None = None,
    ) -> str:
        """Build a formatted context string for LLM injection.

        Performs a :meth:`recall`, then formats the results into the
        ``=== MEMORY CONTEXT ===`` block described in CONTEXT.md.

        Args:
            query: Natural language query.
            top_k: Maximum number of results to include.

        Returns:
            A multi-line string ready for inclusion in an LLM system prompt.
        """
        result = await self.recall(query, top_k=top_k, user_id=user_id)
        return format_context(result)

    async def interact(
        self,
        message: str,
        *,
        user_id: str | None = None,
        system_prompt: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Run the full reasoning loop over memory context and return a response."""
        interaction = self._interaction_layer.receive(message, user_id=user_id)

        async with self._session_factory() as session:
            self._bind_session_context(session)
            async with session.begin():
                reasoning = ReasoningEngine(
                    session,
                    self._embedding_engine,
                    self._extractor,
                    llm_provider=self._llm_provider,
                    llm_model=self._llm_model,
                    ollama_base_url=os.environ.get(ENV_OLLAMA_BASE_URL),
                    cache=self._cache,
                )
                response = await reasoning.respond(
                    interaction.raw_text,
                    user_id=interaction.user_id,
                    system_prompt=system_prompt,
                    top_k=top_k,
                )

        return {
            "response": response.text,
            "memory_context": response.memory_context,
            "facts_stored": [
                {
                    "action": item.action_taken.value,
                    "memory_id": str(item.memory_id) if item.memory_id else None,
                    "conflict": item.conflict_detected,
                    "superseded_id": (
                        str(item.superseded_id) if item.superseded_id else None
                    ),
                }
                for item in response.facts_stored
            ],
            "memories_used": [
                {
                    "memory_id": str(sm.memory.memory_id),
                    "memory_type": sm.memory.memory_type.value,
                    "score": sm.score,
                    "confidence": sm.confidence,
                }
                for sm in response.memories_used
            ],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the scheduler and dispose of the database engine."""
        if self._decay_scheduler is not None:
            self._decay_scheduler.shutdown(wait=False)
        if self._background_writer is not None:
            await self._background_writer.stop()
        if self._cache is not None:
            await self._cache.close()
        self._lance_engine.close()
        await self._engine.dispose()
        logger.info("ClaraMemory closed")

    async def _record_accesses_best_effort(
        self,
        scored_memories: Sequence[ScoredMemory],
    ) -> None:
        memory_ids = [sm.memory.memory_id for sm in scored_memories]
        if not memory_ids:
            return

        async with self._session_factory() as session:
            self._bind_session_context(session)
            retriever = RetrievalEngine(
                session,
                self._embedding_engine,
                cache=self._cache,
                lance_engine=self._lance_engine,
            )
            try:
                await retriever.record_accesses(memory_ids)
                await session.commit()
            except OperationalError:
                await session.rollback()
                if self._engine.dialect.name == "sqlite":
                    logger.debug(
                        "Skipping access-count update after SQLite lock contention",
                        exc_info=True,
                    )
                    return
                raise

    async def cache_health(self) -> dict[str, object]:
        if self._cache is None:
            return {"backend": "disabled", "ok": True}
        return await self._cache.health()

    def _bind_session_context(self, session: AsyncSession) -> None:
        session.sync_session.info.setdefault("_lance_engine", self._lance_engine)
        session.sync_session.info.setdefault("_clara_embedding_engine", self._embedding_engine)
        if self._cache is not None:
            session.sync_session.info.setdefault("_clara_cache", self._cache)
