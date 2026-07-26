"""Background write queue for asynchronous memory updates."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clara.extraction.extractor import ExtractedFact
from clara.retrieval.cache import MemoryCache
from clara.retrieval.embeddings import EmbeddingEngine
from clara.retrieval.engine import LanceRetrievalEngine, RetrievalEngine
from clara.update.engine import MemoryUpdateEngine

logger = logging.getLogger(__name__)

# Bounded by default: an unbounded queue grows without limit under producer
# overload and hides backpressure from callers.
DEFAULT_MAX_QUEUE_SIZE = 256

# One retry after the first failure — transient SQLite lock contention is the
# realistic recoverable case; anything persistent lands in the dropped log.
MAX_ATTEMPTS = 2


class BackgroundWriter:
    """Process extracted facts on a background worker.

    Failure semantics: each fact is attempted :data:`MAX_ATTEMPTS` times in
    its own transaction. Facts that still fail are recorded on
    :attr:`dropped` (a bounded in-memory list surfaced via :meth:`stats`)
    and logged at ERROR — callers of ``remember(wait=False)`` were already
    told "queued", so this is the only channel that makes the loss visible.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_engine: EmbeddingEngine,
        *,
        cache: MemoryCache | None = None,
        lance_engine: LanceRetrievalEngine | None = None,
        similarity_threshold: float | None = None,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_engine = embedding_engine
        self._cache = cache
        self._lance_engine = lance_engine
        self._similarity_threshold = similarity_threshold
        self._queue: asyncio.Queue[tuple[ExtractedFact, str | None] | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False
        self.dropped: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        # Refuse new work first so nothing can be enqueued behind the
        # sentinel and silently never processed (the old shutdown race).
        self._stopping = True
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    async def enqueue(self, fact: ExtractedFact, user_id: str | None = None) -> None:
        if self._stopping:
            raise RuntimeError("BackgroundWriter is stopping; enqueue refused.")
        if self._worker_task is None or self._worker_task.done():
            self.start()
        await self._queue.put((fact, user_id))

    async def join(self) -> None:
        await self._queue.join()

    def stats(self) -> dict[str, Any]:
        """Queue depth and dropped-fact count, for health reporting."""
        return {
            "queued": self._queue.qsize(),
            "dropped": len(self.dropped),
            "running": self._worker_task is not None and not self._worker_task.done(),
        }

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            fact, user_id = item
            try:
                await self._process_with_retry(fact, user_id)
            finally:
                self._queue.task_done()

    async def _process_with_retry(
        self, fact: ExtractedFact, user_id: str | None
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                await self._process_one(fact, user_id)
                return
            except Exception as exc:  # noqa: BLE001 — recorded, not hidden
                last_error = exc
                logger.warning(
                    "Background memory update failed (attempt %d/%d): %s",
                    attempt, MAX_ATTEMPTS, exc,
                )
        logger.error(
            "Dropping fact after %d attempts: %s %s %s — %s",
            MAX_ATTEMPTS, fact.subject, fact.relation, fact.object, last_error,
        )
        if len(self.dropped) < 100:  # bounded; oldest failures are enough signal
            self.dropped.append({
                "subject": fact.subject,
                "relation": fact.relation,
                "object": fact.object,
                "user_id": user_id,
                "error": str(last_error),
            })

    async def _process_one(self, fact: ExtractedFact, user_id: str | None) -> None:
        async with self._session_factory() as session:
            session.sync_session.info.setdefault("_clara_embedding_engine", self._embedding_engine)
            if self._cache is not None:
                session.sync_session.info.setdefault("_clara_cache", self._cache)
            if self._lance_engine is not None:
                session.sync_session.info.setdefault("_lance_engine", self._lance_engine)
            async with session.begin():
                retriever = RetrievalEngine(
                    session,
                    self._embedding_engine,
                    cache=self._cache,
                    lance_engine=self._lance_engine,
                )
                updater_kwargs: dict[str, Any] = {"cache": self._cache}
                if self._similarity_threshold is not None:
                    # Match the wait=True path: the configured threshold must
                    # apply regardless of whether the write is synchronous.
                    updater_kwargs["similarity_threshold"] = self._similarity_threshold
                updater = MemoryUpdateEngine(
                    session,
                    self._embedding_engine,
                    retriever,
                    **updater_kwargs,
                )
                await updater.process(fact, user_id=user_id)
