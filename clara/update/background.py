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
from clara.retrieval.engine import RetrievalEngine
from clara.update.engine import MemoryUpdateEngine

logger = logging.getLogger(__name__)


class BackgroundWriter:
    """Process extracted facts on a background worker."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_engine: EmbeddingEngine,
        *,
        cache: MemoryCache | None = None,
        max_queue_size: int = 0,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_engine = embedding_engine
        self._cache = cache
        self._queue: asyncio.Queue[tuple[ExtractedFact, str | None] | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    async def enqueue(self, fact: ExtractedFact, user_id: str | None = None) -> None:
        if self._worker_task is None or self._worker_task.done():
            self.start()
        await self._queue.put((fact, user_id))

    async def join(self) -> None:
        await self._queue.join()

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            fact, user_id = item
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        retriever = RetrievalEngine(
                            session,
                            self._embedding_engine,
                            cache=self._cache,
                        )
                        updater = MemoryUpdateEngine(
                            session,
                            self._embedding_engine,
                            retriever,
                            cache=self._cache,
                        )
                        await updater.process(fact, user_id=user_id)
            except Exception:
                logger.exception("Background memory update failed")
            finally:
                self._queue.task_done()
