"""
CLARA — bounded retry for SQLite write contention.

Two Claude Code sessions share one store file; WAL serializes writers, and a
write lock held longer than the busy timeout surfaces as an
``OperationalError("database is locked")``. The save path must not hand that
raw traceback to the MCP caller: retry with backoff, then fail with an
actionable message.

Only lock/busy errors are retried — every other error propagates untouched.
Each attempt must be a *whole* transaction (a fresh session), so a failed
attempt has committed nothing and the retry cannot duplicate work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_LOCK_MARKERS = ("database is locked", "database table is locked", "busy")
_DELAYS_S = (0.25, 1.0, 4.0)


def _is_lock_error(exc: OperationalError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _LOCK_MARKERS)


async def with_sqlite_retry(
    op: Callable[[], Awaitable[T]],
    *,
    what: str = "write",
) -> T:
    """Run *op* (a coroutine factory opening its own transaction), retrying
    lock/busy failures with 0.25s/1s/4s backoff. On final failure raise a
    clean ``RuntimeError`` naming the operation."""
    last: OperationalError | None = None
    for attempt, delay in enumerate((0.0, *_DELAYS_S)):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await op()
        except OperationalError as exc:
            if not _is_lock_error(exc):
                raise
            last = exc
            logger.warning(
                "memory store locked during %s (attempt %d): %s", what, attempt + 1, exc
            )
    raise RuntimeError(
        f"memory store is busy (locked through {sum(_DELAYS_S):.0f}s of retries); "
        f"the {what} was not persisted — retry shortly"
    ) from last
