"""
CLARA code index — change journal, index state, and the code graph.

Stdlib-only (sqlite3), like clara/docs and clara/fastpath: this runs from hooks
and background passes where importing SQLAlchemy would dominate the cost.

See docs/plans/memory-systems-plan.md §3.1, §3.2, §4.2. The contract that makes
this affordable on a real repo is: **process only journalled paths, gated by
content hash**. A full rescan happens on first setup, on an explicit rebuild,
or never.
"""

from clara.index.journal import (
    JournalEntry,
    claim_batch,
    complete,
    enqueue,
    pending_count,
    release_stale_claims,
)
from clara.index.state import (
    forget_path,
    is_unchanged,
    record_indexed,
)

__all__ = [
    "JournalEntry",
    "claim_batch",
    "complete",
    "enqueue",
    "forget_path",
    "is_unchanged",
    "pending_count",
    "record_indexed",
    "release_stale_claims",
]
