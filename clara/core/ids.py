"""
Canonical identifier form for CLARA's UUID columns.

SQLite stores ``Uuid(as_uuid=True)`` columns as 32-character dashless hex, but
what Python hands back depends on how the row was read:

    ORM (Mapped[uuid.UUID])   -> uuid.UUID       -> str() gives the DASHED form
    raw SQL (sa_text/sqlite3) -> str             -> str() gives the DASHLESS form

Both spellings were being written into ``graph_edges.belief_id`` -- the live
projection reads the ORM record, the rebuild path reads raw rows -- so a single
store ended up holding both, and joins against ``memories.memory_id`` (always
dashless) had to normalize both sides at query time to find everything. That
made the join unindexable, and made it silently lossy for anyone who "optimised"
the normalization away.

Route every identifier through :func:`canonical_id` at the point of writing so
the stored value matches ``memories.memory_id`` and the join is plain equality.
"""

from __future__ import annotations

from typing import Any


def canonical_id(value: Any) -> str:
    """Return *value* as 32-char dashless hex, matching how SQLite stores it.

    Accepts a ``uuid.UUID`` (uses ``.hex``) or a string in either spelling.
    """
    hex_attr = getattr(value, "hex", None)
    if isinstance(hex_attr, str):
        return hex_attr
    return str(value).replace("-", "")
