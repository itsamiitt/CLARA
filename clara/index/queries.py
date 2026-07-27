"""
Queries over the code graph (plan §7).

Every traversal here is **seed-restricted**: the recursive CTE walks outward
from named seeds and only ever touches their frontier. The belief graph did
not do this and window-ranked the entire edge table per query, which cost
1.4 s on a 100k-edge store for a five-neighbour seed (audit P3). The plan
mandates not repeating it, so the shape below is the fixed one: reachability
first, and the two directions are separate index-seekable branches rather than
``src = ? OR dst = ?``, which is not seekable.

Depth is always bounded. An unbounded walk over a real repo's import graph is
a way to return the whole repo.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Relations that mean "A needs B" for dependency and dead-code purposes.
DEPENDENCY_RELATIONS = ("imports",)

DEFAULT_DEPTH = 3
MAX_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Dependency:
    qualified_name: str
    kind: str
    file_path: str | None
    depth: int


def _resolve_seed(
    conn: sqlite3.Connection, repo_id: str, name_or_path: str
) -> str | None:
    """Node id for a dotted module name or a repo-relative file path."""
    row = conn.execute(
        "SELECT node_id FROM code_nodes "
        "WHERE repo_id = ? AND status = 'active' AND qualified_name = ? "
        "ORDER BY CASE kind WHEN 'module' THEN 0 ELSE 1 END LIMIT 1",
        (repo_id, name_or_path),
    ).fetchone()
    if row is not None:
        return str(row[0])
    normalised = name_or_path.replace("\\", "/")
    row = conn.execute(
        "SELECT node_id FROM code_nodes "
        "WHERE repo_id = ? AND status = 'active' AND file_path = ? "
        "AND kind = 'module' LIMIT 1",
        (repo_id, normalised),
    ).fetchone()
    return str(row[0]) if row is not None else None


# Forward: what this module depends on. Reverse: what depends on it -- the
# impact set. Only the join column changes, so one query text serves both.
_WALK_SQL = """
WITH RECURSIVE reachable (node_id, depth) AS (
    SELECT :seed, 0
    UNION
    SELECT {next_col}, r.depth + 1
    FROM reachable r
    JOIN code_edges e ON e.{join_col} = r.node_id
    WHERE r.depth < :depth
      AND e.repo_id = :repo
      AND e.invalid_at IS NULL
      AND e.relation IN ({relations})
)
SELECT n.qualified_name, n.kind, n.file_path, min(r.depth) AS depth
FROM reachable r
JOIN code_nodes n ON n.node_id = r.node_id
WHERE r.depth > 0 AND n.status = 'active'
GROUP BY n.qualified_name, n.kind, n.file_path
ORDER BY depth, n.qualified_name
LIMIT :lim
"""


def dependencies(
    conn: sqlite3.Connection,
    repo_id: str,
    name_or_path: str,
    *,
    direction: str = "forward",
    depth: int = DEFAULT_DEPTH,
    limit: int = 200,
    relations: tuple[str, ...] = DEPENDENCY_RELATIONS,
) -> list[Dependency]:
    """Modules reachable from *name_or_path*.

    ``forward`` answers "what does this need"; ``reverse`` answers "what needs
    this" -- the impact set of a change.
    """
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be 'forward' or 'reverse'")
    seed = _resolve_seed(conn, repo_id, name_or_path)
    if seed is None:
        return []
    bounded = max(1, min(int(depth), MAX_DEPTH))
    join_col, next_col = ("src_id", "e.dst_id") if direction == "forward" else (
        "dst_id", "e.src_id"
    )
    placeholders = ", ".join(f":rel{i}" for i in range(len(relations)))
    sql = _WALK_SQL.format(
        join_col=join_col, next_col=next_col, relations=placeholders
    )
    params: dict[str, object] = {
        "seed": seed, "depth": bounded, "repo": repo_id, "lim": limit
    }
    for i, relation in enumerate(relations):
        params[f"rel{i}"] = relation
    rows = conn.execute(sql, params).fetchall()
    return [
        Dependency(qualified_name=r[0], kind=r[1], file_path=r[2], depth=int(r[3]))
        for r in rows
    ]


def impact(
    conn: sqlite3.Connection,
    repo_id: str,
    name_or_path: str,
    *,
    depth: int = DEFAULT_DEPTH,
    limit: int = 200,
) -> list[Dependency]:
    """What breaks if this changes — reverse dependencies, transitively."""
    return dependencies(
        conn, repo_id, name_or_path, direction="reverse", depth=depth, limit=limit
    )


def unused_modules(
    conn: sqlite3.Connection, repo_id: str, *, limit: int = 100
) -> list[str]:
    """Modules in this repo that nothing imports.

    Honest about what this is: a module with zero inbound ``imports`` edges.
    That is **not** the same as dead code. Entry points -- a CLI main, a
    pytest module, a plugin loaded by name -- are legitimately unreferenced,
    and the plan's design has entrypoint facts coming from Project Memory to
    filter them out. That part is not built, so callers must treat this as a
    list to look at rather than a list to delete.
    """
    rows = conn.execute(
        "SELECT n.qualified_name FROM code_nodes n "
        "WHERE n.repo_id = ? AND n.status = 'active' AND n.kind = 'module' "
        "AND n.file_path IS NOT NULL "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM code_edges e "
        "  WHERE e.repo_id = n.repo_id AND e.dst_id = n.node_id "
        "  AND e.invalid_at IS NULL AND e.relation = 'imports'"
        ") ORDER BY n.qualified_name LIMIT ?",
        (repo_id, limit),
    ).fetchall()
    return [r[0] for r in rows]


def find_cycles(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    max_length: int = 6,
    limit: int = 20,
) -> list[list[str]]:
    """Import cycles, as lists of module names returning to their start.

    Walks from each module that has both inbound and outbound imports, bounded
    by *max_length*. Cycles are reported once, keyed on their member set, so
    A->B->A and B->A->B are one finding.
    """
    candidates = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT n.node_id FROM code_nodes n "
            "JOIN code_edges out_e ON out_e.src_id = n.node_id "
            "  AND out_e.invalid_at IS NULL AND out_e.relation = 'imports' "
            "JOIN code_edges in_e ON in_e.dst_id = n.node_id "
            "  AND in_e.invalid_at IS NULL AND in_e.relation = 'imports' "
            "WHERE n.repo_id = ? AND n.status = 'active' AND n.file_path IS NOT NULL",
            (repo_id,),
        ).fetchall()
    ]
    if not candidates:
        return []

    names = {
        node_id: name
        for node_id, name in conn.execute(
            "SELECT node_id, qualified_name FROM code_nodes WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
    }
    outgoing: dict[str, list[str]] = {}
    for src, dst in conn.execute(
        "SELECT src_id, dst_id FROM code_edges "
        "WHERE repo_id = ? AND invalid_at IS NULL AND relation = 'imports'",
        (repo_id,),
    ).fetchall():
        outgoing.setdefault(src, []).append(dst)

    seen: set[frozenset[str]] = set()
    found: list[list[str]] = []
    internal = set(candidates)

    def walk(start: str, node: str, path: list[str]) -> None:
        if len(found) >= limit or len(path) > max_length:
            return
        for nxt in outgoing.get(node, ()):
            if nxt == start and len(path) > 1:
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    found.append([names.get(p, p) for p in [*path, start]])
                continue
            if nxt in internal and nxt not in path:
                walk(start, nxt, [*path, nxt])

    for candidate in candidates:
        if len(found) >= limit:
            break
        walk(candidate, candidate, [candidate])
    return found
