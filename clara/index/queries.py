"""
Queries over the code graph (plan §7).

Every traversal here is **seed-restricted**: the walk starts at named seeds and
only ever touches their frontier. The belief graph did not do this and
window-ranked the entire edge table per query, which cost 1.4 s on a 100k-edge
store for a five-neighbour seed (audit P3). The two directions are separate
index-seekable branches rather than ``src = ? OR dst = ?``, which is not
seekable.

The walk is breadth-first in Python, one statement per level, *not* a recursive
CTE. The CTE version carried depth in the recursive row, so ``UNION``
deduplicated (node, depth) pairs rather than nodes and re-expanded every node
once per distinct path reaching it. That is invisible on a small Python tree
and quadratic on a real one: a depth-3 impact query on a 13,303-edge TypeScript
repo took 19.3 s, against 154 ms for the level-at-a-time walk returning
byte-identical results.

Depth is always bounded. An unbounded walk over a real repo's import graph is
a way to return the whole repo.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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


# Frontier chunking: SQLite's parameter limit is per-statement, and a hub
# module in a real repo has hundreds of importers. Chunks keep every statement
# well inside the limit no matter how wide the graph gets.
_FRONTIER_CHUNK = 400


def _walk(
    conn: sqlite3.Connection,
    repo_id: str,
    seed: str,
    *,
    join_col: str,
    next_col: str,
    depth: int,
    relations: tuple[str, ...],
) -> dict[str, int]:
    """Breadth-first reachability from *seed*: node id -> smallest depth.

    Deliberately a level-at-a-time loop rather than one recursive CTE. The CTE
    carried depth in the recursive row, so ``UNION`` deduplicated
    (node, depth) *pairs* rather than nodes, and every node reachable by more
    than one path was re-expanded once per path. On a real TypeScript repo
    (13,303 edges) a depth-3 impact query took 5.4 s that way.

    Visiting each node once, at the first depth that reaches it, makes the work
    proportional to the edges actually touched. Same answer -- first visit in
    breadth-first order *is* the minimum depth -- measured in milliseconds.
    """
    placeholders = ", ".join("?" for _ in relations)
    seen: dict[str, int] = {seed: 0}
    frontier = [seed]
    for level in range(1, depth + 1):
        following: list[str] = []
        for start in range(0, len(frontier), _FRONTIER_CHUNK):
            chunk = frontier[start : start + _FRONTIER_CHUNK]
            marks = ", ".join("?" for _ in chunk)
            # No DISTINCT: the `seen` map below already deduplicates, and
            # asking SQLite to do it too made it prefer an index that made
            # DISTINCT free over one that seeked the frontier -- 30 ms a chunk
            # against 16 ms for the same rows.
            rows = conn.execute(
                f"SELECT {next_col} FROM code_edges "  # noqa: S608
                f"WHERE repo_id = ? AND invalid_at IS NULL "
                f"AND relation IN ({placeholders}) AND {join_col} IN ({marks})",
                (repo_id, *relations, *chunk),
            ).fetchall()
            for (node,) in rows:
                if node not in seen:
                    seen[node] = level
                    following.append(node)
        if not following:
            break
        frontier = following
    del seen[seed]
    return seen


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
    join_col, next_col = ("src_id", "dst_id") if direction == "forward" else (
        "dst_id", "src_id"
    )
    reached = _walk(
        conn, repo_id, seed, join_col=join_col, next_col=next_col,
        depth=bounded, relations=relations,
    )
    if not reached:
        return []

    found: list[Dependency] = []
    ids = list(reached)
    for start in range(0, len(ids), _FRONTIER_CHUNK):
        chunk = ids[start : start + _FRONTIER_CHUNK]
        marks = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT node_id, qualified_name, kind, file_path FROM code_nodes "  # noqa: S608
            f"WHERE status = 'active' AND node_id IN ({marks})",
            chunk,
        ).fetchall()
        found.extend(
            Dependency(
                qualified_name=r[1], kind=r[2], file_path=r[3],
                depth=reached[r[0]],
            )
            for r in rows
        )
    # Nearest first, then by name -- the order the CTE produced, preserved so
    # callers and their tests see no change beyond the speed.
    found.sort(key=lambda d: (d.depth, d.qualified_name))
    return found[:limit]


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


def declared_entrypoints(repo_root: Path) -> set[str]:
    """Modules the project itself declares as entry points.

    Evidence, not heuristics. Two sources, both stated by the repo:

    * ``[project.scripts]`` in pyproject.toml -- ``clara = "clara.cli:main"``
      means clara.cli is run, not imported;
    * ``[tool.pytest.ini_options] testpaths`` -- pytest collects those trees by
      filename, so nothing imports them either.

    A project that declares neither gets an empty set and the unfiltered list,
    which is the honest answer rather than a guess.
    """
    manifest = repo_root / "pyproject.toml"
    try:
        raw = manifest.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        return set()
    try:
        data = tomllib.loads(raw)
    except ValueError:
        return set()

    entrypoints: set[str] = set()
    scripts = data.get("project", {}).get("scripts", {})
    if isinstance(scripts, dict):
        for target in scripts.values():
            if isinstance(target, str) and ":" in target:
                entrypoints.add(target.split(":", 1)[0].strip())
    return entrypoints


def test_roots(repo_root: Path) -> tuple[str, ...]:
    """Dotted prefixes pytest collects, from ``testpaths``."""
    manifest = repo_root / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, ModuleNotFoundError):
        return ()
    paths = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get(
        "testpaths", []
    )
    if isinstance(paths, str):
        paths = [paths]
    return tuple(
        str(p).strip("/").replace("/", ".") for p in paths if isinstance(p, str)
    )


def unused_modules(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    repo_root: Path | None = None,
    limit: int = 100,
) -> list[str]:
    """Modules in this repo that nothing imports and nothing runs.

    "Nothing imports it" alone is not a useful signal: a CLI main, a module
    with an ``if __name__ == "__main__"`` block, and every pytest module are
    legitimately unreferenced. Those are excluded using what the project
    declares about itself -- console scripts and testpaths from pyproject.toml,
    plus the main-guard recorded at index time -- so what remains is worth
    looking at.

    Still not a delete list: a module imported only by name (a plugin loaded
    from config, an entry point declared elsewhere) cannot be seen by a static
    import graph, and this says so rather than pretending otherwise.
    """
    rows = conn.execute(
        "SELECT n.qualified_name, n.attributes FROM code_nodes n "
        "WHERE n.repo_id = ? AND n.status = 'active' AND n.kind = 'module' "
        "AND n.file_path IS NOT NULL "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM code_edges e "
        "  WHERE e.repo_id = n.repo_id AND e.dst_id = n.node_id "
        "  AND e.invalid_at IS NULL AND e.relation = 'imports'"
        ") ORDER BY n.qualified_name",
        (repo_id,),
    ).fetchall()

    declared = declared_entrypoints(repo_root) if repo_root else set()
    roots = test_roots(repo_root) if repo_root else ()

    found: list[str] = []
    for name, attributes in rows:
        if name in declared:
            continue
        if roots and any(name == r or name.startswith(f"{r}.") for r in roots):
            continue
        try:
            attrs = json.loads(attributes or "{}")
        except ValueError:
            attrs = {}
        if attrs.get("entrypoint"):
            continue
        found.append(name)
        if len(found) >= limit:
            break
    return found


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
