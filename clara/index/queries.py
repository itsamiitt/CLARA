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
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from clara.index import indexer

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


# Files a JS/TS toolchain loads by name rather than by import. These are
# conventions the tools themselves define, not guesses about what looks
# unimportant: a test runner collects *.test.*, a bundler reads *.config.*,
# and Next.js routes app/**/page.tsx by filename.
_TEST_MARKERS = (".test.", ".spec.")
_TEST_DIRS = ("__tests__", "__mocks__", "e2e", "cypress")
_ROUTE_FILES = frozenset(
    {"page", "layout", "route", "template", "loading", "error", "not-found",
     "default", "middleware", "instrumentation"}
)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def _normalise(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _collect_paths(value: object, into: set[str]) -> None:
    """Every string that names a script file, at any depth of a JSON value.

    package.json's ``exports`` is a nested map of conditions to paths, and its
    shape varies by package. Walking it is more honest than assuming one form.
    """
    if isinstance(value, str):
        if value.endswith(_SCRIPT_SUFFIXES):
            into.add(_normalise(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_paths(item, into)
    elif isinstance(value, list):
        for item in value:
            _collect_paths(item, into)


_HTML_SRC = re.compile(r"""<script[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.I)


def _find_declaring_files(repo_root: Path) -> dict[str, list[tuple[Path, str]]]:
    """The files that declare entry points, bucketed, in one pruned walk.

    Pruned rather than filtered afterwards, for the same reason walk_repo is:
    node_modules holds thousands of package.json files, none of which describe
    *this* project, and rglob would read every one of them. One walk rather
    than three, because the tree is the expensive part, not the matching.
    """
    buckets: dict[str, list[tuple[Path, str]]] = {
        "package": [], "html": [], "manifest": []
    }
    for current, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in indexer.SKIP_DIRS]
        directory = Path(current)
        rel = directory.relative_to(repo_root).as_posix()
        rel = "" if rel == "." else rel
        for name in filenames:
            if name == "package.json":
                buckets["package"].append((directory / name, rel))
            elif name == "manifest.json":
                buckets["manifest"].append((directory / name, rel))
            elif name.endswith((".html", ".htm")):
                # Any HTML page, not just index.html: a Chrome extension's
                # popup.html and sidepanel.html load their scripts this way,
                # and scanning only index.html left 42 live files looking dead.
                buckets["html"].append((directory / name, rel))
    return buckets


def script_entrypoints(repo_root: Path) -> set[str]:
    """Files the project *declares* as entry points, for JS/TS repos.

    Evidence only, each from a file the project maintains:

    * every package.json's ``main``/``module``/``browser``/``bin``/``exports``,
      and any script file named in a ``scripts`` command (``tsx server/index.ts``
      means server/index.ts is run, not imported);
    * ``<script src=...>`` in any index.html -- the Vite entry convention;
    * a Chrome extension manifest.json, whose background and content scripts
      are loaded by the browser.

    Paths are resolved relative to the manifest that names them, so a monorepo
    package declaring "src/main.tsx" resolves under its own directory.
    """
    found: set[str] = set()
    declaring = _find_declaring_files(repo_root)
    for manifest, rel_dir in declaring["package"]:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        local: set[str] = set()
        for key in ("main", "module", "browser", "bin", "exports", "types"):
            _collect_paths(data.get(key), local)
        scripts = data.get("scripts")
        if isinstance(scripts, dict):
            for command in scripts.values():
                if not isinstance(command, str):
                    continue
                for token in re.split(r"[\s;&|'\"]+", command):
                    if token.endswith(_SCRIPT_SUFFIXES):
                        local.add(_normalise(token))
        found |= {f"{rel_dir}/{p}" if rel_dir else p for p in local}

    for page, rel_dir in declaring["html"]:
        try:
            html = page.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for src in _HTML_SRC.findall(html):
            if src.startswith(("http://", "https://", "//")):
                continue
            path = _normalise(src)
            found.add(f"{rel_dir}/{path}" if rel_dir else path)

    for manifest, rel_dir in declaring["manifest"]:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "manifest_version" not in data:
            continue
        local = set()
        for key in ("background", "content_scripts", "web_accessible_resources",
                    "action", "chrome_url_overrides"):
            _collect_paths(data.get(key), local)
        found |= {f"{rel_dir}/{p}" if rel_dir else p for p in local}

    return found


def is_conventional_entry(rel_path: str) -> bool:
    """True for files a toolchain loads by filename convention.

    Named conventions with defined meanings, not a "looks unused" heuristic:
    test files a runner collects, tool config files read by name, and
    framework route files. Each is a real reason a file has no importer.
    """
    parts = rel_path.split("/")
    name = parts[-1]
    stem = name.split(".")[0]

    if any(marker in name for marker in _TEST_MARKERS):
        return True
    if any(part in _TEST_DIRS for part in parts[:-1]):
        return True
    # foo.config.ts, vite.config.js -- read by the tool, by name.
    if ".config." in name:
        return True
    # A dotfile that is itself a script is a tool's config: .eslintrc.cjs,
    # .dependency-cruiser.cjs. Nothing imports these; the tool loads them.
    if name.startswith(".") and name.endswith(_SCRIPT_SUFFIXES):
        return True
    # Next.js app router and pages router: routed by path, never imported.
    if stem in _ROUTE_FILES and any(p in ("app", "pages", "src") for p in parts[:-1]):
        return True
    # Everything under a pages/ or api/ directory is routed by its path.
    return "pages" in parts[:-1] or "api" in parts[:-1]


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
    # For JS/TS the qualified name is the repo-relative path, so entry points
    # are matched as paths. Without this, a third of a real Node repo reported
    # as unused -- vite.config.ts, every *.test.ts, every framework route.
    scripts = script_entrypoints(repo_root) if repo_root else set()

    found: list[str] = []
    for name, attributes in rows:
        if name in declared or name in scripts:
            continue
        if roots and any(name == r or name.startswith(f"{r}.") for r in roots):
            continue
        # Script suffix, not "contains a slash": a JS qualified name is a path
        # and a Python one is dotted, but a *root-level* config like
        # eslint.config.js has no slash and was wrongly skipped by that test.
        if name.endswith(_SCRIPT_SUFFIXES) and is_conventional_entry(name):
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
