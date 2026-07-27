"""
The indexer — journal entries in, code graph out.

Contract (docs/plans/memory-systems-plan.md §4.2): **process only journalled
paths, gated by content hash.** A file whose bytes are unchanged costs one
index_state lookup. A full walk happens on first setup or an explicit rebuild.

Incrementality (§4.3): when a file changes, its *outbound* edges are
invalidated and re-emitted from the fresh parse. Inbound edges — other modules
importing it — are left alone, because they belong to those files and their
bytes did not change. That is what keeps a one-file edit O(file) instead of
O(repo).

Node ids are content-addressed: ``blake2b(repo_id, kind, qualified_name)``.
Stable, so the same module gets the same id no matter which file mentions it
first, and joinable by plain equality — the belief graph's ``replace(CAST(...))``
join is not repeated here (migration 8).

Edges are invalidated, never deleted, matching the belief graph: a removed
import stays visible as history rather than vanishing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from clara.index import journal, pysource, state

# Only Python for now. The parser is stdlib `ast`; other languages need their
# own and are not pretended to work.
INDEXABLE_SUFFIXES = frozenset({".py"})

_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
        ".tox", ".eggs", "site-packages",
    }
)


@dataclass(slots=True)
class IndexResult:
    """What one indexing pass did. Counts, so callers can report honestly."""

    processed: int = 0
    skipped_unchanged: int = 0
    removed: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    edges_invalidated: int = 0
    syntax_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "skipped_unchanged": self.skipped_unchanged,
            "removed": self.removed,
            "nodes": self.nodes_written,
            "edges": self.edges_written,
            "edges_invalidated": self.edges_invalidated,
            "syntax_errors": self.syntax_errors,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def node_id(repo_id: str, kind: str, qualified_name: str) -> str:
    """Stable id for a code node. Same inputs, same id, any time."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repo_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(kind.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(qualified_name.encode("utf-8"))
    return digest.hexdigest()


def _upsert_node(
    conn: sqlite3.Connection,
    repo_id: str,
    *,
    kind: str,
    qualified_name: str,
    file_path: str | None,
    lang: str | None,
    span: dict[str, int] | None,
) -> str:
    """Insert or refresh one node; returns its id.

    A module referenced by an import before its own file is indexed is created
    with ``file_path`` NULL. When that file is indexed later the same id is
    reused and the path fills in, so the graph converges regardless of order.
    An existing path is never overwritten with NULL.
    """
    identifier = node_id(repo_id, kind, qualified_name)
    conn.execute(
        "INSERT INTO code_nodes "
        "(node_id, repo_id, kind, qualified_name, file_path, lang, span, "
        " attributes, status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '{}', 'active', ?) "
        "ON CONFLICT(repo_id, kind, qualified_name) DO UPDATE SET "
        "  file_path = coalesce(excluded.file_path, code_nodes.file_path), "
        "  lang = coalesce(excluded.lang, code_nodes.lang), "
        "  span = coalesce(excluded.span, code_nodes.span), "
        "  status = 'active', "
        "  updated_at = excluded.updated_at",
        (
            identifier, repo_id, kind, qualified_name, file_path, lang,
            json.dumps(span) if span else None, _now(),
        ),
    )
    return identifier


def _invalidate_outbound(conn: sqlite3.Connection, repo_id: str, src_ids: list[str]) -> int:
    """Retire every live edge leaving these nodes. Returns how many."""
    if not src_ids:
        return 0
    placeholders = ", ".join("?" for _ in src_ids)
    cursor = conn.execute(
        f"UPDATE code_edges SET invalid_at = ? "  # noqa: S608
        f"WHERE repo_id = ? AND invalid_at IS NULL AND src_id IN ({placeholders})",
        (_now(), repo_id, *src_ids),
    )
    return int(cursor.rowcount or 0)


def _node_ids_for_file(conn: sqlite3.Connection, repo_id: str, rel_path: str) -> list[str]:
    rows = conn.execute(
        "SELECT node_id FROM code_nodes WHERE repo_id = ? AND file_path = ?",
        (repo_id, rel_path),
    ).fetchall()
    return [r[0] for r in rows]


def _retire_file(conn: sqlite3.Connection, repo_id: str, rel_path: str) -> tuple[int, int]:
    """A deleted file: retire its nodes and their outbound edges."""
    ids = _node_ids_for_file(conn, repo_id, rel_path)
    invalidated = _invalidate_outbound(conn, repo_id, ids)
    if ids:
        placeholders = ", ".join("?" for _ in ids)
        conn.execute(
            f"UPDATE code_nodes SET status = 'removed', updated_at = ? "  # noqa: S608
            f"WHERE node_id IN ({placeholders})",
            (_now(), *ids),
        )
    state.forget_path(conn, repo_id, path=rel_path)
    return len(ids), invalidated


def _submodule_targets(
    repo_root: Path, target: str, names: tuple[str, ...]
) -> list[str]:
    """Names in ``from pkg import a, b`` that are themselves modules.

    ``from pkg import b`` is the ordinary way to import a submodule, and edging
    to ``pkg`` instead of ``pkg.b`` loses the dependency that matters. Whether
    ``b`` is a module or just a name in ``pkg/__init__.py`` is answered by the
    repo layout, which the parser cannot see and the indexer can -- so it is
    resolved here, against the filesystem, which makes it independent of the
    order files happen to be indexed in.
    """
    if not target or not names:
        return []
    package_dir = repo_root / Path(*target.split("."))
    resolved: list[str] = []
    for name in names:
        if (package_dir / f"{name}.py").is_file() or (
            package_dir / name / "__init__.py"
        ).is_file():
            resolved.append(f"{target}.{name}")
    return resolved


def index_file(
    conn: sqlite3.Connection,
    repo_id: str,
    repo_root: Path,
    rel_path: str,
    result: IndexResult,
    *,
    force: bool = False,
) -> None:
    """Index one file, skipping it when its bytes are unchanged."""
    absolute = repo_root / rel_path
    digest = state.content_hash(absolute)
    if digest is None:
        nodes, edges = _retire_file(conn, repo_id, rel_path)
        result.removed += 1
        result.edges_invalidated += edges
        return
    if not force and state.is_unchanged(
        conn, repo_id, path=rel_path, kind="file", current_hash=digest
    ):
        result.skipped_unchanged += 1
        return

    try:
        # utf-8-sig, not utf-8: a leading BOM is a character to ast.parse and
        # fails the whole file ("invalid non-printable character U+FEFF").
        # Windows-authored sources carry one routinely -- CLARA's own
        # integrations/openclaw_bridge.py does. Harmless no-op without a BOM.
        source = absolute.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        result.removed += 1
        return

    parsed = pysource.parse_module(rel_path, source)
    if parsed.syntax_error:
        result.syntax_errors += 1

    module_name = pysource.module_name_for(rel_path)
    module_node = _upsert_node(
        conn, repo_id, kind="module", qualified_name=module_name,
        file_path=rel_path, lang="python", span=None,
    )
    result.nodes_written += 1

    # Re-emitting is a replace, not an append: retire what this file used to
    # point at before writing what it points at now.
    result.edges_invalidated += _invalidate_outbound(
        conn, repo_id, _node_ids_for_file(conn, repo_id, rel_path)
    )

    for node in parsed.nodes:
        if node.kind == "module":
            continue
        _upsert_node(
            conn, repo_id, kind=node.kind, qualified_name=node.qualified_name,
            file_path=rel_path, lang="python",
            span={"start_line": node.start_line, "end_line": node.end_line},
        )
        result.nodes_written += 1

    seen: set[str] = set()
    targets: list[tuple[str, pysource.SourceImport]] = []
    for imported in parsed.imports:
        resolved = pysource.resolve_import(module_name, imported)
        if not resolved:
            continue
        submodules = _submodule_targets(repo_root, resolved, imported.names)
        # `from pkg import b` where b is a module means pkg.b, not pkg. Where
        # it is a function, the package itself is the dependency.
        for candidate in submodules or [resolved]:
            targets.append((candidate, imported))

    for target, imported in targets:
        if target in seen:
            continue
        seen.add(target)
        dst = _upsert_node(
            conn, repo_id, kind="module", qualified_name=target,
            file_path=None, lang=None, span=None,
        )
        conn.execute(
            "INSERT INTO code_edges "
            "(edge_id, repo_id, src_id, dst_id, relation, confidence, "
            " valid_from, metadata) VALUES (?, ?, ?, ?, 'imports', 1.0, ?, ?)",
            (
                uuid.uuid4().hex, repo_id, module_node, dst, _now(),
                json.dumps({
                    "line": imported.line,
                    "resolver": "static",
                    "deferred": imported.deferred,
                }),
            ),
        )
        result.edges_written += 1

    state.record_indexed(
        conn, repo_id, path=rel_path, kind="file",
        content_hash=digest, lang="python",
    )
    result.processed += 1


def walk_repo(repo_root: Path) -> list[str]:
    """Every indexable file, repo-relative, skipping vendored trees."""
    found: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix not in INDEXABLE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(repo_root)
        except ValueError:  # pragma: no cover - rglob guarantees this
            continue
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        found.append(relative.as_posix())
    return found


def index_repo(
    conn: sqlite3.Connection,
    repo_id: str,
    repo_root: Path,
    *,
    force: bool = False,
) -> IndexResult:
    """Full walk. First setup, or an explicit rebuild — not the steady state."""
    result = IndexResult()
    # One transaction for the whole walk. clara.db.migrations.open_db opens the
    # store in autocommit (isolation_level=None), which turns every INSERT into
    # its own fsync: measured 84.9 s for CLARA's own tree that way versus 0.7 s
    # inside a single transaction. Batching is the difference between a usable
    # command and one nobody runs twice.
    managed = not conn.in_transaction
    if managed:
        conn.execute("BEGIN")
    try:
        for rel_path in walk_repo(repo_root):
            index_file(conn, repo_id, repo_root, rel_path, result, force=force)
        if managed:
            conn.execute("COMMIT")
        else:
            conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return result


def drain_journal(
    conn: sqlite3.Connection,
    repo_id: str,
    repo_root: Path,
    *,
    worker: str,
    limit: int = 200,
) -> IndexResult:
    """Process one claimed batch of journalled changes.

    Entries are completed only after their work commits, so a crash between
    the two leaves them claimed, and the staleness sweep returns them to the
    queue rather than losing them.
    """
    result = IndexResult()
    batch = journal.claim_batch(conn, repo_id, worker=worker, limit=limit)
    if not batch:
        return result
    for entry in batch:
        if entry.change in ("git", "manifest") or not entry.path:
            continue
        if entry.change == "renamed" and entry.old_path:
            removed_nodes, invalidated = _retire_file(conn, repo_id, entry.old_path)
            result.edges_invalidated += invalidated
            result.removed += 1 if removed_nodes else 0
        if entry.change == "removed":
            _, invalidated = _retire_file(conn, repo_id, entry.path)
            result.edges_invalidated += invalidated
            result.removed += 1
            continue
        if Path(entry.path).suffix not in INDEXABLE_SUFFIXES:
            continue
        index_file(conn, repo_id, repo_root, entry.path, result)
    conn.commit()
    # Only after the work is durable: a crash between these two leaves the
    # entries claimed, and the staleness sweep returns them to the queue.
    journal.complete(conn, [e.seq for e in batch])
    return result
