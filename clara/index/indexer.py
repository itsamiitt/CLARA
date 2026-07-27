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

Languages: Python through the standard library's ``ast``, JavaScript and
TypeScript through the scanner in ``jssource``. A Python module's qualified
name is its dotted import path; a JS module has no such thing, so its
repo-relative path *is* its name. Both converge on the same node ids either
way, which is what lets an import edge written before the target file is
indexed join up to it afterwards.

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
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from clara.index import journal, jssource, pysource, state

# Python via stdlib `ast`; JS/TS via the scanner in jssource. Nothing else is
# claimed to work -- a .go or .rb file is simply not indexed.
PYTHON_SUFFIXES = frozenset({".py"})
SCRIPT_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
INDEXABLE_SUFFIXES = PYTHON_SUFFIXES | SCRIPT_SUFFIXES

# Declaration files describe types for code that already has a module node.
# Indexing them doubles the node count and adds no edge anyone asks about.
_SKIP_FILES = (".d.ts",)

# How many written files the apply phase commits per transaction. With
# planning already lock-free, this bounds the write lock's hold time per
# batch; see index_repo.
COMMIT_EVERY_FILES = 200

# Pause after each batch commit so a writer waiting on the lock can take it
# before the next batch begins. Committing alone is not a handoff: the next
# BEGIN retakes the lock in microseconds, and SQLite's busy handler polls with
# backoff that reaches ~100 ms between attempts, so a waiter sleeps straight
# through a smaller gap -- a 20 ms yield still left a cross-process writer
# waiting 7.1 s. The yield must exceed the waiter's poll interval to guarantee
# one poll lands inside it.
COMMIT_YIELD_SECONDS = 0.15

SKIP_DIRS = frozenset(
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
    attributes: dict[str, object] | None = None,
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
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?) "
        "ON CONFLICT(repo_id, kind, qualified_name) DO UPDATE SET "
        "  file_path = coalesce(excluded.file_path, code_nodes.file_path), "
        "  lang = coalesce(excluded.lang, code_nodes.lang), "
        "  span = coalesce(excluded.span, code_nodes.span), "
        "  attributes = excluded.attributes, "
        "  status = 'active', "
        "  updated_at = excluded.updated_at",
        (
            identifier, repo_id, kind, qualified_name, file_path, lang,
            json.dumps(span) if span else None,
            json.dumps(attributes or {}), _now(),
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


def _ancestor_packages(repo_root: Path, target: str) -> list[str]:
    """Packages implicitly imported alongside *target*.

    ``import a.b.c`` executes ``a/__init__.py`` and ``a/b/__init__.py`` -- that
    is a real dependency, not bookkeeping. Without these edges every package
    __init__ looks unreferenced: CLARA's own tree reported 83 "unused" modules,
    most of them packages that are imported constantly.

    Only packages that exist in this repo are emitted; ``os.path`` does not
    invent an ``os`` node here beyond what the import itself already creates.
    """
    parts = target.split(".")
    ancestors: list[str] = []
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if (repo_root / Path(*parts[:i]) / "__init__.py").is_file():
            ancestors.append(prefix)
    return ancestors


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


def _lang_for(rel_path: str) -> str:
    """Language label from the extension, for the ``lang`` column."""
    suffix = Path(rel_path).suffix.lower()
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in (".ts", ".tsx"):
        return "typescript"
    return "javascript"


@dataclass(slots=True)
class _NodeSpec:
    """One def/class node a plan will upsert."""

    kind: str
    qualified_name: str
    span: dict[str, int] | None


@dataclass(slots=True)
class _EdgeSpec:
    """One import edge a plan will write, by target qualified name."""

    target: str
    metadata: dict[str, object]


@dataclass(slots=True)
class _FilePlan:
    """Everything indexing one file will write, decided without writing.

    Planning (stat, read, hash, parse, resolve) is the expensive part of an
    index pass and needs no lock; applying is cheap and does. Separating them
    is what caps how long a background index can block an interactive
    memory_save -- see index_repo.

    ``action`` is one of:
      * ``skip``    -- stat signature matched; nothing to do at all;
      * ``refresh`` -- bytes unchanged but the stat moved; re-record only;
      * ``retire``  -- file unreadable or gone; retire its nodes;
      * ``index``   -- parse results in ``nodes`` and ``edges``.
    """

    rel_path: str
    action: str
    lang: str | None = None
    digest: str | None = None
    signature: tuple[int, int] | None = None
    module_name: str | None = None
    attributes: dict[str, object] | None = None
    nodes: list[_NodeSpec] = field(default_factory=list)
    edges: list[_EdgeSpec] = field(default_factory=list)
    syntax_error: bool = False


def _plan_script(
    repo_root: Path,
    rel_path: str,
    source: str,
    aliases: dict[str, list[str]],
    plan: _FilePlan,
) -> None:
    """Fill *plan* with one JS/TS file's edges.

    A JS module has no dotted name, so the repo-relative path *is* the
    qualified name. Resolution decides the other end: a specifier that lands on
    a file in this repo edges to that file's own node -- the same id the file
    gets when it is indexed itself, so the graph joins up regardless of order.
    Anything else (``react``, ``node:fs``, a bundled SDK's dangling relative
    path) becomes a node with no file_path, which is what "external" means here.
    """
    parsed = jssource.parse_script(rel_path, source)
    seen: set[str] = set()
    for specifier in parsed.specifiers:
        target = jssource.resolve_specifier(repo_root, rel_path, specifier, aliases)
        qualified = target or specifier
        # A module importing itself is not an edge worth recording, and would
        # show up as a one-node cycle in find_cycles.
        if qualified in seen or qualified == rel_path:
            continue
        seen.add(qualified)
        plan.edges.append(_EdgeSpec(qualified, {
            "specifier": specifier,
            "resolver": "static",
            "external": target is None,
            "deferred": specifier in parsed.deferred,
        }))


def _plan_python(
    repo_root: Path, rel_path: str, source: str, plan: _FilePlan
) -> None:
    """Fill *plan* with one Python file's nodes and edges."""
    parsed = pysource.parse_module(rel_path, source)
    plan.syntax_error = parsed.syntax_error is not None
    module_name = pysource.module_name_for(rel_path)
    plan.module_name = module_name
    plan.attributes = {"entrypoint": True} if parsed.has_main_guard else {}

    for node in parsed.nodes:
        if node.kind == "module":
            continue
        plan.nodes.append(_NodeSpec(
            node.kind, node.qualified_name,
            {"start_line": node.start_line, "end_line": node.end_line},
        ))

    seen: set[str] = set()
    targets: list[tuple[str, pysource.SourceImport]] = []
    for imported in parsed.imports:
        resolved = pysource.resolve_import(
            module_name, imported, is_package=parsed.is_package
        )
        if not resolved:
            continue
        submodules = _submodule_targets(repo_root, resolved, imported.names)
        # `from pkg import b` where b is a module means pkg.b, not pkg. Where
        # it is a function, the package itself is the dependency.
        for candidate in submodules or [resolved]:
            targets.append((candidate, imported))
            # ...and every package on the way to it, which Python executes.
            for ancestor in _ancestor_packages(repo_root, candidate):
                targets.append((ancestor, imported))

    for target, imported in targets:
        if target in seen:
            continue
        seen.add(target)
        plan.edges.append(_EdgeSpec(target, {
            "line": imported.line,
            "resolver": "static",
            "deferred": imported.deferred,
        }))


def _plan_file(
    conn: sqlite3.Connection,
    repo_id: str,
    repo_root: Path,
    rel_path: str,
    *,
    force: bool = False,
    aliases: dict[str, list[str]] | None = None,
) -> _FilePlan:
    """Decide what indexing this file requires. Reads only; writes nothing.

    The stat/read/hash/parse work here is nearly all of an index pass's time
    and needs no lock at all -- WAL readers block nobody. Keeping this phase
    write-free is what lets index_repo hold its transaction only for the brief
    apply phase; tests/test_index_twophase.py pins the property with a
    connection that refuses writes.
    """
    absolute = repo_root / rel_path
    lang = _lang_for(rel_path)

    # Cheapest check first. A stat that matches the recorded size and mtime
    # means the file need not be read at all -- reading every file to hash it
    # was the entire cost of a no-op re-index (8.4 s on 5,000 files).
    signature = state.stat_signature(absolute)
    if not force and state.is_unchanged_by_stat(
        conn, repo_id, path=rel_path, kind="file", signature=signature
    ):
        return _FilePlan(rel_path=rel_path, action="skip")

    try:
        raw = absolute.read_bytes()
    except OSError:
        return _FilePlan(rel_path=rel_path, action="retire")

    # One read, not two: hash the bytes already in hand rather than streaming
    # the file a second time. That was 43% of first-index time.
    digest = state.hash_bytes(raw)
    if not force and state.is_unchanged(
        conn, repo_id, path=rel_path, kind="file", current_hash=digest
    ):
        # Bytes are the same but the stat moved (a touch, or a checkout that
        # rewrote the file identically). Record the new signature so the next
        # pass takes the cheap path again.
        return _FilePlan(
            rel_path=rel_path, action="refresh", lang=lang,
            digest=digest, signature=signature,
        )

    # utf-8-sig, not utf-8: a leading BOM is a character to ast.parse and
    # fails the whole file ("invalid non-printable character U+FEFF").
    # Windows-authored sources carry one routinely. Harmless without a BOM.
    source = raw.decode("utf-8-sig", errors="replace")

    plan = _FilePlan(
        rel_path=rel_path, action="index", lang=lang, digest=digest,
        signature=signature, module_name=rel_path, attributes={},
    )
    if lang == "python":
        _plan_python(repo_root, rel_path, source, plan)
    else:
        if aliases is None:
            aliases = jssource.load_path_aliases(repo_root)
        _plan_script(repo_root, rel_path, source, aliases, plan)
    return plan


def _apply_plan(
    conn: sqlite3.Connection,
    repo_id: str,
    plan: _FilePlan,
    result: IndexResult,
) -> None:
    """Write one file's plan to the store — the only phase needing the lock."""
    if plan.action == "skip":
        result.skipped_unchanged += 1
        return
    if plan.action == "retire":
        _, edges = _retire_file(conn, repo_id, plan.rel_path)
        result.removed += 1
        result.edges_invalidated += edges
        return
    if plan.action == "refresh":
        state.record_indexed(
            conn, repo_id, path=plan.rel_path, kind="file",
            content_hash=plan.digest, lang=plan.lang, signature=plan.signature,
        )
        result.skipped_unchanged += 1
        return

    if plan.syntax_error:
        result.syntax_errors += 1

    module_node = _upsert_node(
        conn, repo_id, kind="module", qualified_name=plan.module_name or "",
        file_path=plan.rel_path, lang=plan.lang, span=None,
        attributes=plan.attributes,
    )
    result.nodes_written += 1

    # Re-emitting is a replace, not an append: retire what this file used to
    # point at before writing what it points at now.
    result.edges_invalidated += _invalidate_outbound(
        conn, repo_id, _node_ids_for_file(conn, repo_id, plan.rel_path)
    )

    for spec in plan.nodes:
        _upsert_node(
            conn, repo_id, kind=spec.kind, qualified_name=spec.qualified_name,
            file_path=plan.rel_path, lang=plan.lang, span=spec.span,
        )
        result.nodes_written += 1

    for edge in plan.edges:
        dst = _upsert_node(
            conn, repo_id, kind="module", qualified_name=edge.target,
            file_path=None, lang=None, span=None,
        )
        conn.execute(
            "INSERT INTO code_edges "
            "(edge_id, repo_id, src_id, dst_id, relation, confidence, "
            " valid_from, metadata) VALUES (?, ?, ?, ?, 'imports', 1.0, ?, ?)",
            (
                uuid.uuid4().hex, repo_id, module_node, dst, _now(),
                json.dumps(edge.metadata),
            ),
        )
        result.edges_written += 1

    state.record_indexed(
        conn, repo_id, path=plan.rel_path, kind="file",
        content_hash=plan.digest, lang=plan.lang, signature=plan.signature,
    )
    result.processed += 1


def index_file(
    conn: sqlite3.Connection,
    repo_id: str,
    repo_root: Path,
    rel_path: str,
    result: IndexResult,
    *,
    force: bool = False,
    aliases: dict[str, list[str]] | None = None,
) -> None:
    """Index one file, skipping it when its bytes are unchanged."""
    plan = _plan_file(
        conn, repo_id, repo_root, rel_path, force=force, aliases=aliases
    )
    _apply_plan(conn, repo_id, plan, result)


def walk_repo(repo_root: Path) -> list[str]:
    """Every indexable file, repo-relative, skipping vendored trees.

    Skipped directories are pruned *during* the walk, not filtered after it.
    The earlier version rglob'd everything and discarded what it did not want,
    which on a real Node repo meant visiting 184,642 paths to find 2,568 --
    33 s of the walk spent descending into node_modules only to throw the
    results away. Pruning is the same answer for a fraction of the syscalls,
    and it applies equally to .venv and site-packages in Python repos.
    """
    found: list[str] = []
    root = str(repo_root)
    for current, dirnames, filenames in os.walk(root):
        # Mutating dirnames in place is what stops os.walk descending.
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if os.path.splitext(name)[1] not in INDEXABLE_SUFFIXES:
                continue
            if name.endswith(_SKIP_FILES):
                continue
            absolute = os.path.join(current, name)
            found.append(Path(absolute).relative_to(repo_root).as_posix())
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
    # tsconfig is read once per pass, not once per file: it does not change
    # mid-walk, and re-reading it for every one of thousands of files is pure
    # IO for an unchanging answer.
    aliases = jssource.load_path_aliases(repo_root)

    # Two phases, because SQLite allows one writer and everyone else shares
    # this file. Planning (stat, read, hash, parse, resolve) is nearly the
    # whole cost of the pass -- 20 of 21 s on a 2,568-file repo -- and runs
    # with no transaction open, so an interactive memory_save proceeds
    # untouched throughout. Only applying the accumulated plans takes the
    # write lock. The earlier shapes both failed people: one transaction for
    # the whole walk locked concurrent writers out entirely (two of five
    # refused after the 30 s busy timeout), and interleaving parse work inside
    # chunked transactions still held the lock long enough for a
    # cross-process writer to wait 7.1 s.
    #
    # The plans are small -- ~20k tuples for that same repo -- so holding them
    # in memory is nothing next to parsing the files was.
    plans = [
        _plan_file(conn, repo_id, repo_root, rel_path, force=force, aliases=aliases)
        for rel_path in walk_repo(repo_root)
    ]

    # Apply in batches: commit-per-plan is the one shape that is always wrong
    # (autocommit measured 84.9 s against 2.8 s batched). The yield between
    # batches hands the lock to any waiting writer; "skip" plans write nothing
    # and do not count toward a batch, so a warm no-op pass stays one cheap
    # transaction instead of sleeping its way through empty ones.
    managed = not conn.in_transaction
    written_in_batch = 0
    if managed:
        conn.execute("BEGIN")
    try:
        for plan in plans:
            _apply_plan(conn, repo_id, plan, result)
            if plan.action != "skip":
                written_in_batch += 1
            if managed and written_in_batch >= COMMIT_EVERY_FILES:
                conn.execute("COMMIT")
                time.sleep(COMMIT_YIELD_SECONDS)
                conn.execute("BEGIN")
                written_in_batch = 0
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
    aliases = jssource.load_path_aliases(repo_root)
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
        if entry.path.endswith(_SKIP_FILES):
            continue
        index_file(conn, repo_id, repo_root, entry.path, result, aliases=aliases)
    conn.commit()
    # Only after the work is durable: a crash between these two leaves the
    # entries claimed, and the staleness sweep returns them to the queue.
    journal.complete(conn, [e.seq for e in batch])
    return result
