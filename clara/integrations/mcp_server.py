"""
CLARA — MCP server (clara-mcp)

Exposes the zero-backend :class:`LocalMemory` to Claude Code (or any MCP client)
as a small set of tools. Claude itself decides what to remember and recall — no
LLM, embedding model, API key, or vector server is used by this layer.

Run as an MCP stdio server::

    clara-mcp                      # what Claude Code launches
    python -m clara.integrations.mcp_server

Or print a memory context block (used by a SessionStart hook)::

    clara-mcp recall "deploy postgres"
    clara-mcp recall               # most relevant recent memories

The store lives at ``$CLARA_DB_PATH`` or ``$CLARA_HOME/clara.db`` or, by
default, ``~/.clara/clara.db`` (global across every project).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from clara.flags import DOCS_DISABLED_HINT, docs_enabled
from clara.integrations.local_memory import LocalMemory
from clara.store import global_db_path, resolve_store, secure_store_file

logger = logging.getLogger(__name__)

SERVER_NAME = "clara-memory"

# Opportunistic maintenance cadence: there is no cron/daemon in the CLI
# profile, so decay + pruning run inline (bounded, sub-second at CLI scale)
# when the store is opened and the last pass is older than this.
MAINTENANCE_INTERVAL_SECONDS = 24 * 3600

# A held maintenance lock older than this is a crash residue, not a runner.
_MAINTENANCE_LOCK_STALE_S = 6 * 3600

# ---------------------------------------------------------------------------
# Store resolution + per-store cache
# ---------------------------------------------------------------------------


def default_db_path() -> str:
    """Resolve the store path for the current anchor, creating the parent dir.

    Kept (name and all) for the CLI and tests; delegates to the shared
    resolver so every entry point agrees on the store.
    """
    return str(resolve_store(create=True).db_path)


# One LocalMemory per resolved store file. A session normally touches one
# store; the cap covers a user hopping between a few repos with project
# stores without accumulating engines forever.
_MAX_CACHED_STORES = 4

_memories: dict[str, LocalMemory] = {}
_memory_lock = asyncio.Lock()

_server_ref: Any = None  # the FastMCP instance, set by build_server()

_ANCHOR_TTL_S = 30.0
_anchor_cache: tuple[float, str] | None = None


async def _session_anchor() -> str:
    """Directory anchoring store resolution for this call.

    Order: MCP workspace roots (reflect the client even when the server
    process cwd is stale) -> the session-cwd file a hook wrote (covers
    mid-session ``cd`` when the client never refreshes roots) -> process cwd.
    Cached briefly: a roots round-trip per tool call would be waste.
    """
    global _anchor_cache
    now = time.monotonic()
    if _anchor_cache is not None and now - _anchor_cache[0] < _ANCHOR_TTL_S:
        return _anchor_cache[1]
    anchor: str | None = None
    if _server_ref is not None:
        try:
            ctx = _server_ref.get_context()
            roots_result = await ctx.session.list_roots()
            path_str = str(roots_result.roots[0].uri)
            if path_str.startswith("file://"):
                from urllib.request import url2pathname

                candidate = url2pathname(path_str[7:])
                if os.path.isdir(candidate):
                    anchor = candidate
        except Exception:  # noqa: BLE001 — roots are optional
            anchor = None
    if anchor is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")
        if session_id:
            base = Path(os.environ.get("CLARA_HOME") or (Path.home() / ".clara"))
            hint = base / "session-cwd" / session_id
            try:
                if hint.is_file() and time.time() - hint.stat().st_mtime < 12 * 3600:
                    candidate = hint.read_text(encoding="utf-8").strip()
                    if candidate and os.path.isdir(candidate):
                        anchor = candidate
            except OSError:
                anchor = None
    if anchor is None:
        # Claude Code passes CLAUDE_PROJECT_DIR to stdio MCP servers ("MCP
        # stdio servers now receive CLAUDE_PROJECT_DIR in their environment,
        # matching hooks"). It is fixed at launch, so it ranks below roots and
        # the session-cwd hint, both of which track a mid-session cd — but it
        # beats the process cwd, which for a server spawned by the client is
        # only incidentally related to the project.
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "").strip()
        if project_dir and os.path.isdir(project_dir):
            anchor = project_dir
    if anchor is None:
        anchor = os.getcwd()
    _anchor_cache = (now, anchor)
    return anchor


async def _get_memory_at(key: str) -> LocalMemory:
    memory = _memories.get(key)
    if memory is None:
        async with _memory_lock:
            memory = _memories.get(key)
            if memory is None:
                memory = await LocalMemory.create(key)
                _memories[key] = memory
                while len(_memories) > _MAX_CACHED_STORES:
                    evicted_key, evicted = next(iter(_memories.items()))
                    del _memories[evicted_key]
                    try:
                        await evicted.close()
                    except Exception:  # noqa: BLE001
                        logger.debug("evicted store close failed", exc_info=True)
                # Maintenance rides the first open of a store, off the
                # tool-call path so the first save/search does not pay for
                # a decay pass.
                asyncio.get_running_loop().create_task(
                    _run_maintenance_if_due(memory, key)
                )
    return memory


async def _get_memory() -> LocalMemory:
    """The memory store for this call's anchor (project store when opted in)."""
    anchor = await _session_anchor()
    return await _get_memory_at(str(resolve_store(anchor, create=True).db_path))


def _docs_db_path() -> str:
    """The doc ledger's store: always global (worktrees/clones share one
    ledger, keyed by repo_id), honoring the CLARA_DB_PATH explicit override."""
    path = global_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_store_file(path)
    return str(path)


async def _get_docs_memory() -> LocalMemory:
    return await _get_memory_at(_docs_db_path())


async def _run_maintenance_if_due(memory: LocalMemory, db_path: str) -> None:
    """Run the daily housekeeping pass for this server's session anchor.

    The pass itself lives in clara.maintenance so the CLI can run it too; it
    was private to this module, which meant a CLI-only store never got decayed,
    pruned or backed up. Only the anchor differs between callers: the server
    resolves it from workspace roots, which follow the client across a
    mid-session cd, while `clara maintain` uses its cwd.
    """
    from clara.maintenance import run_if_due

    await run_if_due(memory, db_path, anchor=await _session_anchor())


# ---------------------------------------------------------------------------
# MCP server (built lazily so the `recall` CLI works without the mcp package)
# ---------------------------------------------------------------------------


def _open_index(anchor: str) -> tuple[Any, str]:
    """Open the store for *anchor* and return it with the repo key."""
    from clara.db.migrations import open_db
    from clara.repoid import repo_id
    from clara.store import git_toplevel, resolve_store

    root = git_toplevel(anchor) or anchor
    resolution = resolve_store(anchor, create=True)
    conn = open_db(str(resolution.db_path))
    return conn, repo_id(root)


def _index_populated(conn: Any, repo_key: str) -> bool:
    """False when nothing has been indexed for this repo yet.

    Distinguishing "no dependencies" from "never indexed" matters: the first
    is a fact about the code, the second is a fact about CLARA, and answering
    the second as though it were the first is how a tool misleads.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM code_nodes WHERE repo_id = ? LIMIT 1", (repo_key,)
        ).fetchone()
    except Exception:  # noqa: BLE001 — pre-migration store: simply not indexed
        return False
    return row is not None


def _validated_confidence(confidence: float | None) -> float | None:
    """Reject a confidence outside 0..1 rather than quietly clamping it.

    Both tools document "confidence (0..1)". Accepting 5.0 and storing 1.0
    reports success for a value that was silently replaced, and the caller here
    is a model, which will go on sending 5.0 because nothing told it otherwise.

    Library callers still clamp defensively (that behaviour is relied on and
    tested); this is the boundary where a wrong value is a mistake worth
    naming.
    """
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        raise ValueError(
            f"confidence must be a number between 0.0 and 1.0 (got {confidence!r})."
        ) from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"confidence must be between 0.0 and 1.0 (got {confidence!r})."
        )
    return value


# Fragments of the tool-call wire format. None of these belongs inside a
# memory's text; when one appears, the caller's tool call was malformed and
# the XML framing leaked into a field. Observed in a real store: a belief's
# evidence ended with '.</parameter>\n<parameter name="domain">security' -- the
# call broke mid-stream, the markup was glued into the description, and the
# domain field was silently swallowed. Storing that verbatim preserves a
# mangled fact and hides that a field went missing.
_MARKUP_MARKERS = (
    "<parameter", "</parameter",
    "<invoke", "</invoke",
    "<function_calls", "</function_calls", "<function_results",
)


def _reject_leaked_markup(value: Any, where: str) -> None:
    """Refuse strings carrying tool-call framing, wherever they are nested.

    A save whose text contains the wire format is almost never intentional --
    it means the framing leaked and at least one field was probably lost with
    it. Rejecting is recoverable (the model re-sends plain text); storing is
    not, because nothing downstream can tell mangled from meant. The rare
    legitimate memory *about* this markup loses to the observed failure mode.
    """
    if isinstance(value, str):
        lowered = value.lower()
        for marker in _MARKUP_MARKERS:
            if marker in lowered:
                raise ValueError(
                    f"{where} contains tool-call markup ({marker!r}...). This "
                    "usually means the tool call was malformed and its XML "
                    "framing leaked into a field, losing whatever came after "
                    "it. Nothing was stored - re-send the save as plain text."
                )
    elif isinstance(value, dict):
        for key, nested in value.items():
            _reject_leaked_markup(nested, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            _reject_leaked_markup(nested, f"{where}[{position}]")


def _deps_sync(
    anchor: str, target: str, direction: str, depth: int
) -> dict[str, Any]:
    """Open, query and close the index on a single thread.

    A sqlite3 connection belongs to the thread that created it. Opening inside
    ``asyncio.to_thread`` and then querying the returned connection back on the
    event loop raised "SQLite objects created in a thread can only be used in
    that same thread" on *every* call, so all three code tools failed for any
    repo. The whole operation has to happen in the worker, not just the open.
    """
    from clara.index import queries

    conn, repo_key = _open_index(anchor)
    try:
        if not _index_populated(conn, repo_key):
            return {"indexed": False, "target": target,
                    "hint": "run `clara index` in this repo first"}
        found = queries.dependencies(
            conn, repo_key, target, direction=direction, depth=depth
        )
        return {
            "indexed": True,
            "target": target,
            "direction": direction,
            "depth": depth,
            "count": len(found),
            "modules": [
                {"name": d.qualified_name, "path": d.file_path, "depth": d.depth}
                for d in found
            ],
        }
    finally:
        conn.close()


def _health_sync(anchor: str) -> dict[str, Any]:
    """Cycles and unreferenced modules, entirely on one thread."""
    from pathlib import Path as _Path

    from clara.index import queries
    from clara.store import git_toplevel as _git_toplevel

    conn, repo_key = _open_index(anchor)
    try:
        if not _index_populated(conn, repo_key):
            return {"indexed": False,
                    "hint": "run `clara index` in this repo first"}
        root = _Path(_git_toplevel(anchor) or anchor)
        cycles = queries.find_cycles(conn, repo_key)
        unused = queries.unused_modules(conn, repo_key, repo_root=root)
        return {
            "indexed": True,
            "cycles": [" -> ".join(c) for c in cycles],
            "unused_modules": unused,
            "note": "unused = nothing imports it, after excluding what the "
                    "project declares it runs. For Python: console scripts, "
                    "pytest testpaths, __main__ guards. For JS/TS: package.json "
                    "entry points and script commands, <script src> in any HTML "
                    "page, extension manifests, plus test files, *.config.* and "
                    "framework route files. A module loaded by name at runtime "
                    "cannot be seen statically, so review before deleting.",
        }
    finally:
        conn.close()


async def _current_repo() -> str | None:
    """The session anchor's repo id, for labeling another project's facts.

    None on failure: provenance labels are a nicety, and a repo-resolution
    hiccup must never take memory_search down with it.
    """
    try:
        from clara.repoid import repo_id

        return repo_id(await _session_anchor())
    except Exception:  # noqa: BLE001 — labeling is best-effort
        logger.debug("current-repo resolution failed", exc_info=True)
        return None


def build_server() -> Any:
    """Construct the FastMCP server with all tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The 'mcp' package is required to run the clara-mcp server.\n"
            "Install it with:  pip install 'clara-memory[mcp]'"
        ) from exc

    server = FastMCP(SERVER_NAME)
    global _server_ref
    _server_ref = server

    @server.tool()
    async def memory_save(
        mem_type: str = "belief",
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        is_negation: bool = False,
        event_type: str | None = None,
        name: str | None = None,
        trigger_conditions: list[str] | None = None,
        steps: list[str] | None = None,
        entity_type: str | None = None,
        properties: dict[str, Any] | None = None,
        description: str | None = None,
        domain: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Save one durable memory to long-term storage.

        Call this when you learn something worth remembering across sessions:
        a stable user preference, an architectural decision, a project fact, a
        reusable procedure, or the current state of a tool/service.

        Choose mem_type and fill the matching fields:
        - "belief": an atomic fact. Requires subject, relation, object
          (e.g. subject="user", relation="prefers", object="tabs over spaces").
          Set is_negation=true for "not" facts.
        - "event": something that happened. Requires subject and event_type
          (e.g. subject="user", event_type="migrated", description="moved DB to LanceDB").
        - "skill": a reusable procedure. Requires name; optionally
          trigger_conditions (when to use it) and steps (how).
        - "world_model": current state of an entity. Requires entity_type and
          name; put state in properties (e.g. entity_type="service", name="api",
          properties={"port": 8000, "status": "deployed"}).

        Shared optional fields: description (raw context), domain (a tag like
        "backend"), confidence (0..1), tags (list of strings).
        """
        fields: dict[str, Any] = {
            "subject": subject, "relation": relation, "object": object,
            "event_type": event_type, "name": name,
            "trigger_conditions": trigger_conditions, "steps": steps,
            "entity_type": entity_type, "properties": properties,
            "description": description, "domain": domain, "tags": tags,
        }
        _validate_save_fields(fields, "")
        memory = await _get_memory()
        return await memory.save(
            mem_type=mem_type,
            is_negation=is_negation,
            confidence=_validated_confidence(confidence),
            **fields,
        )

    def _validate_save_fields(fields: dict[str, Any], where: str) -> None:
        for key, value in fields.items():
            if value is not None:
                _reject_leaked_markup(value, f"{where}{key}")

    @server.tool()
    async def memory_save_many(items: list[dict[str, Any]]) -> dict[str, Any]:
        """Save several memories in one call — one transaction, all or nothing.

        Use this instead of parallel memory_save calls whenever you have more
        than a couple of facts: one request, one commit (measured: 100 facts
        in 2.1 s against 7.4 s as sequential saves), and a batch cannot race
        itself the way concurrent single saves can.

        Each item takes the same fields as memory_save: mem_type (default
        "belief") plus that type's fields, and optionally confidence (0..1),
        domain, tags. If any item is invalid the whole batch is rejected with
        the item's index — nothing half-applies.
        """
        # Non-dict items never reach here: FastMCP validates list[dict] at the
        # schema layer and its error already names the index (items.1).
        for position, item in enumerate(items):
            try:
                _validated_confidence(item.get("confidence"))
            except ValueError as exc:
                raise ValueError(f"item {position}: {exc}") from exc
            _reject_leaked_markup(item, f"item {position}")
        memory = await _get_memory()
        return await memory.save_many(items)

    @server.tool()
    async def memory_search(
        query: str,
        top_k: int = 8,
        types: list[str] | None = None,
        graph_depth: int = 1,
    ) -> dict[str, Any]:
        """Search long-term memory by keywords and return matching memories.

        Use this at the start of a task, or whenever you need prior context
        about the user, the project, past decisions, or known procedures. The
        result includes a ready-to-read "MEMORY CONTEXT" block plus structured
        hits. Optionally filter types to any of:
        ["belief", "event", "skill", "world_model"].

        graph_depth (default 1) also walks the knowledge graph from the top
        hits' entities and appends a [GRAPH] relations section; set 0 to skip.
        """
        memory = await _get_memory()
        return await memory.search(
            query, top_k=top_k, types=types, graph_depth=graph_depth,
            current_repo=await _current_repo(),
        )

    @server.tool()
    async def memory_recent(
        n: int = 10,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the most relevant recent memories (no query needed)."""
        memory = await _get_memory()
        return await memory.recent(n=n, types=types, current_repo=await _current_repo())

    @server.tool()
    async def memory_update(
        memory_id: str,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Adjust an existing memory's confidence (0..1) and/or replace its tags."""
        if tags is not None:
            _reject_leaked_markup(tags, "tags")
        memory = await _get_memory()
        return await memory.update(
            memory_id, confidence=_validated_confidence(confidence), tags=tags
        )

    @server.tool()
    async def memory_forget(
        memory_id: str,
        archive: bool = False,
    ) -> dict[str, Any]:
        """Retire a memory: deprecate it (default) or archive it. Never deletes.

        Use when a fact is no longer true or a preference changed. The record is
        kept for audit but excluded from future searches.
        """
        memory = await _get_memory()
        return await memory.forget(memory_id, archive=archive)

    @server.tool()
    async def memory_stats() -> dict[str, Any]:
        """Report where the store lives and how many active memories it holds,
        plus knowledge-graph node/edge counts."""
        memory = await _get_memory()
        return await memory.stats()

    @server.tool()
    async def code_deps(
        target: str,
        direction: str = "forward",
        depth: int = 2,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """What a module imports, or what imports it.

        `target` is a dotted module name (`clara.db.migrations`) or a
        repo-relative path (`clara/db/migrations.py`). `direction="reverse"`
        answers "what depends on this" -- the set a change here can break.
        Depth is bounded; 1 is direct neighbours.

        Reads the code index, which `clara index` builds and the daily
        maintenance pass keeps current. Empty result with `indexed: false`
        means this repo has not been indexed yet -- say so rather than
        concluding the module has no dependencies.
        """
        anchor = repo or await _session_anchor()
        result: dict[str, Any] = await asyncio.to_thread(
            _deps_sync, anchor, target, direction, depth
        )
        return result

    @server.tool()
    async def code_impact(target: str, depth: int = 3,
                          repo: str | None = None) -> dict[str, Any]:
        """What breaks if this module changes — reverse dependencies, transitive.

        Use before editing a shared module, and to size a refactor. Same
        indexing caveat as `code_deps`.
        """
        result: dict[str, Any] = await code_deps(
            target, direction="reverse", depth=depth, repo=repo
        )
        return result

    @server.tool()
    async def code_health(repo: str | None = None) -> dict[str, Any]:
        """Import cycles and modules nothing imports.

        `unused` is literally "no inbound import edge" -- entry points, CLI
        mains and test modules legitimately appear there, so treat it as a list
        to look at, not a list to delete. Cycles are real: each one is a pair
        or loop of modules that import each other, usually survivable only
        because one side defers its import.
        """
        anchor = repo or await _session_anchor()
        result: dict[str, Any] = await asyncio.to_thread(_health_sync, anchor)
        return result

    @server.tool()
    async def project_profile(repo: str | None = None) -> dict[str, Any]:
        """Describe what this project *is*: language, package manager,
        frameworks, build/test tooling, and whether it is a monorepo.

        Read straight from the repository's manifests, so it is current even
        for a project CLARA has never seen before and needs no memories to
        have been saved. Every claim carries the file it came from; anything
        the manifests do not state is simply absent rather than guessed.

        Call this instead of shelling out to inspect package.json/pyproject.
        """
        from clara.project import detect_project

        root = await _repo_root(repo)
        profile = await asyncio.to_thread(detect_project, root)
        summary = profile.summary()
        summary["evidence"] = [
            {"category": f.category, "value": f.value, "source": f.evidence}
            for f in profile.facts
        ]
        return summary

    @server.tool()
    async def statusline_install(
        enable: bool = True,
        refresh_interval: int = 5,
        force: bool = False,
    ) -> dict[str, Any]:
        """Show (or hide) CLARA's live memory counter in the status bar.

        Writes the ``statusLine`` block into the user's own
        ``~/.claude/settings.json`` — a plugin cannot ship one itself. This is
        how users who never open a terminal enable it, since the plugin's
        ``clara`` executable is not on their PATH.

        Existing settings are preserved and backed up first. If a status line
        from another tool is already configured, this reports ``blocked``
        rather than replacing it; pass ``force=True`` to override. Set
        ``enable=False`` to remove CLARA's entry again.
        """
        from clara import statusline_setup

        if not enable:
            return await asyncio.to_thread(statusline_setup.uninstall)
        return await asyncio.to_thread(
            statusline_setup.install,
            refresh_interval=refresh_interval,
            force=force,
        )

    @server.tool()
    async def statusline_status() -> dict[str, Any]:
        """Report whether CLARA's memory counter is in the status bar."""
        from clara import statusline_setup

        return await asyncio.to_thread(statusline_setup.status)

    async def _repo_root(repo: str | None) -> str:
        """Resolve the repo root: explicit arg → session anchor (MCP roots →
        session-cwd hint → server cwd)."""
        from clara.docs.scan import find_repo_root

        if repo is not None:
            return repo
        return find_repo_root(await _session_anchor())

    @server.tool()
    async def docs_status(path_or_query: str, repo: str | None = None) -> dict[str, Any]:
        """Standing of a repository document from the curator ledger.

        Returns its tier (T0 pinned / T1 authoritative / T2 working / T3
        scratch / TX quarantined), lifecycle, deterministic signals (age,
        churn, dead refs, checkboxes, duplicates), and the supersession
        chain. Consult this before executing a plan-type document. repo:
        optional repo-root path; defaults to the session's workspace root
        (MCP roots) or the server's cwd. Run `clara docs scan` first if the
        repo is not in the ledger.
        """
        import sqlite3 as _sqlite3

        from clara.docs.report import get_status
        from clara.repoid import repo_id as compute_repo_id

        if not docs_enabled():
            return {"found": False, "disabled": True, "error": DOCS_DISABLED_HINT}

        root = await _repo_root(repo)

        def _lookup() -> dict[str, Any]:
            rid = compute_repo_id(root)
            conn = _sqlite3.connect(_docs_db_path())
            conn.row_factory = _sqlite3.Row
            try:
                status = get_status(conn, rid, path_or_query)
            finally:
                conn.close()
            if status is None:
                return {
                    "found": False,
                    "query": path_or_query,
                    "hint": "not in the ledger — run `clara docs scan` in the repo",
                }
            return status

        return await asyncio.to_thread(_lookup)

    @server.tool()
    async def docs_classify(
        path: str,
        doc_type: str,
        rationale: str,
        tier: str | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Attest a document's type (and optionally tier) in the curator ledger.

        Use when the heuristic classification is wrong or missing. doc_type:
        e.g. adr, spec, plan, guide, brainstorm, standard. tier: T0 pinned /
        T1 authoritative / T2 working / T3 scratch / TX quarantined. Always
        give a one-sentence rationale. Idempotent for unchanged content.
        """
        memory = await _get_docs_memory()
        return await memory.docs_classify(
            await _repo_root(repo), path=path, doc_type=doc_type, tier=tier,
            rationale=rationale,
        )

    @server.tool()
    async def docs_supersede(
        old_path: str,
        new_path: str,
        rationale: str,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Mark one document as superseded by another (e.g. plan v1 -> v2).

        The old document moves to the quarantine list and future reads of it
        get annotated; the new document becomes the current guidance. Also
        records a `supersedes` edge in the knowledge graph when present.
        """
        memory = await _get_docs_memory()
        return await memory.docs_supersede(
            await _repo_root(repo), old_path=old_path, new_path=new_path,
            rationale=rationale,
        )

    @server.tool()
    async def docs_fulfill(
        path: str,
        distilled: list[dict[str, Any]],
        evidence: str | None = None,
        rationale: str = "plan completed",
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Close out a completed plan document by distilling it into memory.

        Call immediately after finishing the work a plan-type document
        described. distilled: 1-5 memory_save-shaped facts (mem_type +
        fields) capturing the durable outcomes — decisions, constraints,
        standards — not implementation trivia. evidence: the commit/PR/issue
        ref proving completion. Atomic: the memories, the `fulfilled`
        lifecycle transition, and provenance graph edges commit together.
        Returns {doc_id, memory_ids, edge_ids}.
        """
        memory = await _get_docs_memory()
        return await memory.docs_fulfill(
            await _repo_root(repo), path=path, distilled=distilled,
            evidence=evidence, rationale=rationale,
        )

    @server.tool()
    async def docs_report(
        scope: str = "repo", repo: str | None = None
    ) -> dict[str, Any]:
        """Document rot report for the current repo (proposals only).

        Lists stale documents, dead-reference documents, duplicate clusters,
        and archive candidates. Nothing is mutated — present findings to the
        user and act only on their instruction. scope: reserved (only 'repo'
        today).
        """
        memory = await _get_docs_memory()
        return await memory.docs_report(await _repo_root(repo))

    @server.tool()
    async def graph_entity(name: str, include_history: bool = False) -> dict[str, Any]:
        """Look up one entity in the knowledge graph.

        Returns its card: display/canonical name, entity type, aliases,
        possible_duplicates (candidate merges — propose a merge to the user
        instead of guessing), linked world-model id, and its top relations.
        Set include_history=true to also see invalidated (✗) relations.
        """
        memory = await _get_memory()
        return await memory.graph_entity(name, include_history=include_history)

    @server.tool()
    async def graph_neighbors(
        name: str,
        depth: int = 1,
        relation: str | None = None,
        as_of: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Walk the knowledge graph outward from an entity.

        depth: hops to expand (1-2 typical). relation: filter to one relation
        (free-form; normalized like edge relations). as_of: ISO date/time to
        view the graph as it was then (includes edges since invalidated).
        Returns a rendered [GRAPH] section plus structured edges.
        """
        memory = await _get_memory()
        return await memory.graph_neighbors(
            name, depth=depth, relation=relation, as_of=as_of, limit=limit
        )

    @server.tool()
    async def graph_path(
        from_name: str,
        to_name: str,
        max_hops: int = 4,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        """Find how two entities are connected (shortest/best relation path)."""
        memory = await _get_memory()
        return await memory.graph_path(
            from_name, to_name, max_hops=max_hops, as_of=as_of
        )

    @server.tool()
    async def memory_link(
        src: str,
        relation: str,
        dst: str,
        entity_types: list[str] | None = None,
        confidence: float | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Save a belief AND get its graph edge in one call (linking sugar).

        Use active-voice relations (uses, depends_on, deployed_to). Name code
        entities by path ("src/api.py"), symbol ("src/api.py::handler"), or
        decision slug ("decision:move-to-sqlite"). entity_types optionally
        types the endpoints as [src_type, dst_type]. Returns belief_id,
        edge_id, and the resolved node names.
        """
        memory = await _get_memory()
        return await memory.memory_link(
            src,
            relation,
            dst,
            entity_types=entity_types,
            confidence=confidence,
            description=description,
        )

    return server


# ---------------------------------------------------------------------------
# recall CLI (for the SessionStart hook)
# ---------------------------------------------------------------------------


async def _recall(query: str, top_k: int) -> str:
    memory = await LocalMemory.create(default_db_path())
    try:
        if query.strip():
            result = await memory.search(query, top_k=top_k)
        else:
            result = await memory.recent(n=top_k)
    finally:
        await memory.close()

    if not result["total"]:
        return ""
    return (
        "The following is your persistent CLARA memory relevant to this "
        "session. Treat it as background context.\n\n" + str(result["context"])
    )


def _run_recall(args: argparse.Namespace) -> int:
    query = " ".join(args.query).strip()
    text = asyncio.run(_recall(query, args.top_k))
    if text:
        sys.stdout.write(text + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clara-mcp",
        description="CLARA zero-backend memory: MCP server + recall CLI.",
    )
    sub = parser.add_subparsers(dest="command")
    recall_p = sub.add_parser(
        "recall", help="Print a memory context block (for hooks); not an MCP server."
    )
    recall_p.add_argument("query", nargs="*", help="Optional search terms.")
    recall_p.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()

    if args.command == "recall":
        raise SystemExit(_run_recall(args))

    # Default: run the MCP stdio server.
    build_server().run()


if __name__ == "__main__":
    main()
