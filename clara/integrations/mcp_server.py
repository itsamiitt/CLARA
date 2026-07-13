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

from clara.integrations.local_memory import LocalMemory

logger = logging.getLogger(__name__)

SERVER_NAME = "clara-memory"

# Opportunistic maintenance cadence: there is no cron/daemon in the CLI
# profile, so decay + pruning run inline (bounded, sub-second at CLI scale)
# when the store is opened and the last pass is older than this.
MAINTENANCE_INTERVAL_SECONDS = 24 * 3600

# ---------------------------------------------------------------------------
# Store location + lazy singleton
# ---------------------------------------------------------------------------


def default_db_path() -> str:
    """Resolve the global SQLite store path, creating the parent dir."""
    explicit = os.environ.get("CLARA_DB_PATH")
    if explicit:
        path = Path(explicit)
    else:
        base = Path(os.environ.get("CLARA_HOME") or (Path.home() / ".clara"))
        path = base / "clara.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


_memory: LocalMemory | None = None
_memory_lock = asyncio.Lock()


async def _get_memory() -> LocalMemory:
    global _memory
    if _memory is None:
        async with _memory_lock:
            if _memory is None:
                _memory = await LocalMemory.create(default_db_path())
                await _run_maintenance_if_due(_memory, default_db_path())
    return _memory


async def _run_maintenance_if_due(memory: LocalMemory, db_path: str) -> None:
    """Run decay + pruning inline when the last pass is stale.

    Replaces the APScheduler cron jobs for the zero-backend profile: no
    daemon, no timer — maintenance rides the first store access of the day.
    All failures are logged and swallowed; memory availability must never
    depend on housekeeping.
    """
    marker = Path(db_path + ".maintenance")
    try:
        if marker.exists():
            age = time.time() - marker.stat().st_mtime
            if age < MAINTENANCE_INTERVAL_SECONDS:
                return
        from clara.scheduler.decay import DecayScheduler

        scheduler = DecayScheduler(memory._session_factory)
        decay_summary = await scheduler.run_daily_decay()
        prune_summary = await scheduler.run_weekly_pruning()
        marker.touch()
        logger.info(
            "Opportunistic maintenance: %s %s", decay_summary, prune_summary,
        )
    except Exception:  # noqa: BLE001 — housekeeping must never block memory
        logger.exception("Opportunistic maintenance failed")


# ---------------------------------------------------------------------------
# MCP server (built lazily so the `recall` CLI works without the mcp package)
# ---------------------------------------------------------------------------


def build_server():
    """Construct the FastMCP server with all tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The 'mcp' package is required to run the clara-mcp server.\n"
            "Install it with:  pip install 'clara-memory[mcp]'"
        ) from exc

    server = FastMCP(SERVER_NAME)

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
        memory = await _get_memory()
        return await memory.save(
            mem_type=mem_type,
            subject=subject,
            relation=relation,
            object=object,
            is_negation=is_negation,
            event_type=event_type,
            name=name,
            trigger_conditions=trigger_conditions,
            steps=steps,
            entity_type=entity_type,
            properties=properties,
            description=description,
            domain=domain,
            confidence=confidence,
            tags=tags,
        )

    @server.tool()
    async def memory_search(
        query: str,
        top_k: int = 8,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search long-term memory by keywords and return matching memories.

        Use this at the start of a task, or whenever you need prior context
        about the user, the project, past decisions, or known procedures. The
        result includes a ready-to-read "MEMORY CONTEXT" block plus structured
        hits. Optionally filter types to any of:
        ["belief", "event", "skill", "world_model"].
        """
        memory = await _get_memory()
        return await memory.search(query, top_k=top_k, types=types)

    @server.tool()
    async def memory_recent(
        n: int = 10,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the most relevant recent memories (no query needed)."""
        memory = await _get_memory()
        return await memory.recent(n=n, types=types)

    @server.tool()
    async def memory_update(
        memory_id: str,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Adjust an existing memory's confidence (0..1) and/or replace its tags."""
        memory = await _get_memory()
        return await memory.update(memory_id, confidence=confidence, tags=tags)

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
        """Report where the store lives and how many active memories it holds."""
        memory = await _get_memory()
        return await memory.stats()

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
        "session. Treat it as background context.\n\n" + result["context"]
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
