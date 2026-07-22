"""
CLARA — command-line interface (`clara`).

One-command setup plus human-facing visibility and control over the
zero-backend memory store. Works with no API key, no local LLM, and no
server: SQLite (+ FTS5) is the only state.

Commands:
    clara init [--project] [--agent NAME]   set up the store + print wiring
    clara context [QUERY...]                print the memory context block
    clara remember TEXT...                  rule-based extract + store
    clara list [--query Q] [--type T]       inspect stored memories
    clara forget MEMORY_ID [--archive]      retire a memory (never deletes)
    clara stats                             store location + counts
    clara doctor [--quiet]                  health check (exit 0/1/2)
    clara mcp                               run the MCP stdio server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from clara.extraction.heuristic import HeuristicExtractor
from clara.integrations.local_memory import LocalMemory
from clara.integrations.mcp_server import default_db_path
from clara.update.engine import classify_memory_type


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

# Ignore only the store's own artifacts; everything else in .clara/ (policy
# exports, the .gitignore itself) stays visible to git.
_PROJECT_GITIGNORE = "clara.db*\n.maintenance\nquarantine/\n"


def _resolve_db_path(project: bool) -> str:
    if project:
        path = Path.cwd() / ".clara" / "clara.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        gitignore = path.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_PROJECT_GITIGNORE, encoding="utf-8")
        elif gitignore.read_text(encoding="utf-8").strip() == "*":
            gitignore.write_text(_PROJECT_GITIGNORE, encoding="utf-8")
            print(
                "warning: rewrote .clara/.gitignore — the old '*' hid every future "
                ".clara file from git; now only clara.db*, .maintenance and "
                "quarantine/ are ignored",
                file=sys.stderr,
            )
        return str(path)
    return default_db_path()


async def _open(db_path: str | None = None) -> LocalMemory:
    return await LocalMemory.create(db_path or default_db_path())


# ---------------------------------------------------------------------------
# clara init
# ---------------------------------------------------------------------------

_AGENT_SNIPPETS = {
    "claude-code": """\
# Claude Code
#   Register the MCP server (project scope: add --scope project):
claude mcp add clara -- clara-mcp

#   Optional: inject memory at session start — add to ~/.claude/settings.json:
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|compact",
        "hooks": [{"type": "command", "command": "clara-mcp recall --top-k 12"}]
      }
    ]
  }
}""",
    "codex": """\
# OpenAI Codex CLI — append to ~/.codex/config.toml:
[mcp_servers.clara]
command = "clara-mcp"

# Optional: add to ~/.codex/AGENTS.md:
#   At the start of a task, call the clara `memory_search` tool with the task topic.""",
    "gemini": """\
# Gemini CLI — merge into ~/.gemini/settings.json:
{"mcpServers": {"clara": {"command": "clara-mcp"}}}""",
    "cursor": """\
# Cursor — merge into ~/.cursor/mcp.json:
{"mcpServers": {"clara": {"command": "clara-mcp"}}}""",
    "generic": """\
# Any CLI that can shell out (Aider, custom agents):
#   put context into your prompt:      clara context "<task description>"
#   store a fact from plain text:      clara remember "we deploy with fly.io" """,
}


async def _cmd_init(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.project)
    memory = await _open(db_path)
    try:
        stats = await memory.stats()
    finally:
        await memory.close()
    active = sum(stats.get("active_by_type", {}).values())
    print(f"CLARA store ready: {db_path}")
    print(f"  backend: {stats.get('backend')}")
    print(f"  active memories: {active}")
    agents = [args.agent] if args.agent else list(_AGENT_SNIPPETS)
    print("\nWire it into your coding agent:\n")
    for agent in agents:
        snippet = _AGENT_SNIPPETS.get(agent)
        if snippet is None:
            print(f"Unknown agent {agent!r}. Known: {', '.join(_AGENT_SNIPPETS)}")
            return 2
        print(snippet)
        print()
    return 0


# ---------------------------------------------------------------------------
# clara context / remember / list / forget / stats
# ---------------------------------------------------------------------------


async def _cmd_context(args: argparse.Namespace) -> int:
    memory = await _open()
    try:
        query = " ".join(args.query).strip()
        if query:
            result = await memory.search(query, top_k=args.top_k)
        else:
            result = await memory.recent(n=args.top_k)
    finally:
        await memory.close()
    if result["total"]:
        print(result["context"])
    return 0


_FACT_TO_SAVE_FIELDS = {
    "belief": lambda f: {
        "subject": f.subject, "relation": f.relation, "object": f.object,
        "is_negation": f.is_negation,
    },
    "event": lambda f: {
        "subject": f.subject, "event_type": f.relation, "description": f.object,
    },
    "skill": lambda f: {"name": f.object},
    "world_model": lambda f: {
        "entity_type": f.object if f.relation in ("is", "is_a") else "entity",
        "name": f.subject,
        "properties": (
            {} if f.relation in ("is", "is_a") else {f.relation: f.object}
        ),
    },
}


async def _cmd_remember(args: argparse.Namespace) -> int:
    text = " ".join(args.text).strip()
    facts = HeuristicExtractor().extract_sync(text)
    if not facts:
        print("Nothing durable recognized. (Rule-based extraction is "
              "precision-first; rephrase as e.g. 'I use X', 'we deployed Y', "
              "'Z runs on W'.)")
        return 1
    memory = await _open()
    saved = 0
    try:
        for fact in facts:
            mem_type = classify_memory_type(fact).value
            kwargs: dict[str, Any] = {
                "domain": fact.domain,
                "confidence": fact.confidence,
                "description": fact.raw_text,
            }
            kwargs.update(_FACT_TO_SAVE_FIELDS[mem_type](fact))
            result = await memory.save(mem_type=mem_type, **kwargs)
            saved += 1
            negation = " (negation)" if fact.is_negation else ""
            print(f"saved [{mem_type}] {fact.subject} {fact.relation} "
                  f"{fact.object}{negation} -> {result['memory_id']}")
    finally:
        await memory.close()
    return 0 if saved else 1


async def _cmd_list(args: argparse.Namespace) -> int:
    memory = await _open()
    try:
        types = [args.type] if args.type else None
        if args.query:
            result = await memory.search(args.query, top_k=args.limit, types=types)
        else:
            result = await memory.recent(n=args.limit, types=types)
    finally:
        await memory.close()
    if not result["total"]:
        print("(no memories)")
        return 0
    for hit in result["hits"]:
        print(f"{hit['memory_id']}  [{hit['type']}]  "
              f"conf={hit['confidence']:.2f}  {json.dumps(hit['content'])[:120]}")
    return 0


async def _cmd_forget(args: argparse.Namespace) -> int:
    memory = await _open()
    try:
        result = await memory.forget(args.memory_id, archive=args.archive)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await memory.close()
    print(f"{result['action']}: {args.memory_id}")
    return 0


async def _cmd_stats(args: argparse.Namespace) -> int:
    memory = await _open()
    try:
        stats = await memory.stats()
    finally:
        await memory.close()
    print(json.dumps(stats, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# clara doctor
# ---------------------------------------------------------------------------


async def _cmd_doctor(args: argparse.Namespace) -> int:
    """Health check. Exit codes: 0 healthy, 1 degraded-but-usable, 2 unusable."""
    from sqlalchemy import text as sa_text

    from clara.db.fts import FTS_TABLE

    checks: list[tuple[str, bool, str]] = []
    db_path = default_db_path()

    try:
        memory = await _open(db_path)
    except Exception as exc:  # noqa: BLE001 — the whole point is to report it
        if not args.quiet:
            print(f"unusable: cannot open store at {db_path}: {exc}")
        return 2

    try:
        async with memory._engine.connect() as conn:
            integrity = (await conn.execute(sa_text("PRAGMA quick_check(1)"))).scalar()
            checks.append(("sqlite integrity", integrity == "ok", str(integrity)))
            fts_row = (
                await conn.execute(
                    sa_text("SELECT 1 FROM sqlite_master WHERE name = :n"),
                    {"n": FTS_TABLE},
                )
            ).first()
            checks.append((
                "fts5 index", fts_row is not None,
                "present" if fts_row else "missing (lexical scan fallback)",
            ))
        stats = await memory.stats()
        active = sum(stats.get("active_by_type", {}).values())
        checks.append(("store readable", True, f"{active} active"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("store readable", False, str(exc)))
    finally:
        await memory.close()

    hard_failures = [c for c in checks if not c[1] and c[0] != "fts5 index"]
    degraded = [c for c in checks if not c[1]]

    if not args.quiet:
        print(f"store: {db_path}")
        for name, ok, detail in checks:
            print(f"  [{'ok' if ok else '!!'}] {name}: {detail}")
    if hard_failures:
        return 2
    return 1 if degraded else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clara",
        description="CLARA memory: zero-key persistent memory for coding agents.",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Set up the store and print agent wiring.")
    p_init.add_argument("--project", action="store_true",
                        help="Use ./.clara/clara.db instead of the global store.")
    p_init.add_argument("--agent", choices=sorted(_AGENT_SNIPPETS),
                        help="Only print wiring for this agent.")

    p_ctx = sub.add_parser("context", help="Print the memory context block.")
    p_ctx.add_argument("query", nargs="*")
    p_ctx.add_argument("--top-k", type=int, default=8)

    p_rem = sub.add_parser("remember", help="Extract facts from text (no LLM) and store them.")
    p_rem.add_argument("text", nargs="+")

    p_list = sub.add_parser("list", help="List stored memories.")
    p_list.add_argument("--query", default=None)
    p_list.add_argument("--type", choices=["belief", "event", "skill", "world_model"])
    p_list.add_argument("--limit", type=int, default=20)

    p_forget = sub.add_parser("forget", help="Deprecate (or archive) a memory.")
    p_forget.add_argument("memory_id")
    p_forget.add_argument("--archive", action="store_true")

    sub.add_parser("stats", help="Show store stats as JSON.")

    p_doc = sub.add_parser("doctor", help="Health check (exit 0/1/2).")
    p_doc.add_argument("--quiet", action="store_true")

    sub.add_parser("mcp", help="Run the MCP stdio server (same as clara-mcp).")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    if args.command == "mcp":
        from clara.integrations.mcp_server import build_server
        build_server().run()
        return

    handler = {
        "init": _cmd_init,
        "context": _cmd_context,
        "remember": _cmd_remember,
        "list": _cmd_list,
        "forget": _cmd_forget,
        "stats": _cmd_stats,
        "doctor": _cmd_doctor,
    }[args.command]
    raise SystemExit(asyncio.run(handler(args)))


if __name__ == "__main__":
    main()
