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
    clara docs scan|status|report           document lifecycle ledger
    clara graph rebuild|stats|show|path|doctor   knowledge-graph tools
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
        _print_plugin_health(db_path)
    if hard_failures:
        return 2
    return 1 if degraded else 0


def _print_plugin_health(db_path: str) -> None:
    """Plugin-health block: schema version, flags, ledger, venv. Info only."""
    import sqlite3
    import time as _time

    from clara.db.migrations import SCHEMA_VERSION, get_version
    from clara.flags import docs_enabled, graph_enabled

    print("plugin:")
    try:
        conn = sqlite3.connect(db_path)
        try:
            version = get_version(conn)
            print(f"  schema: v{version} (code supports v{SCHEMA_VERSION})")
            for label, sql in (
                ("graph", "SELECT COUNT(*) FROM graph_edges WHERE invalid_at IS NULL"),
                ("docs", "SELECT COUNT(*) FROM doc_registry WHERE invalid_at IS NULL"),
            ):
                try:
                    count = conn.execute(sql).fetchone()[0]
                    print(f"  {label} rows: {count}")
                except sqlite3.Error:
                    print(f"  {label} rows: (tables absent)")
            try:
                per_repo = conn.execute(
                    "SELECT repo_id, COUNT(*) FROM doc_registry "
                    "WHERE invalid_at IS NULL GROUP BY repo_id "
                    "ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
                for repo, count in per_repo:
                    print(f"    ledger {repo}: {count} docs")
            except sqlite3.Error:
                pass
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"  schema: unreadable ({exc})")
    print(f"  flags: graph={'on' if graph_enabled() else 'OFF'} "
          f"docs={'on' if docs_enabled() else 'OFF'}")

    base = Path(db_path).resolve().parent
    quarantine = base / "quarantine"
    manifests = sorted(quarantine.glob("*.tsv")) if quarantine.is_dir() else []
    if manifests:
        newest = max(manifests, key=lambda p: p.stat().st_mtime)
        age_h = (_time.time() - newest.stat().st_mtime) / 3600
        print(f"  quarantine manifests: {len(manifests)} (newest {age_h:.1f}h old)")
    else:
        print("  quarantine manifests: none")

    import os as _os

    data_dir = Path(_os.environ.get("CLAUDE_PLUGIN_DATA") or (base / "plugin"))
    current = data_dir / "current"
    if current.exists():
        try:
            target = current.resolve()
        except OSError:
            target = current
        print(f"  venv: {target}")
    else:
        print(f"  venv: (not installed under {data_dir})")
    install_log = data_dir / "install.log"
    if install_log.is_file():
        tail = install_log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
        for line in tail:
            print(f"    log: {line}")


# ---------------------------------------------------------------------------
# clara docs
# ---------------------------------------------------------------------------


async def _cmd_docs(args: argparse.Namespace) -> int:
    import sqlite3

    from clara.flags import DOCS_DISABLED_HINT, docs_enabled

    if not docs_enabled():
        print(f"error: {DOCS_DISABLED_HINT}", file=sys.stderr)
        return 2

    from clara.docs.report import build_report, format_report, get_status
    from clara.docs.scan import (
        clara_yml_ignored,
        dirty_sidecar,
        find_repo_root,
        scan_repo,
    )
    from clara.policy import load_policy
    from clara.repoid import repo_id as compute_repo_id

    db_path = default_db_path()
    root = find_repo_root(str(Path.cwd()))

    if args.docs_cmd == "scan":
        if clara_yml_ignored(root):
            print(
                "warning: clara.yml exists but is git-ignored — commit it so the "
                "team shares one policy",
                file=sys.stderr,
            )
        changed: list[str] | None = None
        if args.changed is not None:
            if args.changed:
                changed = args.changed
            else:
                sidecar = dirty_sidecar(db_path, compute_repo_id(root))
                changed = (
                    [line for line in sidecar.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
                    if sidecar.exists()
                    else []
                )
        summary = scan_repo(db_path, root, changed=changed)
        print(
            f"scanned {summary['scanned']} (new {summary['new']}, "
            f"unchanged {summary['unchanged']}, moved {summary['moved']}, "
            f"vanished {summary['vanished']}); {summary['total_active']} tracked"
        )
        return 0

    if args.docs_cmd in ("archive", "restore"):
        memory = await _open()
        try:
            handler = memory.docs_archive if args.docs_cmd == "archive" else memory.docs_restore
            result = await handler(root, path=args.path)
        finally:
            await memory.close()
        if not result.get("found"):
            print(f"error: {result.get('hint', 'document not found')}", file=sys.stderr)
            return 2
        if result.get("action") == "refused":
            print(f"refused: {result['reason']}", file=sys.stderr)
            return 2
        moved = " (file moved)" if result.get("moved") else ""
        print(f"{result['action']}: {result.get('to', args.path)}{moved}")
        return 0

    repo = compute_repo_id(root)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.docs_cmd == "status":
            status = get_status(conn, repo, args.path)
            if status is None:
                print(f"(not in the ledger: {args.path!r} — run `clara docs scan`)")
                return 1
            print(status["standing"])
            return 0
        # report
        report = build_report(conn, repo, load_policy(root))
        print(format_report(report))
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# clara graph
# ---------------------------------------------------------------------------


async def _cmd_graph(args: argparse.Namespace) -> int:
    from sqlalchemy import text as sa_text

    from clara.flags import GRAPH_DISABLED_HINT, graph_enabled

    if not graph_enabled():
        print(f"error: {GRAPH_DISABLED_HINT}", file=sys.stderr)
        return 2

    memory = await _open()
    try:
        if args.graph_cmd == "rebuild":
            counts = await memory.graph_rebuild(from_scratch=args.from_scratch)
            print(
                f"graph rebuilt: {counts['nodes']} nodes, {counts['edges']} edges "
                f"({counts['edges_created']} created this run)"
            )
            return 0

        if args.graph_cmd == "stats":
            stats = await memory.stats()
            graph = stats.get("graph")
            if not graph:
                print("graph: no graph tables (run `clara graph rebuild`)")
                return 1
            print(f"nodes: {graph['nodes']}")
            print(f"edges: {graph['edges']}")
            return 0

        if args.graph_cmd == "show":
            card = await memory.graph_entity(args.entity, include_history=args.history)
            if not card.get("found"):
                print(f"(no graph entity for {args.entity!r})")
                return 1
            print(f"{card['name']}  [{card['entity_type']}]")
            print(f"  canonical: {card['canonical_name']}")
            print(f"  mentions: {card['mention_count']}  "
                  f"expandable: {'yes' if card['expandable'] else 'no'}")
            if card["aliases"]:
                print(f"  aliases: {', '.join(card['aliases'])}")
            if card["possible_duplicates"]:
                print(f"  possible duplicates: {', '.join(card['possible_duplicates'])}")
            if card["world_model_id"]:
                print(f"  world model: {card['world_model_id']}")
            for line in card["edges"]:
                print(f"  {line}")
            return 0

        if args.graph_cmd == "path":
            result = await memory.graph_path(args.src, args.dst)
            if not result.get("found"):
                print("(no path found)")
                return 1
            print(f"path ({result['hops']} hops):")
            for line in result["path"]:
                print(f"  {line}")
            return 0

        if args.graph_cmd == "merge":
            result = await memory.graph_merge(args.dup, args.canonical)
            if not result.get("merged"):
                reason = result.get("reason") or f"missing: {result.get('missing')}"
                print(f"error: {reason}", file=sys.stderr)
                return 2
            audit = result["audit"]
            repointed = (
                len(audit["repointed_src_edges"]) + len(audit["repointed_dst_edges"])
            )
            print(
                f"merged {args.dup} into {args.canonical} "
                f"({repointed} edges re-pointed, audit {audit['merge_id']})"
            )
            return 0

        if args.graph_cmd == "export":
            print(await memory.graph_export(format=args.format))
            return 0

        # doctor
        async with memory._session_factory() as session:
            dup_rows = (
                await session.execute(
                    sa_text(
                        "SELECT node_id, display_name, properties FROM graph_nodes "
                        "WHERE status = 'active' "
                        "AND properties LIKE '%possible_duplicates%'"
                    )
                )
            ).all()
            dangling = (
                await session.execute(
                    sa_text(
                        "SELECT edge_id, belief_id FROM graph_edges "
                        "WHERE invalid_at IS NULL AND belief_id IS NOT NULL "
                        "AND NOT EXISTS (SELECT 1 FROM memories m WHERE "
                        "replace(CAST(m.memory_id AS TEXT), '-', '') = "
                        "replace(graph_edges.belief_id, '-', '') "
                        "AND m.status = 'active')"
                    )
                )
            ).all()
        issues = 0
        for node_id, display, props in dup_rows:
            dupes = json.loads(props or "{}").get("possible_duplicates") or []
            if dupes:
                issues += 1
                print(f"possible duplicate: {display} ({node_id}) ~ {', '.join(dupes)}")
        for edge_id, belief_id in dangling:
            issues += 1
            print(f"dangling edge: {edge_id} -> belief {belief_id} (not active)")
        if not issues:
            print("graph doctor: no issues")
        return 0 if not issues else 1
    finally:
        await memory.close()


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

    p_docs = sub.add_parser("docs", help="Document lifecycle ledger.")
    dsub = p_docs.add_subparsers(dest="docs_cmd", required=True)
    d_scan = dsub.add_parser("scan", help="Scan repo documents into the ledger.")
    d_scan.add_argument(
        "--changed", nargs="*", default=None, metavar="PATH",
        help="Only rescan these paths (no paths: consume the dirty sidecar).",
    )
    d_status = dsub.add_parser("status", help="One-sentence standing of a document.")
    d_status.add_argument("path")
    dsub.add_parser("report", help="Stale/dead-ref/duplicate/archive proposals.")
    d_archive = dsub.add_parser("archive", help="Archive a document (never automatic).")
    d_archive.add_argument("path")
    d_restore = dsub.add_parser("restore", help="Reverse an archive.")
    d_restore.add_argument("path")

    p_graph = sub.add_parser("graph", help="Knowledge-graph tools.")
    gsub = p_graph.add_subparsers(dest="graph_cmd", required=True)
    g_rebuild = gsub.add_parser("rebuild", help="Regenerate graph tables from memories.")
    g_rebuild.add_argument("--from-scratch", action="store_true")
    gsub.add_parser("stats", help="Node/edge counts.")
    g_show = gsub.add_parser("show", help="Entity card + relations.")
    g_show.add_argument("entity")
    g_show.add_argument("--history", action="store_true",
                        help="Include invalidated relations.")
    g_path = gsub.add_parser("path", help="Best relation path between two entities.")
    g_path.add_argument("src")
    g_path.add_argument("dst")
    gsub.add_parser("doctor", help="List possible duplicates and dangling edges.")
    g_merge = gsub.add_parser("merge", help="Merge a duplicate node into its canonical.")
    g_merge.add_argument("dup")
    g_merge.add_argument("canonical")
    g_export = gsub.add_parser("export", help="Export the graph.")
    g_export.add_argument("--format", choices=["mermaid", "dot", "json"],
                          default="mermaid")

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
        "graph": _cmd_graph,
        "docs": _cmd_docs,
    }[args.command]
    raise SystemExit(asyncio.run(handler(args)))


if __name__ == "__main__":
    main()
