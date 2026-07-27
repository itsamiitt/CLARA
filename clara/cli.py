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
import contextlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clara.integrations.local_memory import LocalMemory

# Heavy imports are deferred into the commands that need them. Importing
# clara.cli used to cost 7.75 s: these four names pull in SQLAlchemy, lancedb
# and the OpenAI SDK, and `clara statusline` -- which needs none of them, only
# the stdlib stats sidecar -- runs on the status bar's refreshInterval, every
# 5 s by default. It could never keep up with itself.

# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

# Ignore only the store's own artifacts; everything else in .clara/ (policy
# exports, the .gitignore itself) stays visible to git.
_PROJECT_GITIGNORE = "clara.db*\n.maintenance*\nbackups/\nquarantine/\n"


def _resolve_db_path(project: bool) -> str:
    if project:
        from clara.store import git_toplevel

        root = git_toplevel(str(Path.cwd()))
        if root is None:
            print(
                "note: not inside a git repository — creating the project store "
                "under the current directory",
                file=sys.stderr,
            )
        # Anchor at the git toplevel: the SessionStart fastpath looks for
        # .clara/clara.db there, so a store created from a subdirectory
        # would otherwise never be read.
        path = Path(root or Path.cwd()) / ".clara" / "clara.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        gitignore = path.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_PROJECT_GITIGNORE, encoding="utf-8")
        elif gitignore.read_text(encoding="utf-8").strip() == "*":
            gitignore.write_text(_PROJECT_GITIGNORE, encoding="utf-8")
            print(
                "warning: rewrote .clara/.gitignore — the old '*' hid every future "
                ".clara file from git; now only clara.db*, .maintenance*, backups/ "
                "and quarantine/ are ignored",
                file=sys.stderr,
            )
        return str(path)
    from clara.integrations.mcp_server import default_db_path

    return default_db_path()


async def _open(db_path: str | None = None) -> LocalMemory:
    """Open the store every other entry point would resolve for this cwd."""
    from clara.integrations.local_memory import LocalMemory

    if db_path is None:
        from clara.store import resolve_store

        db_path = str(resolve_store(create=True).db_path)
    return await LocalMemory.create(db_path)


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
#   store a fact from plain text:      clara remember "we use fly.io for deploys" """,
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
    from clara.extraction.heuristic import HeuristicExtractor

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
            from clara.update.engine import classify_memory_type

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
    from clara.store import orphaned_project_stores, resolve_store

    checks: list[tuple[str, bool, str]] = []
    resolution = resolve_store(create=True)
    db_path = str(resolution.db_path)

    try:
        memory = await _open(db_path)
    except Exception as exc:  # noqa: BLE001 — the whole point is to report it
        if not args.quiet:
            print(f"unusable: cannot open store at {db_path}: {exc}")
        return 2

    try:
        async with memory.engine.connect() as conn:
            check_sql = (
                "PRAGMA integrity_check" if getattr(args, "deep", False)
                else "PRAGMA quick_check(1)"
            )
            integrity = (await conn.execute(sa_text(check_sql))).scalar()
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

    from clara.db.backup import latest_backup

    newest = latest_backup(db_path)
    if newest is not None:
        import time as _time

        age_h = (_time.time() - newest.stat().st_mtime) / 3600
        checks.append(("backups", True, f"newest {age_h:.1f}h old ({newest.name})"))
    else:
        import time as _time

        store_age_h = (_time.time() - Path(db_path).stat().st_mtime) / 3600
        # A brand-new store legitimately has no backups; only an established
        # store without one is a degradation worth flagging.
        overdue = store_age_h > 48
        checks.append((
            "backups",
            not overdue,
            "none yet — the daily maintenance pass takes one, or run `clara backup`",
        ))

    orphans = orphaned_project_stores(str(Path.cwd()), Path(db_path))
    if orphans:
        checks.append((
            "project stores",
            False,
            f"orphaned store(s) never read by the resolver: "
            f"{', '.join(str(o) for o in orphans)} — move to the git toplevel "
            f".clara/ or delete",
        ))

    hard_failures = [c for c in checks if not c[1] and c[0] not in
                     ("fts5 index", "backups", "project stores")]
    degraded = [c for c in checks if not c[1]]

    if not args.quiet:
        print(f"store: {db_path} (scope: {resolution.scope})")
        for name, ok, detail in checks:
            print(f"  [{'ok' if ok else '!!'}] {name}: {detail}")
        _print_plugin_health(db_path)
        # A corrupt store is the one moment the user most needs to be told what
        # to do, and doctor used to stop at the diagnosis: it printed the
        # integrity failure and, separately, that a backup existed, and left
        # the reader to connect them. Name the command.
        if hard_failures:
            print("\nwhat to do:")
            if newest is not None:
                print(f"  clara restore {newest}")
                print("  (a pre-restore backup of the current file is taken first)")
            else:
                print(
                    f"  no backup found in {Path(db_path).parent / 'backups'} — "
                    "export what still reads with `clara export --out rescue.jsonl`,"
                )
                print(
                    "  then start fresh with `clara init` and "
                    "`clara import rescue.jsonl`"
                )
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
    from clara.integrations.mcp_server import default_db_path
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
        async with memory.session_factory() as session:
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
                        # Plain equality: belief ids are canonical dashless hex
                        # on both sides (clara.core.ids.canonical_id, migration
                        # 8). Normalising here would defeat the index and, worse,
                        # hide a non-canonical writer from this very check.
                        "SELECT edge_id, belief_id FROM graph_edges "
                        "WHERE invalid_at IS NULL AND belief_id IS NOT NULL "
                        "AND NOT EXISTS (SELECT 1 FROM memories m WHERE "
                        "m.memory_id = graph_edges.belief_id "
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
# clara export / import / backup / restore / statusline
# ---------------------------------------------------------------------------


async def _cmd_export(args: argparse.Namespace) -> int:
    from clara.porting import export_records, write_export
    from clara.store import resolve_store

    db_path = str(resolve_store(create=False).db_path)
    if not Path(db_path).is_file():
        print(f"no store at {db_path} — nothing to export", file=sys.stderr)
        return 1
    records = export_records(
        db_path,
        types=args.types,
        status=args.status,
        since=args.since,
        include_graph=args.include_graph,
        include_docs=args.include_docs,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            written = write_export(records, handle)
        print(f"exported {written - 1} records to {args.out}", file=sys.stderr)
    else:
        write_export(records, sys.stdout)
    return 0


async def _cmd_import(args: argparse.Namespace) -> int:
    from clara.porting import ImportError_, import_records
    from clara.store import resolve_store

    db_path = str(resolve_store(create=True).db_path)
    try:
        with open(args.file, encoding="utf-8") as handle:
            stats = import_records(
                db_path,
                handle,
                on_conflict=args.on_conflict,
                dry_run=args.dry_run,
                allow_secrets=args.allow_secrets,
            )
    except (OSError, ImportError_) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    prefix = "would import" if args.dry_run else "imported"
    print(
        f"{prefix} {stats['imported']}, skipped-id {stats['skipped_id']}, "
        f"skipped-content {stats['skipped_content']}, "
        f"conflicts-updated {stats['conflicts_updated']}, "
        f"skipped-secret {stats['skipped_secret']}"
    )
    if stats.get("skipped_malformed"):
        print(
            f"warning: skipped {stats['skipped_malformed']} malformed line(s) — "
            "the file may be truncated or corrupt",
            file=sys.stderr,
        )
    if stats.get("other_kinds"):
        # Graph rows rebuild from memories; doc-ledger rows (attestations,
        # classifications) do not, so their loss is not silent.
        print(
            f"note: {stats['other_kinds']} non-memory record(s) (graph/docs) "
            "were not imported; graph rebuilds from memories, but doc-ledger "
            "judgments do not — re-run `clara docs scan` on the target repo.",
            file=sys.stderr,
        )
    if stats["imported"] or stats["conflicts_updated"]:
        print("hint: run `clara graph rebuild` to project imported memories "
              "into the knowledge graph")
    return 0


async def _cmd_project(args: argparse.Namespace) -> int:
    """Print what CLARA can tell about this project from its manifests."""
    from clara.project import detect_project
    from clara.store import git_toplevel

    root = args.path or git_toplevel(str(Path.cwd())) or str(Path.cwd())
    profile = detect_project(root)
    if args.json:
        summary = profile.summary()
        summary["evidence"] = [
            {"category": f.category, "value": f.value, "source": f.evidence}
            for f in profile.facts
        ]
        print(json.dumps(summary, indent=2))
        return 0

    print(f"project: {profile.name or '(unnamed)'}")
    print(f"  root: {profile.root}")
    if profile.is_monorepo:
        print(f"  monorepo workspaces: {', '.join(profile.workspaces)}")
    detected = profile.summary()["detected"]
    if not detected:
        print("  nothing detected — no recognised manifest at this path")
        return 0
    for category in sorted(detected):
        print(f"  {category.replace('_', ' ')}: {', '.join(detected[category])}")
    if args.evidence:
        print("\nevidence:")
        for fact in profile.facts:
            print(f"  {fact.category}={fact.value}  <- {fact.evidence}")
    return 0


async def _cmd_maintain(args: argparse.Namespace) -> int:
    """Run the daily housekeeping pass by hand.

    The pass normally rides the first store access of the day *by the MCP
    server*, which covers every Claude Code session. A store driven only by
    this CLI was never maintained at all -- no decay, no pruning, no rotated
    backup -- despite the README promising it. This is the explicit way to run
    it, and the only way for a CLI-only setup.
    """
    from clara.maintenance import run_if_due
    from clara.store import resolve_store

    resolution = resolve_store(create=True)
    db_path = str(resolution.db_path)
    memory = await _open(db_path)
    try:
        summary = await run_if_due(
            memory, db_path, anchor=str(Path.cwd()), force=args.force
        )
    finally:
        await memory.close()
    if summary is None:
        print("maintenance already ran within the last 24h — use --force to run it now")
        return 0
    print(f"store: {db_path}")
    print(summary)
    return 0


async def _cmd_uninstall(args: argparse.Namespace) -> int:
    """Remove plugin runtime state; keep memories unless --purge-memories."""
    import os as _os

    base = Path(_os.environ.get("CLARA_HOME") or (Path.home() / ".clara"))
    data_dir = Path(_os.environ.get("CLAUDE_PLUGIN_DATA") or (base / "plugin"))

    removed: list[str] = []
    # Runtime artifacts only: venvs, the shim, the "current" pointer, logs, the
    # per-session sidecar dirs, and the caches rebuilt from the store. The
    # memory DB is deliberately excluded.
    #
    # pytools is the private CPython the bootstraps provision. It normally sits
    # under data_dir, but a layout where CLAUDE_PLUGIN_DATA is the base itself
    # puts it directly under base, where removing data_dir misses it. Measured
    # on this machine: 15,147 files left behind by "uninstall".
    #
    # quarantine/ and proposals/ are projections of the doc ledger, which lives
    # in the store. They are regenerated by `clara docs scan` and the fastpath,
    # so keeping them after the runtime is gone only leaves stale fuel for hooks
    # that are no longer installed.
    runtime_targets = [
        data_dir,
        base / "pytools",
        base / "session-flags",
        base / "session-cwd",
        base / "quarantine",
        base / "proposals",
    ]
    for target in runtime_targets:
        if target.exists():
            try:
                shutil.rmtree(target)
                removed.append(str(target))
            except OSError as exc:
                print(f"warning: could not remove {target}: {exc}", file=sys.stderr)

    if args.purge_memories:
        # clara.db.stats is the status-line sidecar (clara/stats_cache.py). It
        # holds counts derived from the store, so leaving it behind means a
        # command whose whole promise is "irreversibly delete the memories"
        # leaves a file describing them -- and the status line then reports a
        # count for a store that no longer exists.
        for name in (
            "clara.db",
            "clara.db-wal",
            "clara.db-shm",
            "clara.db.stats",
            "backups",
        ):
            target = base / name
            if target.exists():
                try:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    removed.append(str(target))
                except OSError as exc:
                    print(f"warning: could not remove {target}: {exc}", file=sys.stderr)

    if removed:
        print("removed:")
        for path in removed:
            print(f"  {path}")
    else:
        print("nothing to remove (already clean)")
    if not args.purge_memories:
        print(f"memories kept at {base / 'clara.db'} "
              "(re-run with --purge-memories to delete them)")
    return 0


async def _cmd_backup(args: argparse.Namespace) -> int:
    from clara.db.backup import backup_db
    from clara.store import resolve_store

    db_path = str(resolve_store(create=False).db_path)
    if not Path(db_path).is_file():
        print(f"no store at {db_path} — nothing to back up", file=sys.stderr)
        return 1
    dest = backup_db(db_path, reason=args.reason)
    if dest is None:
        print("backup failed (see log)", file=sys.stderr)
        return 1
    print(f"backup written: {dest}")
    return 0


async def _cmd_restore(args: argparse.Namespace) -> int:
    import sqlite3 as _sqlite3

    from clara.db.backup import backup_db
    from clara.db.migrations import SCHEMA_VERSION, get_version
    from clara.store import resolve_store

    source = Path(args.file)
    if not source.is_file():
        print(f"no such file: {source}", file=sys.stderr)
        return 1
    try:
        conn = _sqlite3.connect(source)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            version = get_version(conn)
        finally:
            conn.close()
    except _sqlite3.DatabaseError as exc:
        # Pointing restore at a .zip, a .jsonl export, or a half-downloaded file
        # is a user mistake, not an internal failure. Uncaught, this reached the
        # top-level handler and exited 70 (EX_SOFTWARE), which this CLI documents
        # as "unexpected internal failure" — telling the user to report a bug
        # when the fix is to pass a different path.
        print(
            f"refusing to restore: {source} is not a CLARA store ({exc}).\n"
            f"       Backups live in <store dir>/backups/ and are named "
            f"clara-<timestamp>-<reason>.db",
            file=sys.stderr,
        )
        return 1
    if integrity != "ok":
        print(f"refusing to restore: source failed integrity_check ({integrity})",
              file=sys.stderr)
        return 1
    if version > SCHEMA_VERSION:
        print(f"refusing to restore: source schema v{version} is newer than this "
              f"CLARA supports (v{SCHEMA_VERSION})", file=sys.stderr)
        return 1

    target = resolve_store(create=True).db_path
    if target.is_file() and not args.force:
        print(f"this will replace {target} with {source}.")
        print("re-run with --force to proceed (a pre-restore backup is taken).")
        return 1
    if target.is_file():
        backup_db(str(target), reason="pre-restore")
    # WAL side files from the old store are stale after a swap.
    for side in (f"{target}-wal", f"{target}-shm"):
        Path(side).unlink(missing_ok=True)
    tmp = target.with_suffix(".restore-tmp")
    shutil.copyfile(source, tmp)
    import os as _os

    _os.replace(tmp, target)
    print(f"restored {target} from {source}")
    print("restart any running Claude Code sessions so the MCP server reopens the store.")
    return 0


def _cmd_statusline_install(args: argparse.Namespace) -> int:
    """Register or remove CLARA's status line (see clara.statusline_setup).

    Users who never open a terminal reach the same code through the
    `statusline_install` MCP tool — the plugin's `clara` executable is
    deliberately not on their PATH.
    """
    from clara import statusline_setup

    if args.uninstall:
        result = statusline_setup.uninstall()
        if not result["ok"]:
            print(f"error: {result.get('error')}", file=sys.stderr)
            return 2
        if result["action"] == "absent":
            print("no CLARA status line is configured — nothing to remove.")
        else:
            print(f"removed CLARA from the status bar ({result['path']})")
        return 0

    result = statusline_setup.install(
        refresh_interval=args.refresh_interval, force=args.force
    )
    if result["action"] == "blocked":
        print(
            "a different statusLine is already configured:\n"
            f"  {result['existing_command']}\n"
            "re-run with --force to replace it.",
            file=sys.stderr,
        )
        return 2
    if not result["ok"]:
        print(f"error: {result.get('error')}", file=sys.stderr)
        return 2
    print(f"status bar configured ({result['path']})")
    print(f"  command: {result['command']}")
    print(f"  refresh: every {result['refresh_interval']}s")
    print("start a new Claude Code session to see it.")
    return 0


async def _cmd_statusline(args: argparse.Namespace) -> int:
    """Statusline provider: `CLARA - N memories - scope`.

    Reads the statusline JSON (for the session cwd) from stdin. Counts come
    from the sidecar counter that every write path refreshes
    (:mod:`clara.stats_cache`), so this never opens SQLite on the status-line
    cadence; it falls back to a single count query only when the sidecar is
    missing or stale.

    Always prints exactly one line and always exits 0: Claude Code blanks the
    status bar when the command exits non-zero or produces no output.
    """
    import json as _json

    from clara import stats_cache
    from clara.store import resolve_store

    if args.install or args.uninstall:
        return _cmd_statusline_install(args)

    anchor = None
    if not args.no_stdin and not sys.stdin.isatty():
        try:
            payload = _json.loads(sys.stdin.read() or "{}")
            anchor = (
                payload.get("cwd")
                or payload.get("workspace", {}).get("current_dir")
                or None
            )
        except (ValueError, AttributeError):
            anchor = None
    res = resolve_store(anchor, create=False)
    if not res.exists:
        print("CLARA - no store")
        return 0
    count = stats_cache.read(res.db_path)
    if count is None:
        # Cold sidecar (older store, or a writer that died before refreshing):
        # recount once and repopulate it. Every later pull is a file read.
        count = stats_cache.refresh(res.db_path)
    if count is None:
        print("CLARA - store busy")
        return 0
    label = "memory" if count == 1 else "memories"
    print(f"CLARA - {count} {label} - {res.scope}")
    return 0


async def _cmd_sync(args: argparse.Namespace) -> int:
    from clara.bridge import paths as bridge_paths
    from clara.bridge.exporter import export_native
    from clara.bridge.importer import import_native
    from clara.store import resolve_store

    anchor = str(Path.cwd())
    res = resolve_store(anchor, create=True)
    db_path = str(res.db_path)

    if args.mode == "status":
        memory_md = bridge_paths.memory_md_path(anchor)
        print(f"store: {db_path} (scope: {res.scope})")
        print(f"auto-memory enabled: {bridge_paths.auto_memory_enabled()}")
        print(f"MEMORY.md target: {memory_md if memory_md else '(disabled)'}")
        print(f"topic file: {bridge_paths.topic_file_path(anchor) or '(disabled)'}")
        sources = bridge_paths.claude_md_paths(anchor)
        print(f"import sources: {', '.join(str(s) for s in sources) or '(none found)'}")
        return 0

    # Import BEFORE export: pre-existing native notes (and any hand-edited
    # fence lines) must be captured before the fence is (re)generated.
    if args.mode in (None, "import"):
        memory = await _open(db_path)
        try:
            results = await import_native(memory, anchor, verbatim=args.verbatim)
        finally:
            await memory.close()
        imported = sum(r["imported"] for r in results.values())
        skipped = sum(r["skipped_dup"] + r["skipped_no_extract"] for r in results.values())
        print(f"import: {imported} new memories from {len(results)} file(s), "
              f"{skipped} lines skipped")
        no_extract = sum(r["skipped_no_extract"] for r in results.values())
        if no_extract and not args.verbatim:
            print(f"  ({no_extract} lines did not extract — re-run with --verbatim "
                  "to store them as plain notes)")

    if args.mode in (None, "export"):
        summary = export_native(db_path, anchor)
        print(f"export: {summary}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class _QuietFormatter(logging.Formatter):
    """Log the message; keep the traceback for CLARA_DEBUG=1.

    Nothing configured logging for the CLI, so Python's last-resort handler
    printed WARNING records to stderr *with* their exc_info. A corrupt store
    therefore greeted the user with a SQLite traceback from
    _ensure_versioned_schema before any of doctor's readable output — a stack
    trace at the exact moment they are least equipped to read one. The
    exception is still attached to the record and still shown under
    CLARA_DEBUG=1; only the default rendering drops it.
    """

    def formatException(self, ei: Any) -> str:  # noqa: N802 — logging's name
        return ""

    def format(self, record: logging.LogRecord) -> str:
        record.exc_text = ""
        return super().format(record)


def _configure_logging() -> None:
    debug = os.environ.get("CLARA_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    handler = logging.StreamHandler(sys.stderr)
    if debug:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    else:
        handler.setFormatter(_QuietFormatter("clara: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.WARNING)


def _make_output_lossy() -> None:
    """Never let an un-encodable character fail a command.

    CLARA's own output uses a handful of non-ASCII glyphs (em dashes, arrows,
    check marks). Windows consoles still run legacy code pages, and anything
    under ``LC_ALL=C`` or ``PYTHONIOENCODING=ascii`` gets an ASCII stdout.
    Printing an em dash there raises UnicodeEncodeError, which argparse does
    not catch: verified before this existed, ``clara sync`` died with

        error: 'ascii' codec can't encode character '\\u2014' ... (exit 70)

    after having already done the import — so the work succeeded and the
    command still reported failure. Degrading one glyph to "?" is strictly
    better than failing a command that worked.

    Mirrors _make_stdout_lossy in clara/fastpath/context.py, which fixed the
    same class of bug on the session-start path. stderr is reconfigured too:
    that is where the error text above was written.
    """
    for stream in (sys.stdout, sys.stderr):
        # Not every stream is reconfigurable (a pipe, pytest's capture object);
        # that is fine, it just means there is nothing to harden here.
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> None:
    _make_output_lossy()
    _configure_logging()
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
    p_doc.add_argument("--deep", action="store_true",
                       help="Full PRAGMA integrity_check instead of quick_check.")

    p_export = sub.add_parser("export", help="Export memories as JSONL (stdout default).")
    p_export.add_argument("--out", default=None, metavar="FILE")
    p_export.add_argument("--type", action="append", dest="types",
                          choices=["belief", "event", "skill", "world_model"])
    p_export.add_argument("--status", choices=["active", "all"], default="all")
    p_export.add_argument("--since", default=None, metavar="ISO",
                          help="Only rows updated at/after this ISO timestamp.")
    p_export.add_argument("--include-graph", action="store_true")
    p_export.add_argument("--include-docs", action="store_true")

    p_import = sub.add_parser("import", help="Import a clara-export JSONL file.")
    p_import.add_argument("file")
    p_import.add_argument("--on-conflict", choices=["skip", "newest", "force"],
                          default="skip")
    p_import.add_argument("--dry-run", action="store_true")
    p_import.add_argument("--allow-secrets", action="store_true",
                          help="Import rows even when they match secret patterns.")

    p_backup = sub.add_parser("backup", help="Snapshot the store (rotated).")
    p_backup.add_argument("--reason", default="manual")

    p_restore = sub.add_parser("restore", help="Replace the store with a backup/snapshot.")
    p_restore.add_argument("file")
    p_restore.add_argument("--force", action="store_true",
                           help="Skip the confirmation prompt.")

    p_status_line = sub.add_parser(
        "statusline",
        help="One-line store summary for a Claude Code statusLine command.",
    )
    p_status_line.add_argument("--no-stdin", action="store_true",
                               help="Ignore statusline JSON on stdin (use cwd).")
    p_status_line.add_argument(
        "--install", action="store_true",
        help="Add CLARA's memory counter to your Claude Code status bar.",
    )
    p_status_line.add_argument(
        "--uninstall", action="store_true",
        help="Remove CLARA's entry from the Claude Code status bar.",
    )
    p_status_line.add_argument(
        "--refresh-interval", type=int, default=5, metavar="SECONDS",
        help="How often the bar re-runs the counter (default: 5).",
    )
    p_status_line.add_argument(
        "--force", action="store_true",
        help="Replace an existing statusLine that is not CLARA's.",
    )

    p_sync = sub.add_parser(
        "sync", help="Sync with Claude Code native memory (import then export)."
    )
    p_sync.add_argument("mode", nargs="?", choices=["export", "import", "status"],
                        default=None,
                        help="Run only one direction, or show resolved paths.")
    p_sync.add_argument("--verbatim", action="store_true",
                        help="Also store lines the extractor cannot parse "
                             "(as note/states beliefs).")

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

    p_project = sub.add_parser(
        "project", help="Show what this project is (language, frameworks, tooling)."
    )
    p_project.add_argument("path", nargs="?", help="Repo root (default: current repo).")
    p_project.add_argument("--json", action="store_true", help="Machine-readable output.")
    p_project.add_argument("--evidence", action="store_true",
                           help="Show the file each fact came from.")

    p_maintain = sub.add_parser(
        "maintain",
        help="Run housekeeping now: backup, decay, pruning, graph, native export.",
    )
    p_maintain.add_argument(
        "--force", action="store_true",
        help="Run even if a pass already ran in the last 24h.",
    )

    p_uninstall = sub.add_parser(
        "uninstall",
        help="Remove CLARA's private venv, shim, and install log (keeps memories).",
    )
    p_uninstall.add_argument(
        "--purge-memories",
        action="store_true",
        help="ALSO delete the memory store and backups (irreversible).",
    )

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
        "export": _cmd_export,
        "import": _cmd_import,
        "backup": _cmd_backup,
        "restore": _cmd_restore,
        "statusline": _cmd_statusline,
        "sync": _cmd_sync,
        "project": _cmd_project,
        "maintain": _cmd_maintain,
        "uninstall": _cmd_uninstall,
    }[args.command]
    try:
        raise SystemExit(asyncio.run(handler(args)))
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 — top-level: no raw traceback to users
        # SQLite's "attempt to write a readonly database" arrives wrapped in a
        # SQLAlchemy dump of the INSERT and every bound parameter — true about
        # the cause, useless about the fix. But it has two very different
        # causes, and an earlier version of this handler blamed the wrong one:
        # it reported every occurrence as "written by a newer version of CLARA"
        # and told the user to upgrade, which for a chmod 444 store is advice
        # that cannot work.
        #
        # LocalMemory._require_writable() now intercepts the schema-too-new case
        # before any SQL runs, with its own message. So reaching SQLite's error
        # means the *file* is unwritable: permissions, a read-only mount, or a
        # store on read-only media.
        from clara.integrations.local_memory import StoreReadOnly

        if isinstance(exc, StoreReadOnly):
            # A refusal, not a crash: exit 1 like any other failed operation.
            # It reached the generic handler and exited 70 (EX_SOFTWARE, which
            # this CLI documents as an internal failure) until this branch
            # existed, telling the user to report a bug about a deliberate
            # safety check.
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if "readonly database" in str(exc):
            from clara.store import resolve_store

            hint = ""
            with contextlib.suppress(Exception):
                path = resolve_store().db_path
                writable = os.access(path, os.W_OK)
                hint = (
                    f"\n       store: {path}"
                    f"\n       the file is {'writable' if writable else 'READ-ONLY'}"
                    " to this user"
                )
            print(
                "error: the store file cannot be written, so nothing was saved."
                f"{hint}\n"
                "       Check its permissions (chmod u+w on macOS/Linux, clear the\n"
                "       read-only attribute on Windows) and that the disk is not full\n"
                "       or mounted read-only. Reading keeps working meanwhile.",
                file=sys.stderr,
            )
            logging.getLogger(__name__).debug("read-only store write", exc_info=True)
            raise SystemExit(1) from exc
        # Exit code 70 (EX_SOFTWARE) for an unexpected internal failure, kept
        # distinct from 1 (operation failed) and 2 (usage/precondition error).
        print(f"error: {exc}", file=sys.stderr)
        logging.getLogger(__name__).debug("unhandled CLI error", exc_info=True)
        raise SystemExit(70) from exc


if __name__ == "__main__":
    main()
