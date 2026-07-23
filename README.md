<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/v/clara-memory?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/pyversions/clara-memory?label=Python" alt="Python 3.10+"></a>
</p>

<h1 align="center">CLARA</h1>
<p align="center"><b>Cognitive Living Architecture for Reliable Agents</b></p>

CLARA is a persistent memory layer for AI agents. Its primary mode needs **no
API key, no local LLM, no vector server, and no daemon**: a single SQLite file
with full-text search, exposed to any coding-agent CLI over MCP. An optional
full pipeline adds LLM fact extraction, embeddings, and vector retrieval on
top of the same store.

## Claude Code plugin

The repo doubles as a Claude Code plugin: session-start memory + knowledge-map
injection, six memory MCP tools, a knowledge graph, and a document-lifecycle
curator. Install from the marketplace:

```
/plugin marketplace add itsamiitt/clara
/plugin install clara@clara-marketplace
```

The first session prints `CLARA is installing in the background` while a
bootstrap script builds a private virtualenv (prefers `uv`, falls back to
pip); from the second session on, memory context is injected automatically.
Run `scripts/bootstrap.sh` manually to warm the environment up front, and
`clara doctor` to check plugin health (schema version, flags, ledger counts,
venv, install log).

- **Platforms**: macOS, Linux, and WSL at launch; native Windows best-effort
  (see "Supported platforms").
- **Store choice**: the global store (`~/.clara/clara.db`) shares memory
  across every project and is what the doc ledger keys on (worktrees and
  clones of one repo share a ledger); a per-project store
  (`clara init --project` → `./.clara/clara.db`) isolates memories but
  gives up cross-project recall. Default to global.
- **Kill switches**: `CLARA_GRAPH_ENABLED=0` disables the knowledge graph,
  `CLARA_DOCS_ENABLED=0` the doc curator — independently. Tools report a
  clear disabled error, projection/scan become no-ops, the fastpath omits
  the corresponding context sections, and hooks short-circuit; the memory
  core keeps working with both off.
- **Versioning**: `plugin.json` carries a `version` for marketplace
  installs — set and bump it for releases (commits are the effective
  version while iterating).

## Quick start (coding-agent memory, zero keys)

```bash
pip install "clara-memory[cli]"
clara init
```

`clara init` creates the store (`~/.clara/clara.db`, or `./.clara/clara.db`
with `--project`), builds the FTS5 search index, and prints ready-to-paste
wiring for your agent:

- **Claude Code** — `claude mcp add clara -- clara-mcp`, plus an optional
  SessionStart hook that injects your memory at the start of every session:
  ```json
  {"hooks": {"SessionStart": [{"matcher": "startup|resume|compact",
    "hooks": [{"type": "command", "command": "clara-mcp recall --top-k 12"}]}]}}
  ```
- **OpenAI Codex CLI** — in `~/.codex/config.toml`:
  ```toml
  [mcp_servers.clara]
  command = "clara-mcp"
  ```
- **Gemini CLI / Cursor** — add `{"mcpServers": {"clara": {"command": "clara-mcp"}}}`
  to their MCP config.
- **Anything else** — `clara context "<task>"` prints a context block;
  `clara remember "<text>"` stores facts from plain text. No MCP needed.

The MCP server (`clara-mcp`) exposes **15 tools**: 6 memory tools
(`memory_save`, `memory_search`, `memory_recent`, `memory_update`,
`memory_forget`, `memory_stats`), 5 doc-curator tools (`docs_status`,
`docs_classify`, `docs_supersede`, `docs_fulfill`, `docs_report`), and 4
knowledge-graph tools (`graph_entity`, `graph_neighbors`, `graph_path`,
`memory_link`). **Your agent's own model decides what to remember and
recall** — CLARA does durable typed storage, ranked retrieval, and
prompt-injection-safe context formatting. That's why no key is needed: the
intelligence is the host model you already run.

### When memory is read and written

- **Session start**: the SessionStart hook (or your agent calling
  `memory_search`) injects relevant memories.
- **During a session**: the model calls `memory_save` when it learns
  something durable, `memory_search` when it needs prior context.
- **Housekeeping**: confidence decay and pruning run opportunistically when
  the store is first opened each day — no cron, no background process.

### The `clara` CLI

```
clara init [--project] [--agent NAME]   set up the store + print agent wiring
clara context [QUERY...]                print the memory context block
clara remember TEXT...                  rule-based fact extraction (no LLM) + store
clara list [--query Q] [--type T]       inspect what's stored
clara forget MEMORY_ID [--archive]      retire a memory (never hard-deletes)
clara stats                             store location + counts (JSON)
clara doctor [--quiet] [--deep]         health check: exit 0 ok / 1 degraded / 2 unusable
clara export [--out F] [--type T]...    dump memories as JSONL (portable)
clara import FILE [--on-conflict ...]   load a clara-export file (dedup-aware)
clara backup [--reason R]               rotated store snapshot (VACUUM INTO)
clara restore FILE [--force]            replace the store with a snapshot
clara sync [export|import|status]       bridge to Claude Code native memory
clara statusline                        one-line summary for a statusLine command
clara mcp                               run the MCP stdio server (same as clara-mcp)
```

### Native-memory bridge

`clara sync` keeps CLARA and Claude Code's own memory files coherent:
imports facts from `CLAUDE.md` and the auto-memory directory
(`~/.claude/projects/<project>/memory/`), then writes a marker-fenced
section into `MEMORY.md` (≤ 60 lines, so Claude's own notes keep the native
200-line window) plus a fully CLARA-owned `clara-memory.md` topic file.
Hand-edits inside the fence are imported, never overwritten; the daily
maintenance pass refreshes the export automatically. `/clara:sync` runs it
from inside a session.

### Backups & portability

The daily maintenance pass snapshots the store (`backups/` beside it,
rotation via `CLARA_BACKUP_KEEP`, default 7) and a snapshot is always taken
before a schema migration. `clara export | import` moves memories between
machines as JSONL with ids, timestamps, and provenance intact.

### Statusline

Add to `~/.claude/settings.json` for an at-a-glance memory count:

```json
{"statusLine": {"type": "command", "command": "clara statusline"}}
```

## The two tiers

| | Zero-key tier (default) | Full pipeline (optional) |
|---|---|---|
| Write path | host model calls `memory_save`; `clara remember` uses deterministic rule-based extraction | LLM fact extraction (OpenAI / Anthropic / Ollama) with automatic fallback to the rule-based extractor |
| Read path | SQLite FTS5 (porter-stemmed BM25) + confidence/recency/usage ranking | embeddings + LanceDB vector search, hybrid-degrading to FTS5 on failure |
| Requirements | none | API key or Ollama; LanceDB dir |
| Entry points | `clara`, `clara-mcp`, `LocalMemory` | `ClaraMemory`, FastAPI service |

Both tiers share one SQLite schema: rows written keyless are fully visible to
the full pipeline later, and vice versa.

## Library usage (full pipeline)

```python
from clara.agent import ClaraMemory

agent = await ClaraMemory.create(
    db_url="sqlite+aiosqlite:///clara.db",
    embedding_backend="local",   # "openai" | "local" | "ollama"
    llm_provider="ollama",       # "openai" | "anthropic" | "ollama" | "none"
)

results = await agent.remember("I switched from npm to pnpm")
memories = await agent.recall("package manager", top_k=8)
block = await agent.context_for("package manager")
reply = await agent.interact("What do I use for packages?", user_id="alice")
await agent.close()
```

- `remember()` returns per-fact results (`created` / `reinforced` /
  `superseded` / `retained_both`). If the extraction LLM is unreachable it
  **does not silently store nothing**: it falls back to the rule-based
  extractor and marks results `degraded_heuristic`, or returns an
  `extraction_failed` result naming the cause.
- `llm_provider="none"` selects the rule-based extractor outright.
  `interact()` then runs in memory-only mode: `{"response": None,
  "status": "memory_only", "memory_context": ...}` — the host application's
  model generates replies, CLARA supplies the context.
- Retrieval score: `0.65·similarity + 0.20·confidence + 0.10·recency +
  0.05·usage`. If the vector index is corrupt or unavailable, recall degrades
  to lexical search instead of returning nothing.

## Memory model

Four typed memories in one `memories` table:

- **belief** — atomic facts (`user prefers pnpm`), source-weighted Bayesian
  confidence, reinforcement on re-observation, negation-aware.
- **event** — append-only timeline with a status state machine
  (`created → in_progress → completed/failed/abandoned`). Events are never
  superseded or reinforced — history is history.
- **skill** — procedures with trigger conditions and steps; confidence moves
  with recorded success/failure outcomes; unused skills deprecate after 60
  days of no recorded use.
- **world_model** — entity state (`entity_type` + `name` + `properties`) with
  merge-on-upsert, mutation history, and a partial unique index that makes
  concurrent upserts safe (SQLite).

Conflict resolution (no LLM involved): same subject+relation with opposite
polarity or a different object is a conflict; the new fact supersedes at
confidence > 0.6, both are retained when domains differ or confidence is
ambiguous. Superseded rows are status-flipped and back-linked, never deleted.

Facts extracted from an assistant's own generated text are stored as
`agent_inference` (trust 0.5) — model output can never outrank what the user
actually said.

## Storage & paths

- **SQLite** (WAL, 30s busy timeout) — all memory rows, plus the
  `memories_fts` FTS5 index (porter) and a `memories_fts_tri` trigram twin
  (CJK/substring queries), both maintained by column-scoped triggers.
- **LanceDB** (full tier only) — 1536-dim vectors at `CLARA_LANCE_PATH`
  (default `./clara_vectors`). `LocalMemory`/MCP never create or touch it.
- **One resolver everywhere** (`clara/store.py`): `$CLARA_DB_PATH` (explicit)
  → `<git toplevel>/.clara/clara.db` *if it exists* (created by
  `clara init --project`) → `$CLARA_HOME/clara.db` → `~/.clara/clara.db`.
  The SessionStart hook, the MCP tools, and the CLI all resolve identically,
  so what gets injected is exactly what gets read and written. The doc
  ledger deliberately always lives in the global store (keyed by repo_id, so
  worktrees and clones share one ledger).
- Writes are guarded: secret-looking content is rejected before storage
  (`CLARA_SECRET_POLICY=reject|redact|off`), content is size-capped
  (`CLARA_MAX_CONTENT_BYTES`, default 16 KB), and lock contention retries
  with backoff instead of surfacing "database is locked".

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `CLARA_DB_PATH` / `CLARA_HOME` | `~/.clara/clara.db` | CLI + MCP store location |
| `CLARA_SECRET_POLICY` | `reject` | `reject` / `redact` / `off` — credential patterns on the save path |
| `CLARA_MAX_CONTENT_BYTES` | `16384` | per-memory content size cap |
| `CLARA_BACKUP_KEEP` | `7` | rotated snapshots kept in `backups/` |
| `CLARA_QUERY_EXPANSION` | on | dev-vocabulary synonym expansion (`postgres`↔`postgresql`); `0` disables |
| `CLARA_DB_URL` | `sqlite+aiosqlite:///clara.db` | library/API database URL |
| `CLARA_EMBEDDING_BACKEND` | `openai` | `openai` needs `OPENAI_API_KEY`; `local` needs sentence-transformers; `ollama` needs a running daemon |
| `CLARA_LLM_PROVIDER` | `openai` | also `anthropic`, `ollama`, `none` (rule-based) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | required by the matching providers |
| `CLARA_OPENAI_MODEL` | `gpt-4o-mini` | extraction/reasoning model |
| `CLARA_ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | |
| `CLARA_OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `CLARA_OLLAMA_MODEL` | `llama3.2` | extraction + reasoning + reflection |
| `CLARA_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | |
| `CLARA_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `CLARA_LANCE_PATH` | `./clara_vectors` | vector store dir (full tier) |
| `CLARA_START_SCHEDULER` | `true` | APScheduler decay/pruning/reflection in the library/API profile |
| `CLARA_CACHE_URL` | unset (disabled) | `memory://` or a Redis URL |
| `CLARA_AUTH_REQUIRED` | `false` | API: require `X-User-ID` header (identification only — put real auth in front of it) |
| `CLARA_CORS_ORIGINS` | unset | API: comma-separated allowed origins |

## HTTP API (optional service)

`pip install "clara-memory[api]"`, then `uvicorn clara.main:app`. Routes:
`POST /interact`, `POST /memory/learn`, `GET /memory/search`,
`GET /memory/timeline`, `GET /memory/beliefs`, `GET /memory/{id}`, and
`/admin/stats|conflicts|decay-report|skills/leaderboard|health`. The API's
auth is header-trust only — deploy it behind a gateway, never on a public
interface.

## Reliability

- Every degradation is loud but non-fatal: missing LLM → heuristic extraction
  (`degraded_heuristic`); broken vector index → lexical retrieval; missing
  FTS index → scan fallback; background write failures are retried once and
  then recorded (`BackgroundWriter.stats()`), never silently dropped.
- `clara doctor` checks store writability, SQLite integrity
  (`PRAGMA quick_check`), and the FTS index in milliseconds, and exits
  0/1/2 so a host CLI can gate on it cheaply.
- Failed LanceDB syncs re-queue instead of orphaning vectors.

## Supported platforms

**macOS, Linux, WSL, and native Windows are all first-class.** Every hook
ships as a polyglot `.cmd` dispatcher (line 1 execs the POSIX `sh` body;
`cmd.exe` routes to a native PowerShell 5.1 body under `scripts/win/`), the
MCP server is spawned through a real-executable shim
(`${CLAUDE_PLUGIN_DATA}/shim/clara-mcp` — resolves to `clara-mcp.exe` on
Windows, where stdio servers are spawned without a shell), and CI runs the
full suite plus hook smoke tests on `windows-latest`.

## Development

```bash
pip install -e ".[dev,api,mcp]"
pytest                # default suite (stress tier excluded)
pytest -m stress      # heavy concurrency tier
```

The suite runs entirely without API keys or network — all providers are
faked; SQLite, FTS5, and LanceDB run embedded and real.

## Project documents

- [`CLARA_ADVANCEMENT_PLAN.md`](CLARA_ADVANCEMENT_PLAN.md) — full audit,
  field comparison (Mem0/Letta/Zep/LangMem/ChatGPT/Claude Code), ranked gaps,
  and the roadmap this codebase is executing.
- [`docs/AUDIT_2026-06-14.md`](docs/AUDIT_2026-06-14.md) — prior audit record.
- [`docs/HALLUCINATION_REPORT.md`](docs/HALLUCINATION_REPORT.md) — honest
  scoping of what memory grounding can and cannot claim.

## Not implemented yet

- First-class document storage / chunked document ingestion
- Rich procedural skill graphs
- PostgreSQL (the world-model unique index uses SQLite `json_extract`)
- Embedding tiers below sentence-transformers (a model2vec static-embedding
  tier is designed in the advancement plan)

## License

MIT
