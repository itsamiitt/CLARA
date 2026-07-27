<!-- Badges: PyPI badges intentionally omitted until the package is published
     (they would render "not found"). Re-add once a v* tag ships to PyPI. -->

<h1 align="center">CLARA</h1>
<p align="center"><b>Cognitive Living Architecture for Reliable Agents</b></p>

CLARA is a persistent memory layer for AI agents. Its primary mode needs **no
API key, no local LLM, no vector server, and no daemon**: a single SQLite file
with full-text search, exposed to any coding-agent CLI over MCP. An optional
full pipeline adds LLM fact extraction, embeddings, and vector retrieval on
top of the same store.

CLARA doubles as a **Claude Code plugin** — session-start memory + knowledge-map
injection, 18 MCP tools, a knowledge graph, and a document-lifecycle curator —
and as a **standalone CLI + MCP server** for any agent (Codex, Gemini, Cursor,
or your own). Both use the same single SQLite store.

---

# Installation

> **Requirements:** `git`, and Python **3.10+**.
>
> On **native Windows you do not need to install Python yourself** — if none is
> found, CLARA downloads a private, checksum-verified CPython into its own data
> directory (no installer, no admin rights, no change to your `PATH`). This is
> what makes it work on locked-down machines where `winget install` is blocked
> by policy. Opt out with `CLARA_NO_AUTO_PYTHON=1`.
>
> On **macOS/Linux**, install Python 3.10+ with your package manager if it is
> missing (`brew install python@3.12`, `sudo apt-get install -y python3
> python3-venv`, …) — the bootstrap prints the exact command for your system.
>
> Optionally [`uv`](https://github.com/astral-sh/uv) (bootstrap uses it when
> present and falls back to `pip` otherwise). On native Windows the plugin
> hooks run under either PowerShell or Git Bash — both are supported.
>
> **PyPI note:** `clara-memory` is not published to PyPI yet, so `pip install
> clara-memory` will not resolve. Install from the plugin marketplace (which
> builds its own private environment) or from source, as shown below. A tagged
> PyPI release is planned.

## Option A — Claude Code plugin (recommended)

From inside Claude Code:

```
/plugin marketplace add itsamiitt/clara
/plugin install clara@clara-marketplace
```

**What happens on first use.** The plugin ships a bootstrap that builds a
private virtualenv from the plugin's own checkout (no PyPI needed) into
`${CLAUDE_PLUGIN_DATA}/` (survives updates). The install runs in the
background, so your **first session** prints:

```
CLARA is installing in the background — memory will be available next session.
```

The build takes roughly one to two minutes, once per plugin version (measured:
113 s on a Windows machine using the bundled interpreter; a warm pip cache and
a faster disk cut it). **From the second session on**, your memory context is
injected automatically at session start and the `memory_*` tools are live.
Nothing blocks the session while it installs.

Because the `memory` MCP server points at a shim the bootstrap creates, that
server is not available during the very first session (and briefly into the
second if the build is still running) — expected, not an error. Warm it up
front to skip the wait.

**Warm it up front (optional)** so the first session already has memory:

```bash
# macOS / Linux / WSL / Git Bash
CLAUDE_PLUGIN_ROOT="$(pwd)" sh scripts/bootstrap.sh   # from the plugin checkout
```
```powershell
# native Windows PowerShell
$env:CLAUDE_PLUGIN_ROOT = (Get-Location).Path
powershell -ExecutionPolicy Bypass -File scripts\win\bootstrap.ps1
```

**Verify plugin health at any time:**

```bash
clara doctor          # exit 0 healthy / 1 degraded / 2 unusable
```
It prints schema version, kill-switch flags, ledger counts, the resolved venv,
and the tail of the install log — the first place to look if memory is missing.

## Option B — standalone CLI + MCP server (any agent)

Install from source (until the PyPI release lands):

```bash
# from a clone
git clone https://github.com/itsamiitt/CLARA.git && cd CLARA
pip install -e ".[cli]"        # or ".[mcp]" for the MCP server, ".[full]" for the LLM tier

# or directly from git
pip install "clara-memory[cli] @ git+https://github.com/itsamiitt/CLARA.git"
```

This puts two commands on your `PATH`: **`clara`** (the CLI) and **`clara-mcp`**
(the MCP stdio server). Then:

```bash
clara init            # create the store + print ready-to-paste agent wiring
```

`clara init` creates the store (`~/.clara/clara.db`, or `./.clara/clara.db` with
`--project`), builds the FTS5 index, and prints wiring for common agents:

- **Claude Code (manual, without the plugin)** — register the MCP server and add
  a SessionStart hook that injects memory:
  ```bash
  claude mcp add clara -- clara-mcp
  ```
  ```json
  {"hooks": {"SessionStart": [{"matcher": "startup|resume|clear|compact",
    "hooks": [{"type": "command", "command": "clara-mcp recall --top-k 12"}]}]}}
  ```
- **OpenAI Codex CLI** — in `~/.codex/config.toml`:
  ```toml
  [mcp_servers.clara]
  command = "clara-mcp"
  ```
- **Gemini CLI / Cursor** — add to their MCP config:
  ```json
  {"mcpServers": {"clara": {"command": "clara-mcp"}}}
  ```
- **Anything that can shell out** — `clara context "<task>"` prints a context
  block; `clara remember "<text>"` stores facts from plain text. No MCP needed.

## Verify it works

In Claude Code, run `/clara:doctor` — it works for a plugin install and needs
nothing on your `PATH`.

From a terminal, the same checks:

```bash
clara doctor                 # store health
echo '{}' | clara statusline # e.g. "CLARA - 0 memories - global"
clara-mcp recall --top-k 3   # prints your memory context block (empty at first)
```

**A plugin-only install does not put `clara` on your `PATH`.** The bootstrap
keeps the CLI inside the plugin's private venv and shims it next to the MCP
server, so use that path (or install Option B alongside):

```bash
~/.clara/plugin/shim/clara doctor          # macOS / Linux / WSL / Git Bash
%USERPROFILE%\.clara\plugin\shim\clara.exe doctor   # Windows
```

---

# Usage

## The 18 MCP tools

Once wired, your agent's model calls these directly — **CLARA is storage +
ranked retrieval; the host model is the intelligence** (that is why no API key
is needed).

- **Memory (6):** `memory_save`, `memory_search`, `memory_recent`,
  `memory_update`, `memory_forget`, `memory_stats`.
- **Docs curator (5):** `docs_status`, `docs_classify`, `docs_supersede`,
  `docs_fulfill`, `docs_report`.
- **Knowledge graph (4):** `graph_entity`, `graph_neighbors`, `graph_path`,
  `memory_link`.
- **Status bar (2):** `statusline_install`, `statusline_status` — used by
  `/clara:statusline` to put the live memory counter in your status bar.
- **Project (1):** `project_profile` — language, package manager, frameworks,
  build/test tooling and monorepo layout, read from the repo's own manifests.

## Slash commands (plugin)

| Command | What it does |
|---|---|
| `/clara:remember <fact>` | store one durable fact (chooses the memory type) |
| `/clara:recall <topic>` | search memory and answer from the hits |
| `/clara:memories [n]` | show store stats + the most recent memories |
| `/clara:forget <id\|desc>` | retire a memory (never hard-deletes) |
| `/clara:graph [entity]` | entity card + relations, or graph stats |
| `/clara:docs [path]` | a document's standing, or the repo rot report |
| `/clara:docs-review [scope]` | interactive doc-rot review → one reviewable commit |
| `/clara:done [plan]` | close out a completed plan, distilling it into memory |
| `/clara:sync [export\|import\|status]` | bridge to Claude Code native memory |
| `/clara:statusline [on\|off]` | show the live memory counter in the status bar |
| `/clara:doctor` | health check that works without `clara` on your `PATH` |

## How memory flows

- **Session start** — the SessionStart hook injects a ranked `=== MEMORY
  CONTEXT ===` block (top memories) plus a `[KNOWLEDGE MAP]` of authoritative
  docs. Re-injected after `/clear` and `/compact`.
- **During the session** — the model calls `memory_save` when it learns
  something durable and `memory_search` when it needs prior context. You can
  also drive it explicitly with the slash commands above.
- **Housekeeping** — confidence decay, pruning, a rotated backup, and the
  native-memory export run opportunistically the first time the store is opened
  each day. No cron, no daemon.

## Storing your first memory (CLI, zero keys)

```bash
clara remember "I use pnpm"                    # -> belief: user uses pnpm
clara remember "We deployed the API to fly.io" # -> event
clara remember "The database runs on postgres" # -> world_model
clara context "package manager"                # -> ranked context block
clara list                                     # inspect what's stored
```

The rule-based extractor (`clara remember`) is **precision-first** and matches
canonical first-person forms (`I use X`, `we deployed Y`, `Z runs on W`,
`I switched from A to B`). For richer capture, let a model call `memory_save`
with explicit `subject`/`relation`/`object`, or enable the optional LLM tier.

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
clara project [PATH] [--evidence]       what this project is (language, frameworks, tooling)
clara statusline                        one-line summary for a statusLine command
clara statusline --install              add the live memory counter to your status bar
                                        (in Claude Code, use /clara:statusline instead)
clara mcp                               run the MCP stdio server (same as clara-mcp)
clara uninstall [--purge-memories]      remove the private venv/shim/logs (keeps memories)

clara docs scan|status|report|archive|restore    doc-curator ledger operations
clara graph rebuild|stats|show|path|doctor|merge|export   knowledge-graph operations
```

The `docs` and `graph` groups are also reachable as the `/clara:docs` and
`/clara:graph` slash commands inside Claude Code.

### Native-memory bridge

`clara sync` keeps CLARA and Claude Code's own memory files coherent:
imports facts from `CLAUDE.md` and the auto-memory directory
(`~/.claude/projects/<project>/memory/`), then writes a marker-fenced
section into `MEMORY.md` (≤ 60 lines, so Claude's own notes keep the native
200-line window) plus a fully CLARA-owned `clara-memory.md` topic file.
Your own notes are never overwritten. Anything **outside** the fence — in
`MEMORY.md`, `CLAUDE.md`, or the auto-memory files — is imported on the next
sync. Lines added **inside** the fence are deliberately not imported (they are
CLARA's own export, so re-importing them would loop); CLARA leaves them alone
and stops refreshing that section until they are removed, and says so. Put
notes outside the fence, or save them with `clara remember` / `/clara:remember`.
The daily maintenance pass refreshes the export automatically. `/clara:sync`
runs it from inside a session.

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

## Configuration

### Global vs project store

One resolver decides the store for every entry point (hook, MCP, CLI), so what
gets injected is exactly what gets read and written. Order (first match wins):

1. `CLARA_DB_PATH` — explicit override, used verbatim.
2. **Project store** — `<git toplevel>/.clara/clara.db`, *if that file exists*
   (create it with `clara init --project`). Isolates a repo's memory but gives
   up cross-project recall.
3. **Global store** — `$CLARA_HOME/clara.db` else `~/.clara/clara.db`. **Default.**
   Shared across every project; the doc ledger always lives here (keyed by repo
   identity, so worktrees and clones of one repo share one ledger).

`clara stats` and `clara doctor` print the resolved store and its scope.

### Kill switches

Disable the add-ons independently; the memory core keeps working with both off.
Accepts `0/false/no/off` (any case):

```bash
CLARA_GRAPH_ENABLED=0   # no knowledge graph: graph tools return a clear
                        # disabled error, projection is a no-op, [GRAPH] omitted
CLARA_DOCS_ENABLED=0    # no doc curator: docs tools disabled, scan no-ops,
                        # [KNOWLEDGE MAP] omitted, doc hooks short-circuit
```

### Write-path guardrails

```bash
CLARA_SECRET_POLICY=reject      # reject (default) | redact | off — refuses to
                                # store credential-shaped content (store a
                                # reference, not the secret). redact scrubs
                                # every field (subject/object/properties/steps/
                                # tags/…), not just the top-level strings.
CLARA_MAX_CONTENT_BYTES=16384   # per-memory content cap
CLARA_BACKUP_KEEP=7             # rotated snapshots kept in backups/
CLARA_QUERY_EXPANSION=0         # disable dev-vocabulary synonym expansion
```

### Retrieval / decay tuning

Invalid or out-of-range values are logged at WARNING and fall back to the
default (they are never applied silently):

```bash
CLARA_RETRIEVAL_TOP_K=8          # default results per query (1–1000)
CLARA_SIMILARITY_THRESHOLD=0.82  # duplicate-detection cosine cutoff (0–1)
CLARA_ARCHIVAL_THRESHOLD=0.15    # auto-archive below this confidence (0–1)
CLARA_EVENT_STALE_DAYS=90        # prune unlinked events older than this
CLARA_SKILL_UNUSED_DAYS=60       # deprecate skills unused this long
```

### REST API auth (optional `[api]` tier)

The API is off by default and, when run, binds `127.0.0.1`. It has no
implicit identity — enable real auth before exposing it:

```bash
CLARA_AUTH_REQUIRED=true                 # require a bearer token
CLARA_API_TOKENS=alice:s3cret-…,bob:…    # user:token pairs (token ≥16 chars)
CLARA_API_HOST=127.0.0.1                 # bind address (default loopback)
CLARA_CORS_ORIGINS=https://app.example   # explicit origins; "*" disables creds
```

With `CLARA_AUTH_REQUIRED=true` the app refuses to start unless tokens are set,
and every route (including `/admin/*`) is scoped to the token's user.

### Repo policy (`clara.yml`)

Teams tune the doc curator and knowledge graph with a committed `clara.yml` at
the repository root: document tiers, type patterns, staleness windows, archive
behavior, and terminology aliases. See [`clara.yml.example`](clara.yml.example)
for the full shape. `archive_dir` must be a repo-relative path (absolute or
`..` values are rejected at load time).

## Live memory counter (status bar)

See at a glance that CLARA is actually storing things. In Claude Code, run:

```
/clara:statusline
```

Then start a new session, and the bar shows:

```
CLARA - 1,248 memories - global
```

The count updates as soon as a memory is saved or forgotten, and covers every
memory type in the store. Turn it off again with `/clara:statusline off`.

Nothing else to configure: the slash command writes the `statusLine` block into
`~/.claude/settings.json` for you, keeping every other setting intact and
backing the file up first. If you already use a different status line, it asks
before replacing it.

From a terminal (only if you have the `clara` executable on your PATH — the
plugin's copy deliberately is not) the same thing is:

```bash
clara statusline --install --refresh-interval 10   # re-run the counter every 10s
clara statusline --uninstall                       # remove it again
```

Why a command rather than automatic setup: Claude Code plugins may ship a
`subagentStatusLine`, but **not** a main `statusLine` — that has to live in your
own settings file, so CLARA writes it for you instead of asking you to hand-edit
JSON.

Under the hood the counter never opens the database on the status-bar cadence:
writes refresh a small sidecar file next to the store, and the status line just
reads it.

## Removing CLARA

Disable it from Claude Code via `/plugin`, then remove the state it wrote:

```bash
clara uninstall            # removes the private venv, shim, and install log
rm -rf ~/.clara            # ALSO removes the memory store — omit to keep memories
```

`clara uninstall` never deletes your memory database; drop `~/.clara`
separately only if you want the stored memories gone too.

## Troubleshooting

- **"CLARA is installing in the background" every session / memory never
  appears.** The venv build is failing. Run `clara doctor` and read the
  `install log` tail it prints, or open `~/.clara/plugin/install.log`
  directly. Common causes: no Python 3.10+ on `PATH`, or no network for the
  first `pip` install.
- **`clara` / `clara-mcp` not found.** The plugin's venv is private (under
  `${CLAUDE_PLUGIN_DATA}`) and not on your shell `PATH` — that is expected;
  the plugin invokes it directly. For a shell-usable CLI, install Option B
  (`pip install -e ".[cli]"`).
- **Native Windows.** Hooks dispatch through polyglot `.cmd` files (PowerShell
  body under `scripts/win/`); the MCP server is spawned via a real-executable
  shim (`${CLAUDE_PLUGIN_DATA}/shim/clara-mcp`, resolves to `clara-mcp.exe`).
  If a hook seems inert, confirm either PowerShell or Git Bash is available and
  re-run `clara doctor`.
- **`clara doctor` exits 2 (unusable).** The store is corrupt or newer than
  your code. Restore the newest snapshot: `clara restore
  ~/.clara/backups/<newest>.db --force` (a pre-restore backup is taken first).
- **Schema "newer than supported".** An older CLARA opened a store a newer one
  wrote — it opens read-only and never corrupts it. Upgrade the plugin/package
  to write again.
- **`clara remember` says "Nothing durable recognized."** The rule-based
  extractor only matches canonical first-person forms. Rephrase (`I use X`), or
  let a model call `memory_save` with explicit fields.

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
faked. SQLite and FTS5 run embedded and real; LanceDB is provisioned with a
real per-test path, but vector ranking itself is exercised through fakes
(the ANN ordering is not asserted end-to-end).

Lint and type checks (enforced in CI, not optional):

```bash
ruff check clara tests
mypy clara
```

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
