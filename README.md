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
injection, 22 MCP tools, a knowledge graph, and a document-lifecycle curator —
and as a **standalone CLI + MCP server** for any agent (Codex, Gemini, Cursor,
or your own). Both use the same single SQLite store.

---

# What problem this solves

A coding agent starts every session with no memory of the last one. You
re-explain the same decisions — which package manager, which database, why that
library was dropped — and the agent re-reads the same files to rediscover facts
you already told it.

CLARA writes those facts to a SQLite file and injects the relevant ones back at
the start of the next session.

## With and without CLARA

This compares mechanics, not productivity. Everything below is a factual
difference in what happens; none of it is a claim about how much time you save,
because that has not been measured.

| | Without CLARA | With CLARA |
|---|---|---|
| **Start of a session** | Agent knows only what is in `CLAUDE.md` and what it reads | A `=== MEMORY CONTEXT ===` block of ranked memories is injected automatically, plus a `[KNOWLEDGE MAP]` of which docs are authoritative |
| **After `/clear` or `/compact`** | Context is gone; you re-explain | The block is re-injected — the SessionStart hook matches `clear` and `compact` |
| **A fact you stated last week** | Gone unless you wrote it into `CLAUDE.md` yourself | Stored as a typed memory, retrievable by search, decays if never used again |
| **Recording a decision** | You hand-edit `CLAUDE.md` and it grows unbounded | `memory_save` / `/clara:remember`; the file the agent reads stays a ≤60-line fenced section |
| **Contradictions** | Both statements sit in the file; the agent sees both | A negation supersedes the old belief, which is archived rather than deleted |
| **Stale documents** | Agent reads an outdated design doc as current | Reading a quarantined doc is annotated: "archived — treat as historical record" |
| **Relationships between things** | Implicit in prose | An explicit graph: `api runs_on fly.io`, queryable by entity, neighbours, or path |
| **Secrets in notes** | Whatever you paste is stored | The write path refuses credential-shaped content (default `reject`) |
| **Across two repos** | One `CLAUDE.md` per repo, no sharing | A global store by default, or a per-project store with `clara init --project` |
| **Losing the file** | Whatever you had is gone | Rotated backups (default 7) plus `clara restore`, and a JSONL export |

### What it does not do

Stated plainly so the table above is not read as more than it is:

- **It does not make the model smarter.** CLARA supplies context; the model
  still decides what to do with it.
- **It does not read your mind.** In the zero-key tier, memories are saved when
  the agent calls `memory_save` or you run `/clara:remember`. There is no
  background model inferring facts from your conversation.
- **It does not guarantee the agent uses a memory.** Injection puts it in
  context; the model chooses.
- **It has no measured effect on token usage.** The injected block is capped
  (≤60 lines in the native file, budgeted in the session-start block), but
  whether that nets out cheaper than re-reading files depends on your session
  and has not been benchmarked.
- **The zero-key tier does no semantic search.** Retrieval is SQLite FTS5
  keyword matching plus recency/confidence ranking. Embeddings and vector
  search exist only in the optional `[full]` tier.

---

# How it works

Four moving parts, all on one SQLite file.

### 1. The store

One file — `~/.clara/clara.db` by default, or `.clara/clara.db` inside a repo.
Tables for memories, the knowledge graph, and the document ledger, plus an FTS5
index. Schema changes are forward-only numbered migrations; a store written by
a **newer** CLARA than the one you are running is opened **read-only** rather
than written to, so a downgrade degrades visibly instead of corrupting data.

### 2. Session start (the hook)

The plugin registers a `SessionStart` hook matching
`startup|resume|clear|compact|fork`. It runs a deliberately stdlib-only Python
fastpath — no SQLAlchemy, no LLM imports — which:

1. resolves which store this directory maps to,
2. ranks active memories and renders the `=== MEMORY CONTEXT ===` block,
3. adds `[KNOWLEDGE MAP]` from the document ledger,
4. stamps `.git/clara-marker` so later hooks can find the repo cheaply.

It always exits 0. If anything fails, the session starts without memory rather
than not starting.

### 3. During the session (MCP tools)

The plugin ships an MCP server exposing 22 tools. The agent calls
`memory_search` when it needs prior context and `memory_save` when it learns
something durable. Three more hooks run alongside: `UserPromptSubmit` recalls
stored facts that match the words of each prompt (so a topic that first comes
up mid-session still gets its memory — each fact shown at most once per
session, silent when nothing matches, ~290 ms measured per prompt),
`PostToolUse(Read)` annotates reads of quarantined documents, and `Stop`
offers a one-line nudge when a plan document looks finished.

**You are the only intelligence in this tier.** There is no backend model doing
extraction or embeddings — the agent decides what is worth storing, and CLARA
stores it, indexes it, and gives it back.

### 4. Housekeeping

The first time the **MCP server** opens the store each day, one pass runs:
a rotated backup, confidence decay, pruning, graph maintenance, and the
native-memory export. No cron and no daemon. The `clara` CLI deliberately does
not trigger it, so a one-off command never pays for a decay pass and a VACUUM —
run `clara maintain` yourself if you drive the store from the CLI alone.

Confidence decays exponentially with age; a belief that falls below the
archival threshold is **archived, never deleted**, so it leaves search results
but still exports. Skills are exempt from archival.

---

# Measured performance

Measured on one Windows machine (Windows Server 2025, Python 3.12) with the
bundled interpreter. Your numbers will differ; these are recorded so the claims
above are checkable rather than adjectives.

| Operation | Measured |
|---|---|
| First plugin install (builds the private venv) | 113 s, once per plugin version |
| SessionStart hook, end to end | ~1.5 s |
| `PostToolUse(Read)` hook | ~190 ms |
| `Stop` hook | ~155 ms |
| MCP tool call, warm | 60–190 ms |
| MCP tool call, first of a session (opens the store) | ~2.2 s |
| `clara --help` | ~350 ms |
| Status-bar counter render | ~400 ms |
| Store-opening CLI command (`list`, `stats`, `context`) | ~1.5 s |
| Graph traversal, 100k-edge store, depth 1 | ~3 ms |

Concurrency: six processes writing thirty memories at once all committed, with
`PRAGMA integrity_check` clean afterwards.

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

### Step by step — your first five minutes

What you will actually see, in order. Nothing here needs a terminal.

**1. Install the plugin.** In Claude Code:

```
/plugin marketplace add itsamiitt/clara
/plugin install clara@clara-marketplace
```

**2. Start a new session.** It prints:

```
CLARA is installing in the background — memory will be available next session.
```

This is expected, not an error. CLARA is building its own private Python
environment so it never touches your system packages. It takes about two
minutes and happens once per plugin version. Keep working — nothing is
blocked.

**3. Start another session.** The install is done. Now the session begins with
a block like this, which the model can see and you normally cannot:

```
=== MEMORY CONTEXT ===

[BELIEFS]
- user uses pnpm (confidence: 0.85)
- api runs_on fly.io (confidence: 0.90)

[KNOWLEDGE MAP]
active work (T2): 3
rule: treat quarantined/archived documents as historical record.
```

On a brand-new store this is empty. That is correct — there is nothing to
remember yet.

**4. Save your first memory.** Either tell Claude something durable and let it
call `memory_save`, or be explicit:

```
/clara:remember we use pnpm, never npm
```

**5. Confirm it stuck.**

```
/clara:memories
```

…or check the health of the whole install:

```
/clara:doctor
```

That command works without anything on your `PATH`, which matters because a
plugin install deliberately does not put `clara` there (see
[Verify it works](#verify-it-works)).

**6. Next session, it is already there.** Nothing to do. The block from step 3
now contains what you saved, and it comes back after `/clear` and `/compact`
too.

### If something looks wrong

| What you see | What it means | What to do |
|---|---|---|
| "installing in the background" on every session | The build is failing partway | `/clara:doctor` — it prints the tail of the install log |
| `memory` MCP server unavailable | First session, or the build is still running | Start a new session; it resolves itself |
| `1 error during load` right after installing | Expected on a first install. `/reload-plugins` tries to start the memory server while the two-minute build is still running, so the binary it points at does not exist yet | Nothing. Start a new session — the binary is there by then. `/clara:doctor` confirms it (`[ok] shim clara-mcp: present`) |
| Memory tools missing after installing mid-session | The MCP server attaches at session start, so a plugin installed during a session has no server until the next one | Start a new session |
| `clara: command not found` | Expected — a plugin install does not put the CLI on `PATH` | Use `/clara:doctor`, or the full path `~/.clara/plugin/shim/clara` |
| Memory block is empty | Nothing saved yet, or you are in a different project | `/clara:memories` to check; stores are per-project when a `.clara/` exists |
| Nothing at all happens | The plugin's hooks are not firing | `/plugin` to confirm it is enabled, then start a new session |

**What happens on first use.** The plugin ships a bootstrap that builds a
private virtualenv from the plugin's own checkout (no PyPI needed) into
`${CLAUDE_PLUGIN_DATA}/` (survives updates). The install runs in the
background, so your **first session** prints:

```
CLARA is installing in the background — memory will be available next session.
```

The build takes about two minutes, once per plugin version (measured on a
Windows machine using the bundled interpreter, three cold installs: 113 s,
114 s and 130 s; a warm pip cache and a faster disk cut it). **From the second
session on**, your memory context is
injected automatically at session start and the `memory_*` tools are live. The block ends with a `[MEMORY PROTOCOL]` footer that tells the model, every session, to save facts the moment they appear, batch with `memory_save_many`, and search before asking — real-time memory is an instruction the session starts with, not a habit it must remember.
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

## The 22 MCP tools

Once wired, your agent's model calls these directly — **CLARA is storage +
ranked retrieval; the host model is the intelligence** (that is why no API key
is needed).

- **Memory (7):** `memory_save`, `memory_save_many`, `memory_search`,
  `memory_recent`, `memory_update`, `memory_forget`, `memory_stats`.
  `memory_save_many` writes a whole batch in one transaction, all or
  nothing — measured, 100 facts in 2.1 s against 7.4 s as sequential
  saves — and one batch cannot race itself the way parallel single
  saves can.
- **Code index (3):** `code_deps`, `code_impact`, `code_health` — the
  import graph of *this repo's source*, built by `clara index` and kept
  current by the daily maintenance pass. `code_impact` answers "what
  breaks if I change this module" before you edit it. They report
  `indexed: false` on a repo that has not been indexed, rather than
  reporting no dependencies.

  Python (`.py`) and JavaScript/TypeScript (`.ts .tsx .js .jsx .mjs .cjs`)
  are indexed. Python uses the standard library's own `ast`; JS/TS uses a
  scanner checked against the TypeScript compiler over 2,100 files of two
  production repos — 8,313 module specifiers, zero missed and zero invented.
  `tsconfig.json` path aliases (`@/lib/db`) are resolved, so imports written
  that way are internal edges rather than phantom packages. Other languages
  are not indexed; a Go or Ruby file is skipped, not half-parsed.

  Measured on a 2,568-file TypeScript repo: first index 16 s, re-index with
  nothing changed 1.15 s, `code_impact` at depth 3 about 150 ms.

  `code_health`'s "nothing imports this" list uses what the project declares
  about itself — `package.json` entry points and script commands, `<script
  src>` in any HTML page, a Chrome extension manifest — plus conventions the
  tools define (test files, `*.config.*`, framework routes). On that same repo
  the raw "no importer" list was 848 files, 33% of the repo, and was mostly
  configs, tests and app entries; with that evidence applied it is 86 files
  (3%), spot-checked as genuinely unreferenced. It is still not a delete list:
  a file loaded by name at runtime cannot be seen by a static import graph, and
  the tool says so rather than pretending otherwise.
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
  native-memory export run opportunistically the first time the **MCP server**
  opens the store each day. No cron, no daemon. That covers every Claude Code
  session (the plugin starts the server) and any agent wired to `clara-mcp`.
  The `clara` CLI deliberately does not trigger it, so a one-off command never
  pays for a decay pass and a VACUUM. If you drive the store from the CLI alone,
  run it yourself with `clara maintain` (`--force` to override the once-a-day
  gate) — same pass, same single-winner lock.

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
clara maintain [--force]                run housekeeping now: backup, decay,
                                        pruning, graph, native export
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

**How one shared store keeps projects straight.** Every memory is stamped at
save time with the repository it was saved from, and every read surface uses
one locality rule (local = this repo's stamp, no stamp, or a fact about the
user — preferences follow the person):

| surface | behaviour with another project's facts |
|---|---|
| session-start block | ranked after local facts, labeled `[from another project]`, at most 3 shown |
| per-prompt recall | need a stronger match (two overlapping words *and* a store-rare naming word) and rank last |
| `memory_search` / `memory_recent` / `clara context` | relevance order kept — you asked — but labeled in the block and flagged `foreign` in structured hits |
| `MEMORY.md` export (`clara sync`) | project file, so: local first, labeled, capped; the `clara-memory.md` topic file labels but always contains the full set |

This was built against a real store where one project's nine audit findings
filled eight of ten context slots in every other project's sessions. If you
want hard isolation instead of labels, use a project store (`clara init
--project`).

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
