# CLARA Advancement Plan

**Date:** 2026-07-13
**Scope:** Full audit of CLARA as it exists at commit `f5bf0f0`, comparison against the memory-system field, ranked gap analysis, and a sequenced plan to make CLARA a zero-API-key, zero-heavy-model, drop-in memory layer for any coding-agent CLI.
**Method:** Every file in `clara/`, `tests/`, and all 11 root-level docs was read end-to-end by a fan-out of audit agents; all load-bearing claims were then independently re-verified against source (file:line cited throughout). Comparative research used primary sources (GitHub source, official docs, papers) — cited inline.

---

## Executive Summary

CLARA today is **two products sharing one schema**, and the README only documents the wrong one for the CLI-agent goal:

1. **The full cognitive pipeline** (`ClaraMemory`): LLM fact extraction → heuristic classification → embedding → conflict resolution → SQLite + LanceDB. Powerful on paper, but it hard-requires an OpenAI key (default embedding backend fails fast without `OPENAI_API_KEY`), silently stores *nothing* when the extraction LLM fails (`extractor.py:430-432` catches every exception and returns `[]`), and carries a set of real correctness bugs (confidence dilution on reinforcement, immortal skills, permanent vector loss on failed background sync).

2. **The zero-backend path** (`LocalMemory` + `clara-mcp` MCP server, `clara/integrations/`): already built, already tested, already the package's only console script — **no LLM, no embeddings, no LanceDB, no API key**. It is exactly the "drop-in memory layer for CLI agents" this plan is asked to design, and it is *completely absent from the README*.

**The core strategic recommendation of this plan:** stop treating the LLM pipeline as the product and the MCP server as an appendix. Invert it. The MCP/LocalMemory path becomes the primary, documented, one-command install; the LLM pipeline becomes an optional enrichment tier. This matches where the field went in 2025-26 (Mem0 moved extraction cost down to a single pass; Letta V2 moved memory to plain git-backed files; Claude Code's own auto-memory is plain markdown, zero-model) and directly answers the most common real-world complaints (silent failure, API-key setup burden, vendor lock-in).

What must be built/fixed, in one paragraph: upgrade lexical retrieval from naive token-overlap to **SQLite FTS5/BM25** with an optional **model2vec static-embedding tier** (~8-32 MB, numpy-only, no torch); add **rule-based extraction** so `remember()` works keyless; make failure **loud** (structured status instead of silent `[]`); fix the four high-severity correctness bugs; ship `clara init` + per-agent config snippets (Claude Code hooks + MCP, Codex `config.toml`, Gemini/Cursor MCP, generic CLI); re-package so the keyless core installs without `openai`/`lancedb`; and delete ~7 stale root docs plus 3 junk files.

---

## Phase 1 — How CLARA works today

### 1.1 System diagram

```mermaid
flowchart TB
    subgraph Public API — clara/agent.py
        R[remember_text_] --> IL[InteractionLayer.receive<br/>whitespace normalize]
        RC[recall_query_] --> RE
        CF[context_for_query_] --> RC
        IN[interact_message_] --> IL
    end

    IL --> EX[FactExtractor.extract<br/>1 LLM call - openai/anthropic/ollama<br/>extractor.py — ALL failures swallowed to empty list]
    EX --> UE[MemoryUpdateEngine.process<br/>update/engine.py]
    UE --> CLS[classify_memory_type<br/>keyword frozensets — NO LLM<br/>event > skill > world_model > belief]
    UE --> SIM[similarity search top-10<br/>threshold 0.82]
    SIM --> RE
    UE --> DEC{conflict?}
    DEC -->|same subj+rel, opposite polarity or diff object| RES[resolve: supersede if fact.confidence > 0.6<br/>retain-both if domains differ or conf <= 0.6]
    DEC -->|similar + same belief| REINF[reinforce: BeliefMemory.update<br/>BUG: confidence dilutes toward 0]
    DEC -->|novel| NEW[create new row]

    RES --> SQL[(SQLite memories table<br/>WAL, busy_timeout 30s<br/>partial unique index on world_model<br/>json_extract — SQLite-only)]
    REINF --> SQL
    NEW --> SQL
    SQL -->|after_commit hook<br/>daemon thread, fire-and-forget| LDB[(LanceDB clara_vectors<br/>1536-dim, brute-force cosine<br/>no ANN index built)]

    RE[RetrievalEngine.search] --> EMB[EmbeddingEngine<br/>openai: needs OPENAI_API_KEY - fails fast<br/>local: sentence-transformers + torch<br/>ollama: needs daemon]
    RE --> LDB
    RE --> SQL
    RE --> SCORE[score = 0.65 sim + 0.20 conf<br/>+ 0.10 recency + 0.05 usage]

    IN --> RSN[ReasoningEngine.respond<br/>retrieve -> LLM generate -> extract OWN response -> store<br/>HARD LLM dependency, uncaught errors]

    SCHED[DecayScheduler - APScheduler<br/>daily decay 02:00 / weekly prune Sun 02:30<br/>daily reflection 03:00 - LLM optional] --> SQL

    subgraph Zero-backend path — clara/integrations/ (UNDOCUMENTED)
        MCP[clara-mcp stdio server<br/>6 tools: save/search/recent/update/forget/stats<br/>+ recall CLI for SessionStart hooks] --> LM[LocalMemory<br/>no LLM, no embeddings, no LanceDB]
        LM --> LEX[LexicalRetriever<br/>token-overlap similarity, ILIKE scan<br/>LIMIT 1000 — FTS5 named as upgrade path]
        LM --> SQL2[(SQLite ~/.clara/clara.db<br/>separate default from API's ./clara.db)]
    end
```

### 1.2 The four call paths, in plain English

**`remember(text, *, user_id=None, wait=True)`** (`agent.py:393-468`)
Your text is whitespace-normalized, then sent in **one LLM call** to the configured provider with a fixed JSON-only system prompt (verbatim in `extractor.py:111-152`: extract subject/relation/object/domain/source_type/confidence/is_negation triples; ignore hedged statements; emit negation pairs for "switched from X to Y"). The response is parsed defensively — malformed JSON, wrong wrapper keys, low-confidence (<0.4) or empty-field facts are dropped with only WARNING logs, **never retried**. Each surviving fact goes through the update engine: memory type is decided by **keyword lookup on the relation** (not the LLM — `update/engine.py:85-128`; "deployed"→event, "knows"→skill, "runs_on"→world_model, everything else→belief); the fact text is embedded and compared against top-10 similar memories (threshold 0.82) plus exact-SQL belief matches; then it is **created**, **reinforced** (same belief seen again), **superseded** (conflicting fact with confidence > 0.6 — the *old* row's confidence is never consulted), or **retained-both** (differing domains, or ambiguous ≤ 0.6). Supersede = status flip on the old row + new active row + `metadata.superseded_by`/`supersedes` links; nothing is ever deleted. With `wait=True` all facts share one transaction (one failure rolls back all); with `wait=False` facts go to an **in-memory queue whose worker swallows every exception** — the caller is told `"queued"` and can never learn of failure.
**Critical failure mode:** if the extraction LLM is unreachable (no key, network down, Ollama stopped), `extract()` returns `[]` and `remember()` reports success-with-no-facts, forever, with no error surfaced (`extractor.py:430-432`).

**`recall(query, top_k=8, *, user_id=None)`** (`agent.py:470-502`)
The query is embedded (a **synchronous** provider call inside the async path — blocks the event loop), LanceDB is searched brute-force cosine (no ANN index anywhere) with prefilter `status='active'` (+ user_id), top_k×4 candidates are hydrated from SQLite (the source of truth — rows re-checked for status/tenant), and re-ranked with `score = 0.65·similarity + 0.20·confidence + 0.10·recency + 0.05·usage` (`retrieval/engine.py:41-49,135-147`; recency is e^(−0.01·days since *updated_at*, not last-access, despite the docstring); usage is log-normalized access_count within the candidate set). Access counts are then bumped best-effort in a second session (SQLite lock errors swallowed at DEBUG).
**Critical failure modes:** any LanceDB failure returns `[]` silently with no lexical fallback (`engine.py:357-359`); a `remember()` immediately followed by `recall()` can miss the new memory because vector sync rides a background daemon thread (read-after-write gap, by design per test comments); if that thread's flush fails, pending vectors are **dropped permanently** — rows exist in SQLite but are invisible to vector search forever; no backfill tool exists (`engine.py:250-280`).

**`context_for(query, ...)`** (`agent.py:504-524`)
`recall()` + `format_context()`: renders a prompt-ready block
`=== MEMORY CONTEXT === / [BELIEFS] / [WORLD MODEL] / [RECENT EVENTS] / [RELEVANT SKILLS] / === END MEMORY CONTEXT ===`, one line per memory, `- (none)` for empty sections. Every interpolated value passes `sanitize_memory_text` (`core/text.py`) — control chars stripped, `===` fences and `[SECTION]` markers defanged, 500-char truncation — a genuinely good prompt-injection defense, contractually tested. **Maintenance hazard:** the formatter exists twice (`agent.py:119-210` and `reasoning/context.py:9-93`), both live; fixes must land in both or `context_for()` and `interact()` diverge.

**`interact(message, ...)`** (`agent.py:526-580`)
One DB transaction wraps the whole loop: retrieve context → **LLM call #1** generates a reply (default system prompt + caller prompt + memory context) → **LLM call #2 extracts facts from the assistant's own reply** (not the user's message!) → facts stored in the same transaction. This is the only entry point with a *hard* LLM dependency: no key/package → uncaught `EnvironmentError` → HTTP 500 via the API. Two design defects: (a) the transaction is held across two network round-trips (30s timeout × 2 SDK retries each — minutes of SQLite write-lock in the worst case); (b) response-derived facts default to `source_type="user_direct"` (trust weight 1.0), so **assistant hallucinations enter memory with maximum trust and can supersede genuine user statements** at confidence > 0.6.

### 1.3 Scheduler & reflection

`DecayScheduler` (APScheduler 3.x, UTC cron): daily decay 02:00 (`confidence ×= e^(−decay_rate·days)`, archive below 0.15 — skills exempt), weekly prune Sunday 02:30 (stale events > 90d archived; skills unused > 60d deprecated), daily reflection 03:00 (pattern detection over last 7 days — pure Python counting, ≥3 occurrences — then one LLM sentence per pattern; the sentence lands only in `metadata.raw_text`, the stored belief triple and its embedding never see it, so the LLM call is semantically cosmetic; missing key silently degrades to template text). Job exceptions are caught by APScheduler and logged to a logger nobody configures — **no listener, no retry, no alert**; with default `misfire_grace_time=1s` a busy event loop silently skips a night's run.
**Verified high bug:** the daily decay job touches every skill row, which fires the ORM `onupdate` hook on `updated_at` (`models.py:171`), so the weekly pruner's `updated_at < now-60d` predicate can never match — **skills are immortal in production**; the unit tests miss it because they use fakes without `onupdate` semantics.

### 1.4 Config & env matrix (what actually breaks)

`ClaraConfig` reads 17 env vars; **11 of the 17 fields are decorative** — parsed then ignored. The five tuning knobs (`CLARA_RETRIEVAL_TOP_K`, `CLARA_SIMILARITY_THRESHOLD`, `CLARA_ARCHIVAL_THRESHOLD`, `CLARA_EVENT_STALE_DAYS`, `CLARA_SKILL_UNUSED_DAYS`) do nothing; engines hardcode module constants (`update/engine.py:42`, `scheduler/decay.py:49-55`). Provider model fields work only by env-var coincidence, read directly by the modules. Live and load-bearing:

| Var | Default | Unset consequence |
|---|---|---|
| `OPENAI_API_KEY` | — | default embedding backend → `EnvironmentError` at `ClaraMemory.create` (agent won't start); extraction → **silent permanent no-op** |
| `CLARA_EMBEDDING_BACKEND` | `openai` | keyless default is a trap; `local` needs sentence-transformers+torch; `ollama` needs a daemon |
| `CLARA_LLM_PROVIDER` | `openai` | learn silently no-ops without key; interact 500s |
| `CLARA_DB_URL` | `sqlite+aiosqlite:///clara.db` | file in CWD |
| `CLARA_LANCE_PATH` | `./clara_vectors` | dir in CWD; **written back into os.environ** by `create()` (process-global mutation) |
| `CLARA_DB_PATH` / `CLARA_HOME` | `~/.clara/clara.db` | MCP-server store — **note: different default DB than the API path** |
| `CLARA_START_SCHEDULER` | `true` | scheduler runs in-process |
| `CLARA_AUTH_REQUIRED` | `false` | API trusts any `X-User-ID` header (identification, not authentication) |

Other footguns verified: `ClaraMemory.create()` mutates `os.environ` with resolved Ollama values (`agent.py:324-326` — two agents in one process clobber each other); `close()` closes the process-shared Lance singleton (breaks a second agent); `main.py` hardcodes `uvicorn.run(host="0.0.0.0", port=8000, reload=True)`; explicit kwargs equal to their defaults silently lose to env vars.

### 1.5 The undocumented integration layer (the headline finding)

`clara/integrations/mcp_server.py` (251 lines) is a **complete, tested, packaged MCP stdio server** on the official `mcp>=1.2` FastMCP SDK — the package's only console script (`clara-mcp`, `pyproject.toml:79-80`). Six tools: `memory_save` (typed, with per-type required fields the docstring teaches the calling model), `memory_search`, `memory_recent`, `memory_update`, `memory_forget` (archive/deprecate, never delete), `memory_stats`. Plus a `clara-mcp recall` CLI subcommand purpose-built for a Claude Code SessionStart hook (prints the memory-context block, empty store → prints nothing, exit 0). It runs on `LocalMemory` (`local_memory.py`, 365 lines): the same SQLite schema and typed stores, LanceDB short-circuited via a session flag, retrieval via `LexicalRetriever`. **Design doctrine stated in its docstring: "Claude itself decides what to remember and recall — no LLM, embedding model, API key, or vector server is used by this layer."** README mentions none of this. The delegation of extraction to the host agent is the same pattern Letta uses for agentic memory (the LLM calls memory tools) — except here the host CLI's model does it for free.

Ceilings of the current zero-backend path: lexical similarity is plain token-overlap (no BM25/IDF/stemming), candidate fetch is `ILIKE` over serialized JSON capped at 1000 rows (`lexical.py:38` names FTS5 as the upgrade path), no user_id scoping in the MCP tools, and its default DB (`~/.clara/clara.db`) is not the API's default (`./clara.db`).

---

## Phase 2 — Comparative research

Primary-source findings (each row verified against source/docs in July 2026; details and URLs in the appendix source list):

| System | Write trigger | Read trigger | Conflict resolution | Storage | Needs API key / local LLM to function at all? |
|---|---|---|---|---|---|
| **CLARA (full)** | app calls `remember()`/`interact()` → 1 LLM extraction call | app calls `recall()`/`context_for()` → embed + vector search (0 LLM) | heuristic: keyword classify + 0.82 similarity + supersede/reinforce/retain-both rules (no LLM) | SQLite + embedded LanceDB | **Yes** — key or Ollama for writes; key/torch/Ollama for reads |
| **CLARA (MCP path)** | host agent calls `memory_save` (host's own model decides) | host agent calls `memory_search`; or `recall` CLI at session start | none (append; explicit update/forget tools) | SQLite only | **No** — fully keyless |
| **Mem0 OSS v2.x ("v3 algo", Apr 2026)** | app calls `add()` → **1** LLM call (was 2; ADD-only now, no UPDATE/DELETE/NOOP) + spaCy entity linking (local, no LLM) | `search()` → **0 LLM**: vector + BM25 + entity-boost fusion | **append-only**; contradictions linked via `linked_memory_ids`, resolved at retrieval ranking; decay/temporal reasoning platform-only | any of ~24 vector stores (default Qdrant) + SQLite history log | Default yes (OpenAI); fully local possible via Ollama/FastEmbed; `infer=False` = raw storage, no LLM |
| **Letta / MemGPT (V1)** | the agent LLM calls memory tools (`core_memory_replace`, `archival_memory_insert`); sleeptime agent consolidates every N steps | core blocks always in-prompt; agent calls `conversation_search`/`archival_memory_search` (hybrid + RRF) | none automatic — LLM is the merge engine; block char-limit forces condensing; V2 = git commits/worktrees (MemFS markdown) | Postgres+pgvector (prod) / SQLite (dev); V2: git repo of markdown | **Yes** for anything agentic; server CRUD works keyless but inert |
| **Zep / Graphiti** | `add_episode()` → 5-stage LLM pipeline (entities → dedup → edges → resolution → temporal invalidation) | `search()` — **0 LLM**: cosine + BM25 + graph BFS, RRF/MMR rerank | **bi-temporal edge invalidation**: contradicted edge gets `invalid_at`, never deleted; LLM judges contradictions | Neo4j / FalkorDB (+ cloud "Context Lake") | **Yes** — no LLM-free ingestion path exists (open issue #1299) |
| **LangMem / LangGraph** | hot path: agent calls `manage_memory` tool; background: dev-wired `create_memory_store_manager` (+ debounced `ReflectionExecutor`) | agent calls `search_memory` or dev calls `store.search`; nothing auto-injected | trustcall patch-based upsert; `RemoveDoc` deletions opt-in; LLM judgment + schema validation | LangGraph `BaseStore` (InMemory / Postgres) | **Yes** — some chat model required for extraction/consolidation |
| **ChatGPT memory (Dreaming V3, Jun 2026)** | model-initiated `bio` tool ("Memory updated" chip) + background synthesis over all chats | everything pre-injected into system prompt (no per-query retrieval) | background process auto-rewrites memories as circumstances change ("went to Singapore in July 2026"); criticized for weak audit trail | OpenAI-internal | n/a (bundled) |
| **Claude Code native** | user edits CLAUDE.md; auto-memory MEMORY.md the model maintains; hooks fire deterministically | CLAUDE.md concatenated at session start; MEMORY.md first 200 lines/25KB every session; hooks inject via stdout/additionalContext; MCP tools on model decision | none — concatenation + precedence; contradictions "picked arbitrarily" per docs | plain files (`~/.claude/projects/<project>/memory/`) | **No** — pure file conventions |
| **Codex CLI native** | user edits AGENTS.md; auto "Memories" summaries | AGENTS.md concatenated root→cwd at start; MCP via `[mcp_servers.*]` in `~/.codex/config.toml` | none | plain files; machine-local | **No** |
| **Local-first prior art** (studiomeyer local-memory-mcp, memento, memoirs, ICM, sqlite-memory) | host agent calls MCP tools | MCP search tools; hybrid FTS5/BM25 + optional sqlite-vec, RRF | heuristic curators / recency signals; no LLM | SQLite + FTS5 (+ sqlite-vec) | **No** — that's the category's selling point |

**Field-wide convergences that matter for CLARA:** (1) Everyone made **reads LLM-free**; the fight is over write-path cost — Mem0 cut writes from 2 LLM calls to 1 and moved entity work to **spaCy (local, no LLM)**; the local-first category cut writes to 0 by letting the *host* model do the structuring — exactly CLARA's MCP design. (2) **Hybrid lexical+vector retrieval with RRF is table stakes** (Mem0, Letta archival, Zep, every local-first tool); CLARA has vector-only (full path) or overlap-only (local path), never hybrid. (3) Append + link/invalidate is replacing destructive updates (Mem0 append-only, Zep invalidation, Letta git history) — CLARA's supersede-with-links is actually ahead of Mem0 here. (4) The integration surface for CLI agents is **MCP + config-file conventions**, full stop.

---

## Phase 3 — Ranked gap list (cross-referenced with real user complaints)

Complaint evidence (primary citations): Mem0 issues [#5245](https://github.com/mem0ai/mem0/issues/5245) (silent memory loss on batch embed failure), [#3009](https://github.com/mem0ai/mem0/issues/3009) ("3 out of 5 memory creations lost — fact extraction inconsistently returns empty"), [#2443](https://github.com/mem0ai/mem0/issues/2443) (info not stored), [#4985](https://github.com/mem0ai/mem0/issues/4985) (switching embedding provider silently drops writes), [#4037](https://github.com/mem0ai/mem0/issues/4037) (auto-recall injection silently broken); Letta [#2388](https://github.com/letta-ai/letta/issues/2388)/[#2772](https://github.com/letta-ai/letta/issues/2772) (local/keyless config friction); Zep/Graphiti [#868](https://github.com/getzep/graphiti/issues/868) (local models break structured output), [#1299](https://github.com/getzep/graphiti/issues/1299) (no LLM-free ingestion); Simon Willison's ChatGPT-memory critique (invisible dossier, no edit/delete, contamination); ChatGPT "Memory Full → silently stops saving"; cross-tool amnesia patterns (session resets, compaction loss, silent deletion) per r/ClaudeAI / r/ChatGPT / r/codex roundups.

Complaint patterns → CLARA status:

| # | User complaint pattern (field-wide) | CLARA today | Evidence |
|---|---|---|---|
| C1 | **Memory silently fails, no error surfaced** | ❌ Worst-in-class: extraction swallows all errors → `[]` (`extractor.py:430-432`); `wait=False` queue drops facts silently; Lance sync failure = permanent invisible vectors; API returns 200 `{"stored": 0}` | matches Mem0 #5245/#3009/#4985 exactly |
| C2 | **Setup needs API key / heavy model** | ❌ full path (default backend fails without key; `local` = torch; `openai` SDK + `lancedb` are unconditional deps) / ✅ MCP path — but undocumented | Letta #2388/#2772, Zep #868 |
| C3 | **Works only in one vendor's app** | ✅ architecture (MCP = every major CLI) / ❌ practice (zero docs, no install flow) | field-wide lock-in complaints |
| C4 | **Memory not persisting across sessions** | ✅ SQLite persists; ❌ two different default DB paths (API `./clara.db` vs MCP `~/.clara/clara.db`) guarantee "where did my memories go" | duet.so amnesia roundup |
| C5 | **Irrelevant or stale facts recalled** | ⚠️ decay/supersede exist (good) but: confidence-dilution bug *destroys* repeatedly-confirmed beliefs; immortal-skill bug; reflection spam (`user appears_frequently_in recent_activity` daily); recall has no hybrid ranking | Willison contamination critique |
| C6 | **No visibility into what's stored** | ⚠️ admin API + `memory_stats` exist; no CLI list/inspect for the MCP store; no "what was saved this session" signal (ChatGPT's "Memory updated" chip is the UX bar) | Willison; ChatGPT criticisms |
| C7 | **No easy correct/delete** | ⚠️ `memory_update`/`memory_forget` tools exist; nothing user-facing (CLI/flags); no delete-by-query | ChatGPT "Memory Full" complaints |
| C8 | **Memory bloats context/cost** | ⚠️ `top_k` + 500-char truncation cap the block; but recall quality ceiling (C5) wastes the budget | field-wide |

Ranked gaps (impact = complaints addressed × severity; cost = engineering size):

| Rank | Gap | Complaints hit | Cost | Type |
|---|---|---|---|---|
| **G1** | Silent-failure design: extraction/queue/Lance failures invisible | C1 | S | reliability |
| **G2** | Keyless mode is unusable as shipped: undocumented, weak lexical ranking, no extraction, heavy mandatory deps | C2, C3, C8 | M | architecture |
| **G3** | No one-command install / per-CLI integration story | C3, C2 | M | UX |
| **G4** | Correctness bugs: confidence dilution; immortal skills; hallucinated-reply facts stored as `user_direct` and able to supersede user beliefs; events superseded (timeline rewritten); update-engine world_model rows bypass unique guard + invisible to store | C5 | M | bugs |
| **G5** | Retrieval quality: no hybrid lexical+vector, no FTS5, no RRF; silent `[]` on Lance failure with no fallback | C5, C8 | M | architecture |
| **G6** | Durability: fire-and-forget vector sync (permanent loss, no backfill), unbounded in-memory write queue, read-after-write gap | C1, C4 | M | reliability |
| **G7** | Split-brain defaults: two DB paths; 11/17 dead config fields; env-var/kwarg precedence surprises; process-global env mutation | C4 | S | hygiene |
| **G8** | No visibility/edit UX: no `clara list/forget/stats` CLI for humans; no per-session "stored N memories" summary | C6, C7 | S | UX |
| **G9** | interact()'s hard LLM dependency + transaction-across-LLM-calls + self-extraction trust bug | C2, C5 | S (deprecate) | scope |
| **G10** | Scheduler fragility: silent job failures, 1s misfire grace, all-tenants-one-transaction reflection, decay/cache staleness | C1, C5 | S | reliability |
| **G11** | Repo hygiene: 7 stale/contradictory root docs, junk scripts, unrelated client .docx | (trust) | XS | cleanup |
| **G12** | Postgres unsupported (json_extract index; `now()` server default; misleading "identical on PG" comment) | — | M (defer) | scope |
| **G13** | README claims unverifiable (PyPI badges, static test counts) and omissions (MCP, auth/CORS vars, extras) | (trust) | XS | docs |

Explicitly deferred (documented as out of scope, matching README's own "not implemented"): document storage/chunked ingestion, procedural skill graphs, Postgres (G12) until a real deployment needs it.

---

## Phase 4 — Designs for the ranked gaps

Stable public surface: `remember() / recall() / context_for() / interact()` signatures unchanged; additions only.

### G1 — Kill silent failure (do this first; it re-frames everything else)

**Change:** introduce a structured write result and a degradation flag, everywhere.
- `clara/extraction/extractor.py`: `extract()` returns `ExtractionResult(facts: list[ExtractedFact], status: Literal["ok","llm_unavailable","malformed_response","empty"], detail: str|None)` instead of a bare list. The `except Exception` at :430 narrows to provider/network errors and *records* them; programming errors propagate. `_parse_llm_response` moves inside the try, fixing the verified crash path (`AttributeError` on non-dict fact items, :310 vs :320 — `{"facts": ["a string"]}` from the LLM currently crashes `remember()`).
- `clara/agent.py`: `remember()` return gains `{"status": ...}`; on `llm_unavailable` it falls back to the rule-based extractor (G2) and reports `{"status": "degraded_heuristic"}`. Never again 200-with-zero-and-no-reason.
- `update/background.py`: bounded queue (`maxsize=256`), per-fact retry (×2), dead-letter table `failed_writes` in SQLite, `stop()` drains, `enqueue()` after stop raises. Fixes the verified shutdown race (enqueue-behind-sentinel drop).
- `retrieval/engine.py`: Lance search failure → log ERROR once per process + **fallback to LexicalRetriever** (G5) instead of `return []`.
**Tests:** extraction returns status not `[]` on mocked provider failure; remember reports degraded status; background writer failure lands in `failed_writes`; Lance-broken recall still returns lexical results.

### G2 — Zero-key core (full design in Phase 6)

### G3 — One-command install + per-CLI adapters (full design in Phase 5)

### G4 — Correctness bug fixes

1. **Confidence dilution** (`memory/belief.py:88-91`, verified): reinforcement computes `(prior·decay + w·s)/(prior_weight+1)` — repeated confirmations of a true belief drive confidence → 0 (1.0 → 1.0 → 0.667 → 0.417 → …). Fix: weighted mean `(prior·prior_weight·decay + w·s)/(prior_weight + 1)`. Test: N successive identical `user_direct` confirmations must be monotonically non-decreasing toward 1.0; add the missing regression the current test suite codifies *wrongly* (`test_belief.py:178-194` locks in the halving).
2. **Immortal skills** (`models.py:171` onupdate × `decay.py:281`, verified): track decay in `metadata.last_decay_at` only — decay job must not dirty `updated_at` (use `flag_modified` on metadata alone or a separate `last_activity_at` column); pruner keys on real activity (`last_used` in metadata, falling back to created_at). Test with real ORM (not fakes) asserting a skill decayed nightly for 61 simulated days gets deprecated.
3. **Self-extraction trust** (`reasoning/engine.py:128` + `extractor.py:315`): facts extracted from the assistant's own reply get `source_type="agent_inference"` (weight 0.5 — cannot supersede user beliefs at the 0.6 threshold). One-line change in `respond()` to pass a source override; add `source_override` param to `extract()`.
4. **Events superseded** (`update/engine.py:164-166,451-459`): exclude `MemoryType.event` from conflict resolution entirely — two similar events are two occurrences; timeline history must be append-only. Route to create-new instead.
5. **Update-engine world_model rows bypass the unique guard** (`engine.py:553-561`): when classification says world_model, delegate storage to `WorldModelStore.upsert` (already race-safe via savepoint + IntegrityError retry) instead of `_store_fact`'s bare row; same for events→`EventStore.create` and skills→`SkillStore.create` so all rows carry the shapes the stores and the partial unique index expect.
6. **Cross-tenant None leak** (`world_model.py:294-313` et al.): `user_id=None` should mean "the None tenant", not "all tenants", on every write path; read paths keep None=global only for explicit admin queries. Add `tenant_strict` sessions test.
Each fix ships with a regression test named for the bug; the brittle stress-census assertions (`test_agent_stress.py:273-278`) get updated in the same PRs — they will break by design.

### G5 — Retrieval: hybrid, with FTS5 floor

- Add SQLite **FTS5** virtual table `memories_fts(content_text, tokenize='porter unicode61')` maintained by triggers on `memories` (contentless-delete pattern), replacing `LexicalRetriever`'s ILIKE-over-JSON scan; score = BM25. This alone lifts the keyless tier from token-overlap to state-of-practice (the upgrade `lexical.py:37-38` already names).
- `RetrievalEngine.search` becomes **hybrid**: run vector (if embeddings available) and FTS5 in parallel, fuse with RRF (k=60), then apply the existing composite re-rank (similarity term = fused score). Lance failure degrades to FTS5-only with a logged warning (G1).
- Embed queries via `run_in_executor` (fixes event-loop blocking, `engine.py:474-477`).
- Fix recency to use real last-access when present (`metadata.last_accessed` is already written, never read — one-line).
**Tests:** keyless search finds stemmed matches ("deployed"→"deploy"); hybrid beats each single retriever on a fixture corpus; Lance-corrupt fallback test.

### G6 — Durability

- Lance sync: keep records in `_pending` until sync *succeeds* (currently popped before — verified permanent-loss path `engine.py:250-258`); add `clara doctor --backfill-vectors` scanning SQLite rows missing from Lance (tool doesn't exist today); `atexit` flush as belt-and-braces alongside `close()`.
- Optional `remember(..., wait=False)` durability: enqueue to a SQLite `pending_facts` table instead of RAM; worker deletes on success. Crash = replay on next start. (Small; reuses the dead-letter table from G1.)
- Read-after-write: `remember()` (wait=True) ends by synchronously flushing just its own records to Lance (bounded, small) — closes the gap the tests hand-patch with `flush_pending_sync()`.

### G7 — Config sanity

- Delete the 11 dead `ClaraConfig` fields or wire them for real: thread `similarity_threshold`, `retrieval_top_k`, decay knobs into the engines via constructor params (engines keep constants as defaults). One source of truth; `from_env()` documented as the only env reader; stop mutating `os.environ` (pass resolved values explicitly; `interact()` takes the base URL from `self`).
- **One default DB path everywhere:** `~/.clara/clara.db` (the MCP default wins; per-project via `clara init --project`). `ClaraMemory.create()` and the API default to the same resolver `default_db_path()`.
- `close()` must not close the shared Lance singleton unless it owns it (track ownership flag).

### G8 — Visibility & control UX

Extend the existing `clara-mcp` argparse into a real CLI (see Phase 5): `clara list [--type --limit --query]`, `clara forget <id|--query>`, `clara stats`, `clara doctor`, `clara export --json`. MCP server gains one behavior: `memory_save` returns `"saved: <one-line summary>"` so host agents naturally echo what was stored (the "Memory updated" chip equivalent, in text).

### G9 — interact() narrowing (design in Phase 6.3)

### G10 — Scheduler hardening

- Register an APScheduler `EVENT_JOB_ERROR` listener → structured log + `failed_jobs` counter surfaced in `/admin/health` and `clara doctor`; set `misfire_grace_time=3600, coalesce=True` on all jobs.
- Reflection: one transaction **per tenant** (blast radius = that tenant); skip reflection when provider unavailable *loudly* (status in health); drop the `recurring_entity` pattern for bare subject "user" (perpetual noise, verified §1.3); store the insight sentence in `content.object` so it is actually embedded/searchable, or stop calling the LLM for it (the fallback template is equivalent today).
- Decay job invalidates the cache after commit (verified staleness: `scheduler/decay.py` never touches `MemoryCache`).
- In the MCP/embedded profile the scheduler does not run as cron at all — see Phase 8 (opportunistic maintenance).

### G12 — Postgres (deferred, but stop lying)

Fix the false comment at `models.py:201-204` ("behaves identically on SQLite and PostgreSQL" — the `json_extract` key expressions are SQLite-only) and replace `server_default=text("now()")` (PG-ism; verified broken for raw-SQL inserts on SQLite) with `sa.func.current_timestamp()`. Actual PG support waits for demand; the design (dialect-conditional index via `content->>'entity_type'`) is documented in AUDIT_2026-06-14 already.

---

## Phase 5 — CLI-agent integration: one command, every agent

### 5.1 Decision: MCP-first (reuse `clara-mcp`), FastAPI demoted to optional service

| Criterion | MCP stdio (`clara-mcp`) | FastAPI (`clara.main:app`) |
|---|---|---|
| Exists today | ✅ complete + tested | ✅ complete + tested |
| Keyless | ✅ by design | ❌ default backend needs OPENAI_API_KEY |
| Claude Code / Codex / Cursor / Gemini native support | ✅ all speak MCP natively | ❌ nobody speaks it without custom glue |
| Server to run | none (spawned per session over stdio) | uvicorn process, port, auth story (header-trust only) |
| Auth surface | inherits process/user | ❌ `X-User-ID` identification only; admin routes cross-tenant |
| Deployment weight | `pip install clara-memory[mcp]` | `[api]` extra + ops |

**Recommendation:** `clara-mcp` is the integration layer. The FastAPI app survives as the *optional team/service* deployment for people who want the full pipeline centrally (and gets a real auth gate before that's advertised). No new HTTP server is built.

### 5.2 Install flow (the actual commands)

```bash
pip install clara-memory[cli]        # or: uv tool install / pipx install
clara init                            # idempotent, < 1s
```

`clara init` (new module `clara/cli.py`, absorbing `mcp_server.main`'s argparse; console script `clara`, keep `clara-mcp` as alias):
1. Resolve store dir: `~/.clara/` (or `./.clara/` with `--project`, which also drops a `.gitignore`).
2. Create `clara.db`, run schema + FTS5 migration (versioned via `PRAGMA user_version` — no Alembic).
3. Run self-checks (Phase 8), print status.
4. Detect installed CLIs and offer (or `--agent claude-code|codex|gemini|cursor|aider|print`):
   - **Claude Code:** `claude mcp add clara -- clara-mcp` (or write `.mcp.json` entry); plus optional hook block into `~/.claude/settings.json`.
   - **Codex CLI:** append to `~/.codex/config.toml`:
     ```toml
     [mcp_servers.clara]
     command = "clara-mcp"
     ```
   - **Gemini CLI:** `~/.gemini/settings.json` → `mcpServers.clara = {"command": "clara-mcp"}`.
   - **Cursor:** `~/.cursor/mcp.json` same shape.
   - **Aider / anything else (no MCP):** generic path below.
   Every write to a foreign config is shown as a diff and requires `-y` or interactive confirm.

### 5.3 Per-agent wiring (when memory is read/written, with no manual calls)

**Claude Code (first-class):**
- *Read at session start:* SessionStart hook (matchers `startup|resume|compact` — re-injects after compaction, the docs' own recommended pattern):
  ```json
  {"hooks": {"SessionStart": [{"matcher": "startup|resume|compact",
    "hooks": [{"type": "command", "command": "clara-mcp recall --top-k 12"}]}]}}
  ```
  `recall` already prints the context block on stdout → injected. Empty store prints nothing (verified behavior). Budget: hook default timeout is generous; recall is a local SQLite query (ms).
- *Read per-prompt (optional, `clara init --per-prompt`):* UserPromptSubmit hook `clara-mcp recall --query-stdin --top-k 5` (30s hook budget; we use milliseconds). Off by default — session-start injection + on-demand MCP search covers most sessions without per-turn token cost.
- *Write during session:* the model calls MCP `memory_save` when it judges something durable — the docstrings already teach it the four types. A short "when to save" paragraph ships as an optional CLAUDE.md snippet (`clara init` offers to append it), mirroring how ChatGPT's bio tool and LangMem's `manage_memory` are prompted.
- *Write at session end (safety net, optional):* Stop/PreCompact hook pipes `last_assistant_message`/transcript path to `clara ingest --heuristic` (rule-based extractor, Phase 6 — no LLM in the hook path).

**Codex CLI (first-class):** MCP server as above (Codex supports MCP natively via `mcp_servers` in `config.toml`, verified against the [Codex config reference](https://developers.openai.com/codex/config-reference) and [MCP docs](https://developers.openai.com/codex/mcp)). Codex has no SessionStart-hook equivalent, so session-start context uses the AGENTS.md convention: `clara init --agent codex` writes a one-line instruction into `~/.codex/AGENTS.md` ("At the start of a task, call the clara `memory_search` tool with the task topic"). Codex's own "Memories" feature is machine-local summaries; CLARA complements rather than fights it.

**Generic CLI (no MCP):** two commands cover any tool that can shell out:
`clara context "<task description>"` → prints the block (put it in your prompt);
`clara remember "<text>"` → heuristic extraction + store, prints one-line receipt.
Aider: `clara context` output appended to a conventions file, or used via `/run`.

### 5.4 Automatic read/write policy (default profile)

- **Session start:** inject `context_for(project-cwd + recent)` — top 12, recency-weighted (the empty-query lexical mode already ranks by recency+confidence).
- **During session:** reads and writes are host-model-initiated via MCP tools (the Letta/LangMem hot-path pattern — zero extra infrastructure, the host model is already authenticated and already smart).
- **Session end (opt-in):** heuristic sweep of the final transcript for missed durable facts.
- **Never:** CLARA calling its own LLM in any hook path.

---

## Phase 6 — Zero-API-key, zero-heavy-model architecture

Hard constraint honored: no OpenAI/Anthropic key, **no Ollama, no local LLM of any kind** in the core loop.

### 6.1 Packaging: three tiers

```
clara-memory              core: sqlalchemy, aiosqlite ONLY  → LocalMemory, FTS5 lexical,
                          heuristic extraction, MCP-ready storage. No openai. No lancedb.
clara-memory[cli]         + mcp                              → clara / clara-mcp entry points
clara-memory[semantic]    + model2vec (numpy-only, ~8-32MB)  → static-embedding tier
clara-memory[full]        + openai/lancedb/…                 → today's full pipeline
```

Moving `openai` and `lancedb` out of core (`pyproject.toml:39-45`) is the single biggest install-weight win; the keyless path never imports either (verified: `LocalMemory` short-circuits Lance via session flag; imports are already lazy in the right places, remaining eager imports get guarded).

### 6.2 Extraction without an LLM (`clara/extraction/heuristic.py`, new)

Deterministic pipeline producing the same `ExtractedFact` shape:
1. **Sentence split** (regex; no NLTK).
2. **Pattern bank** ordered by specificity — negation transitions first ("switched from X to Y", "no longer uses X" → negation + positive pair, mirroring the LLM prompt's rule 4), then preference/possession/skill/event/world-model verb frames ("I use/prefer X (for D)", "X runs on Y", "learned/knows X", "deployed/finished X"). Relations emitted from the same vocabulary as `classify_memory_type`'s frozensets so downstream classification keeps working unchanged (`update/engine.py:85-102`).
3. **Hedge filter** = the LLM prompt's rule 2 ("maybe/might/I think/probably" → skip) as a stoplist check.
4. **Confidence:** pattern-specific priors (explicit first-person present → 0.85; inferred → 0.55), consistent with the existing 0.4 floor.
5. Optional spaCy tier (`[nlp]` extra, following Mem0's precedent of spaCy-not-LLM entity work) for proper-noun entity normalization — never required.
Precision-first doctrine: extract less, never wrongly — the host agent's `memory_save` remains the *primary* write channel in CLI use; heuristics are the safety net and the `clara remember` backend. Wire-up: `FactExtractor` falls back to it on `llm_unavailable` (G1) and `provider="none"` selects it explicitly. `ClaraMemory.create(llm_provider="none", embedding_backend="lexical")` must construct successfully with zero network access — this becomes the tested default profile.
**Tests:** golden corpus of ~60 utterances → expected fact tuples; negation pairs; hedge suppression; property test that no extraction ever raises.

### 6.3 Embeddings ladder

| Tier | Backend | Weight | When |
|---|---|---|---|
| 0 | **FTS5/BM25 only** (no vectors) | 0 | default keyless install |
| 1 | **model2vec** static embeddings (`potion-base-8M`) | ~8 MB model, numpy only, ~90% of MiniLM quality, 100-500× faster on CPU ([MinishLab/model2vec](https://github.com/MinishLab/model2vec), [potion-base-8M](https://huggingface.co/minishlab/potion-base-32M)) | `[semantic]` extra; vectors stored in SQLite (256-dim float32 blobs — at CLI-memory scale, thousands of rows, brute-force numpy dot is sub-ms; LanceDB unnecessary) |
| 2 | sentence-transformers / OpenAI / Ollama + LanceDB | heavy | `[full]`, existing behavior |
Sentence-transformers is **rejected as the keyless default**: it drags torch (hundreds of MB, slow cold start) — exactly the "heavy local model" the constraint excludes. Vectors get a `backend` tag column (fixes the verified mixed-vector-space hazard when backends switch, Mem0 issue #4985's exact failure class).

### 6.4 What `interact()` means with no LLM

`interact()`'s reply-generation role is **deprecated, not broken**: with `llm_provider="none"` it returns `{"response": None, "memory_context": <block>, "facts_stored": <heuristic-ingest of the user message>, "status": "memory_only"}` — i.e., it degrades to `context_for()` + `remember()` in one call, and the docstring + README direct new users to those two calls. Rationale: in every CLI integration the *host* model is the generator (already authenticated, already better); CLARA generating replies was only ever sensible for the standalone API deployment, where it remains available under `[full]`. This also dissolves the transaction-across-LLM-calls problem and the self-extraction trust bug for everyone except opted-in full-pipeline users (who get the G4.3 fix).

### 6.5 What's honestly unavailable keyless (accepted trade-offs)

- LLM conflict adjudication — never existed in CLARA anyway (conflict logic is already heuristic; keyless loses nothing).
- Nuanced extraction of oblique phrasing — mitigated by host-model `memory_save` being the primary channel.
- Reflection insight *sentences* — pattern detection is already pure Python; the LLM sentence is provably cosmetic today (§1.3), so keyless reflection keeps the useful part.
- Semantic recall below tier 1 — FTS5+porter stemming is the floor, which is where most of the local-first category lives ([studiomeyer local-memory-mcp](https://github.com/studiomeyer-io/local-memory-mcp), [memento](https://github.com/iAchilles/memento), [memoirs](https://github.com/misaelzapata/memoirs)).

---

## Phase 7 — Cleanup (de-duplicated, with verdicts)

Root docs (lineage verified: BUG_FIXES v2 → V4_REPORT/buggy → AUDIT_2026-06-14; all 13 canonical bugs re-verified FIXED by code spot-checks):

| File | Verdict | Why |
|---|---|---|
| `buggy.md` | **Delete** | self-declared tombstone, 8 lines |
| `CLARA_V4_BUG_REPORT.md` | **Delete** (git history preserves it) | superseded banner; stale "❌ Not fixed" tables are re-fix bait for agents — the exact hazard AUDIT:13 warns about |
| `CLARA_BUG_FIXES.md` | **Delete** | all 8 bugs fixed; 700+ lines of obsolete instructions whose proposed defaults differ from shipped code — actively misleading |
| `CLARA_OLLAMA_PROMPT.md` | **Delete** | spent one-shot prompt; contains now-false constraints ("Do Not Touch engine.py") |
| `INSTALLATION_PLAN.md` | **Delete** | fully absorbed into README |
| `implementation_status.md` | **Delete** | body contradicts its own addenda; dead `file:///c:/Users/Administrator/...` links |
| `clara_implementation_plan.md`, `pending_implementation_plan.md` | **Archive → `docs/history/`** | historical value (design rationale) but target the abandoned Postgres/pgvector/Redis/Docker stack |
| `AUDIT_2026-06-14.md` | **Keep** (move to `docs/`) | authoritative prior audit |
| `HALLUCINATION_REPORT.md` | **Keep, refresh** | honest scoping doc; stale test count (371 vs current) |
| `README.md` | **Rewrite** per this plan | must lead with MCP/keyless path; fix env-var omissions (`CLARA_AUTH_REQUIRED`, `CLARA_CORS_ORIGINS`, `CLARA_DB_PATH`/`CLARA_HOME`); drop static test-count badge |
| `Closo_Website_Quotation.docx` | **Remove** | client business document, unrelated — also a data-hygiene concern (private pricing in a public repo); needs history scrub consideration if repo is public |

Code/scripts:

| Item | Verdict |
|---|---|
| `tmp_check_retrieve.py` | **Delete** — committed debug scratch (tracked since 2026-03-11), duplicates test coverage |
| `examples_openclaw_bridge_demo.py` | **Delete or move to `examples/`** — silently prints zero results without a key (the G1 bug in demo form) |
| `clara/integrations/openclaw_bridge.py` (+ its tests) | **Deprecate** — session-partitioned `user_id="session:{id}"` defeats cross-session memory; superseded by the MCP path. Keep one release with a deprecation warning |
| `scripts/migrate_to_lancedb.py` | **Delete** — Postgres-source one-shot from the pgvector era; no supported install can produce its input |
| `clara/core/schemas.py` (`ExtractionCandidate`, `MemorySummary`, duplicate `InteractionRecord`) | **Delete** + retarget `tests/test_core.py` — production-unused, superseded |
| Never-raised exceptions (`ConflictError`, `TenantViolationError`, `ConfigurationError`) | **Delete** (or start raising `ConfigurationError` for missing-key cases — preferred, pairs with G1) |
| Dead code list: `STABLE_DECAY_RATE`, `RetrievalEngine._cosine_similarity`, `LanceMemoryRecord.is_new`, embeddings `get_engine()` singleton (test-only), `ActionTaken.skipped`, unused imports across `agent.py`/`engine.py`/`decay.py` | **Delete**; keep `ActionTaken.skipped` removal noted in AUDIT (intentional WONTFIX there) |
| Vestigial pgvector/JSONB `@compiles` fixture blocks duplicated in ~12 test files | **Delete repo-wide** — dead since the LanceDB migration; `pgvector` isn't even a dependency |
| `format_context` duplication (`agent.py` vs `reasoning/context.py`) | **Fix** — single module (`clara/core/format.py`), both import |
| Dead dev deps `hypothesis`, `pytest-mock` | **Remove** from `[dev]` (or actually adopt hypothesis for the heuristic extractor — preferred) |
| CI gap: `[api]`/`[mcp]` not installed → ~490 lines of route/MCP tests silently skipped in CI | **Fix** — CI installs `.[dev,api,mcp]` |

---

## Phase 8 — Always-on reliability ("works every session, no server, no error")

### 8.1 Process model: embedded + on-demand subprocess. No daemon.

- **Library callers:** in-process `LocalMemory`/`ClaraMemory` — no separate process at all.
- **CLI agents:** `clara-mcp` spawned by the host over stdio per session, dies with the session. SQLite WAL + 30s busy_timeout (already configured, `agent.py:106-108`) makes concurrent sessions safe — WAL allows one writer + many readers, and CLI memory write rates are trivially low.
- **Rejected — background daemon:** violates "no server", adds a liveness problem worse than the one it solves, and nothing here needs sub-ms shared state.
- Maintenance (decay/prune) stops being cron in this profile: **opportunistic maintenance** — on MCP server start, if `meta.last_maintenance < now-24h`, run decay+prune inline (bounded: LIMIT 500 rows per pass, resumable) then stamp. No scheduler process, same effect, and it fixes G10's silent-cron problem for the CLI profile wholesale.

### 8.2 Startup self-checks (target < 50 ms total, all fail-soft)

On first tool call (not import): (1) store dir exists/writable (create, else read-only mode); (2) `PRAGMA quick_check` on a 1-page sample + `user_version` matches expected schema (mismatch → run migrations; failure → degrade); (3) FTS5 table present (absent → rebuild from `memories` — it's derived data); (4) vector tier sanity if `[semantic]` installed (model file loads; dim tag matches stored vectors — mismatch → lexical-only + warn). Any failure sets `Health(degraded=True, reasons=[...])`; tools keep answering from whatever tier survives. **The host agent's session must never crash because memory is sick** — worst case every tool returns `{"status": "memory_unavailable", "reason": ...}` and `recall` prints nothing (its verified current behavior for an empty store).

### 8.3 Failure-mode table (each defined, non-fatal)

| Failure | Behavior |
|---|---|
| First run, no DB | create dir + schema silently (verified: `default_db_path` already mkdirs) |
| Corrupted SQLite | quick_check fails → rename `clara.db.corrupt-<ts>`, start fresh, WARN with recovery hint (`clara doctor --recover` attempts `.recover` into the new DB); session proceeds with empty memory rather than crashing |
| Corrupted/missing FTS index | drop + rebuild from source table (derived data) |
| Corrupted vector blobs / model missing | lexical-only tier, warn once |
| Disk full | writes fail → `memory_save` returns `{"status": "disk_full"}`; reads still work (SQLite reads need no space); never raise through MCP |
| Concurrent write race | WAL + busy_timeout 30s + bounded retry on `SQLITE_BUSY`; world_model races already guarded by the partial unique index + savepoint retry (verified `world_model.py:108-139`) |
| Schema from a newer version | refuse writes, allow reads, tell user to upgrade (`user_version` gate) |

### 8.4 Health check for hosts

`clara doctor` (human) and MCP `memory_stats` (already exists — gains `health` fields): `{ok, degraded, tier: "lexical|semantic|full", db_path, schema_version, active_memories, last_maintenance, failed_writes}`. SessionStart hooks may call `clara doctor --quiet` (exit 0 healthy / 1 degraded-but-usable / 2 unusable) — milliseconds, no network.

---

## Sequenced roadmap

| Step | Contents | Size |
|---|---|---|
| 1 | G11 cleanup + G13 README rewrite (safe, immediate trust win) | XS |
| 2 | G1 loud failures + G4 bug fixes (correctness before features) | M |
| 3 | G5 FTS5 + hybrid retrieval; G7 config/path unification | M |
| 4 | Phase 6: packaging tiers, heuristic extractor, model2vec tier, interact() narrowing | M |
| 5 | Phase 5: `clara init` + per-CLI adapters + docs | M |
| 6 | Phase 8: self-checks, doctor, opportunistic maintenance; G6 durability | M |
| 7 | G10 scheduler hardening (service profile); G12 honesty fixes | S |

## Appendix — Open questions & risks

1. **PyPI reality:** README badges claim `clara-memory` on PyPI — unverified; if unpublished, every install instruction fails. Verify/publish before advertising `pip install`.
2. **Repo publicity vs the Closo docx:** if the GitHub repo is public, the client quotation has been public; consider history scrubbing (BFG), not just deletion.
3. **Embedding-tier migration:** moving tier 0→1 later requires embedding backfill (`clara doctor --backfill-vectors` covers it); tier changes must refuse to mix vector spaces (backend tag column).
4. **MCP tool-call reliance:** the primary write channel depends on host models actually calling `memory_save`; mitigations are the docstring quality (already good), the CLAUDE.md/AGENTS.md snippet, and the heuristic session-end sweep. Field evidence (ChatGPT bio tool, LangMem manage_memory) says models do call well-described memory tools.
5. **Multi-user scoping in MCP tools:** deliberately single-user today (per-OS-user store). Fine for CLI use; the service profile keeps user_id. Adding tenancy to MCP tools is deferred until someone actually shares a store.
6. **Rate-limited research residue:** three research agents (Codex-ecosystem deep dive, local-first survey, complaint mining) were partially completed inline with primary citations after hitting a session rate limit; conclusions drawn only from verified sources listed below. A fuller complaint-mining pass can only strengthen, not weaken, the C1-C3 rankings (they are already the field's loudest issues).

### Source list (comparative research)
- Mem0: [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) (source @ v2.0.11), [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), [docs.mem0.ai migration v2→v3](https://docs.mem0.ai/migration/oss-v2-to-v3), [mem0.ai/research](https://mem0.ai/research)
- Letta/MemGPT: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560), [docs.letta.com](https://docs.letta.com/concepts/memory-management/), [letta-code (MemFS)](https://github.com/letta-ai/letta-code), issues #2388/#2772/#2512
- Zep/Graphiti: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), [github.com/getzep/graphiti](https://github.com/getzep/graphiti), [help.getzep.com](https://help.getzep.com/concepts), issues #1489/#1299/#868
- LangMem: [langchain-ai/langmem](https://github.com/langchain-ai/langmem), [extraction.py source](https://raw.githubusercontent.com/langchain-ai/langmem/main/src/langmem/knowledge/extraction.py), [langmem docs](https://langchain-ai.github.io/langmem/)
- ChatGPT memory: [Embrace The Red teardown](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work/), [Simon Willison](https://simonwillison.net/2025/May/21/chatgpt-new-memory/), OpenAI Memory FAQ + Dreaming V3 announcement (via search snippets; direct fetch 403)
- Claude Code: [code.claude.com/docs — memory](https://code.claude.com/docs/en/memory), [hooks](https://code.claude.com/docs/en/hooks), [MCP](https://code.claude.com/docs/en/mcp), [skills](https://code.claude.com/docs/en/skills)
- Codex CLI: [config reference](https://developers.openai.com/codex/config-reference), [MCP docs](https://developers.openai.com/codex/mcp), [memory-layer ecosystem overview](https://codex.danielvaughan.com/2026/04/06/codex-cli-persistent-memory-mcp-servers/)
- Local-first prior art: [model2vec](https://github.com/MinishLab/model2vec), [potion models](https://huggingface.co/minishlab/potion-base-32M), [studiomeyer/local-memory-mcp](https://github.com/studiomeyer-io/local-memory-mcp), [memento](https://github.com/iAchilles/memento), [memoirs](https://github.com/misaelzapata/memoirs), [sqliteai/sqlite-memory](https://github.com/sqliteai/sqlite-memory)
- Complaints: Mem0 issues [#5245](https://github.com/mem0ai/mem0/issues/5245), [#3009](https://github.com/mem0ai/mem0/issues/3009), [#2443](https://github.com/mem0ai/mem0/issues/2443), [#4985](https://github.com/mem0ai/mem0/issues/4985), [#4037](https://github.com/mem0ai/mem0/issues/4037); [AI-amnesia roundup](https://duet.so/guides/ai-amnesia-why-your-ai-keeps-forgetting)

---

*Implementation of this plan begins after explicit approval; the sequenced roadmap above is the execution order.*
