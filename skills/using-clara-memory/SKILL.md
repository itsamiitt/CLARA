---
name: using-clara-memory
description: When and how to use CLARA persistent memory — save durable facts, search before asking the user, trust document tiers, link entities.
---

# Using CLARA memory

CLARA is a plain SQLite store exposed through 17 MCP tools — 6 memory
(`memory_save`, `memory_search`, `memory_recent`, `memory_update`,
`memory_forget`, `memory_stats`), 5 docs (`docs_status`, `docs_classify`,
`docs_supersede`, `docs_fulfill`, `docs_report`), 4 graph (`graph_entity`,
`graph_neighbors`, `graph_path`, `memory_link`), 2 status bar
(`statusline_install`, `statusline_status` — only for `/clara:statusline`;
never call them unprompted, they edit the user's settings). **You** are the only
intelligence — there is no backend model doing extraction or embeddings.
You decide what to store and how to query it. Never store credentials: the
save path rejects secret-shaped content — store a reference (env var name,
vault path) instead.

## When to recall

- At the start of a non-trivial task, call `memory_search` with the key terms
  (the user, the project, the technology, the decision at hand) before acting.
- **Search before asking** the user something they may already have told you —
  preferences, stack choices, past decisions.
- Use `memory_recent` to see what is top-of-mind without a query.

## When to save (`memory_save`)

Save durable, reusable facts — not transient chatter. Pick the type:

- **belief** — stable preference or fact. Requires `subject`, `relation`,
  `object` (e.g. subject="user", relation="prefers", object="pytest over
  unittest"). Corrections: save the new belief and mark the old one wrong with
  `is_negation: true` (e.g. "user switched from npm to pnpm" → negation of
  "user uses npm" plus a new belief).
- **event** — a notable thing that happened. Requires `subject`,
  `event_type`; add a `description` (a migration, a release, an incident,
  an architectural decision).
- **skill** — a reusable procedure worth repeating. Requires `name`; add
  `trigger_conditions` (when to use it) and `steps` (how).
- **world_model** — current state of a service/tool/repo. Requires
  `entity_type`, `name`; put state in `properties`. Upserts replace the
  active record for the same entity.

Add `tags`, a `domain`, and `confidence` (0..1) when useful. Use
`memory_update` to adjust confidence/tags, `memory_forget` to retire a memory
(it is never hard-deleted).

## Hygiene

- **Never store secrets** — no API keys, tokens, passwords, or credentials,
  even if the user pastes them.
- Do not save what the repo or git history already records, or one-off
  trivia. Save what was non-obvious and will matter again.
- Prefer specific, atomic memories over long blobs.

## Linking (knowledge graph)

Beliefs project into a knowledge graph (nodes + edges) automatically; use
`memory_link(src, relation, dst)` when you specifically want the graph edge
back, and `graph_entity` / `graph_neighbors` / `graph_path` to explore.

- **Entity naming**: name code entities by path (`src/api.py`), symbol
  (`src/api.py::handler`), or decision slug (`decision:move-to-sqlite`).
  Consistent names keep the graph connected.
- **Relations in active voice**: `uses`, `depends_on`, `deployed_to`,
  `prefers` — subject acts on object. Relations are normalized to
  snake_case lemmas, so "Depends On" and "depends" land on `depends_on`.
- **Propose merges, never pick silently**: when a node card shows
  `possible_duplicates`, tell the user and ask which node wins instead of
  guessing. Resolution creates a new node rather than merging on its own.

## Document trust (curator ledger)

A `[KNOWLEDGE MAP]` block at session start summarizes the repo's document
ledger: authoritative (T0/T1) docs, active T2 work, quarantined and archived
entries. Trust rules:

- **Consult `docs_status` before executing a plan-type document** — it may
  be stale, fulfilled, or superseded even though the file still exists.
- **On conflict prefer T0/T1** (pinned/authoritative) over T2/T3 content,
  and say you did so.
- Treat quarantined (superseded/TX) and archived documents as historical
  record, not current guidance.
- The ledger updates via `clara docs scan`; if a document is missing from
  it, suggest a scan rather than guessing.

## Fulfillment (closing out plans)

- **Immediately after completing the work a plan-type document described**,
  call `docs_fulfill` with 1-5 distilled durable facts — the decisions,
  constraints, and standards that outlive the plan, not implementation
  trivia. Pass the confirming commit/PR ref as evidence. `/clara:done`
  walks this flow with user confirmation.
- **When authoring a v2 of a plan**, call `docs_supersede(old, new)` so the
  old plan is quarantined and future reads of it get annotated.
- **When a fact changes**, save the correction (negation or new belief) and
  let supersession invalidate the old edge — never claim deletion; CLARA
  retires, it does not destroy.

## Proportionality

A `=== MEMORY CONTEXT ===` block is injected at session start. For trivial
tasks that block already covers, do not spend tool calls on memory — answer
directly. Reach for the tools when the task is non-trivial, when the user
references past context, or when you learn something durable.
