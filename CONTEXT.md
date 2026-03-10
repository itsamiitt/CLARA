# Cognitive Living Memory Architecture for AI Agents
### Version 2.0 — Improved Design

---

## Overview

This document describes a **modular, living-memory architecture for autonomous AI agents**. It is designed for agents that operate across extended time horizons, handle evolving knowledge, and must reason reliably across multiple sessions.

The architecture is built on five principles:

- **Separation of memory types** — different knowledge requires different storage strategies
- **Non-destructive updates** — facts are versioned, not overwritten
- **Temporal awareness** — all memory carries provenance and timestamps
- **Conflict resolution** — contradictory facts are adjudicated, not silently merged
- **Adaptive forgetting** — low-confidence, stale memory decays rather than accumulating

---

## Memory Types

The architecture separates memory into **four complementary stores**:

| Type | What it holds | Analogy |
|---|---|---|
| **Belief Memory** | Atomic facts about entities | Semantic memory |
| **Event Memory** | Time-indexed occurrences | Episodic memory |
| **Skill Memory** | Reusable procedural knowledge | Procedural memory |
| **World Model** | Live environment state | Spatial/situational awareness |

---

## High-Level Architecture

```
User / Environment / Tools
          │
          ▼
   ┌──────────────┐
   │  Interaction │   ← Raw input: messages, API results, file changes, tool outputs
   │     Layer    │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │    Fact      │   ← Entity/relation extraction, event detection, preference capture
   │  Extraction  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │              Memory Update Engine             │
   │  similarity search → type classification      │
   │  → conflict check → create / update / merge  │
   └──────┬───────────┬──────────────┬────────────┘
          │           │              │           │
          ▼           ▼              ▼           ▼
       Belief       Event          Skill       World
       Memory       Memory         Memory      Model
          │           │              │           │
          └───────────┴──────────────┴───────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │  Unified Memory  │   ← Shared store: PostgreSQL + vector index
                   │      Store       │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  Vector Retrieval│   ← Semantic search + recency + confidence ranking
                   │     Engine       │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │    Reasoning     │   ← Chain-of-thought, planning, tool use
                   │     Engine       │
                   └──────────────────┘
```

---

## Component Details

### 1. Interaction Layer

Receives and normalizes all inputs into a standard `InteractionRecord`:

```json
{
  "interaction_id": "int_20260310_0042",
  "source": "user",
  "raw_input": "I switched from Python to Rust for my systems work.",
  "timestamp": "2026-03-10T14:22:00Z",
  "session_id": "sess_abc123",
  "confidence_floor": 0.5
}
```

Input sources: user messages · tool outputs · API responses · file system events · environment signals

---

### 2. Fact Extraction

Converts raw interactions into **typed knowledge candidates** before memory is touched.

**What is extracted:**

- Entities (people, projects, systems, technologies)
- Relations (uses, builds, prefers, owns, manages)
- Events (started, completed, failed, updated)
- Preferences and constraints
- Negations and corrections (`no longer`, `switched from`, `stopped`)

**Extraction example:**

```
Input:    "I switched from Python to Rust for my systems work."

Extracted:
  - Entity: user
  - Relation update: user → uses → Rust (NEW)
  - Relation invalidation: user → uses → Python (for systems work)
  - Context tag: domain = systems
```

**Extraction rules:**

- Ignore hedged statements (`maybe`, `I think`, `might`)
- Flag negations explicitly — never silently discard
- Tag domain context when available (`for web work`, `at my day job`)
- Score extraction confidence; discard candidates below 0.4

---

### 3. Memory Update Engine

The update engine is the most critical and complex component. It determines how an extracted fact modifies the memory stores.

#### Processing pipeline

```
Extracted fact
     │
     ▼
[1] Similarity Search
     Embed the fact, search Unified Memory Store
     Return top-k candidates (k = 10, threshold = 0.82)
     │
     ▼
[2] Memory Type Classification
     Classifier assigns fact to: Belief / Event / Skill / World Model
     Ambiguous facts may produce multiple candidates
     │
     ▼
[3] Conflict Detection
     Compare against retrieved candidates:
       - Direct contradiction?  → Conflict Resolution
       - Refinement / addition? → Update with versioning
       - Novel fact?            → Create new record
     │
     ▼
[4] Write
     Non-destructive: version old record, write new
```

#### Conflict Resolution Protocol

When a new fact directly contradicts an existing belief:

```
CONFLICT DETECTED
  Existing: user → uses → Python  (confidence: 0.85, last seen: 2024-11)
  Incoming: user → uses → Rust    (confidence: 0.72, timestamp: 2026-03)

Resolution steps:
  1. Check temporal order — newer fact wins by default
  2. Apply source weight (see Confidence Scoring)
  3. If incoming confidence > 0.6 AND newer → supersede old belief
  4. Mark old belief as: status = "superseded", superseded_by = <new_id>
  5. Retain old belief for audit trail — never delete
  6. If confidence is ambiguous → store both with domain tags
```

Conflict classes:

| Class | Example | Resolution |
|---|---|---|
| **Direct replacement** | Python → Rust | Supersede old, keep audit |
| **Domain-scoped** | Python at work, Rust personally | Retain both with context tags |
| **Temporal overlap** | Project active vs completed | Use timestamps to order |
| **Source disagreement** | User says X, tool log says Y | Source-weight arbitration |

---

## Memory Systems

### 4. Belief Memory

Stores **atomic facts** about entities. Each belief is a single subject–relation–object triple, not a compound sentence.

#### Schema

```json
{
  "belief_id": "bel_user_lang_rust_001",
  "subject": "user",
  "relation": "uses",
  "object": "Rust",
  "domain": "systems",
  "confidence": 0.78,
  "source_weight": 0.9,
  "evidence": [
    {
      "text": "I switched from Python to Rust for my systems work.",
      "source": "user",
      "timestamp": "2026-03-10T14:22:00Z"
    }
  ],
  "status": "active",
  "supersedes": "bel_user_lang_python_001",
  "created_at": "2026-03-10T14:22:00Z",
  "updated_at": "2026-03-10T14:22:00Z",
  "decay_rate": 0.02,
  "embedding": [...]
}
```

#### Confidence Scoring

Confidence uses **source-weighted Bayesian updating**, not a naive evidence count:

```
Source weights:
  User direct statement         → 1.0
  User indirect implication     → 0.7
  Tool/API log                  → 0.85
  System observation            → 0.75
  Agent inference               → 0.5

Bayesian update formula:
  confidence_new = (prior × decay_factor + source_weight × observation_strength)
                  ─────────────────────────────────────────────────────────────
                               (prior_weight + 1)

  where:
    decay_factor   = e^(−decay_rate × days_since_last_seen)
    prior_weight   = total weighted observations so far
```

This means a single strong direct statement can establish high confidence, while a weak inference requires multiple confirmations.

---

### 5. Event Memory

Records **time-indexed occurrences** — things that happened, not stable facts.

#### Schema

```json
{
  "event_id": "evt_20260310_proj_start",
  "event_type": "project_started",
  "entity": "systems-rewrite",
  "description": "User began rewriting backend systems in Rust",
  "timestamp": "2026-03-10T14:22:00Z",
  "duration": null,
  "outcome": null,
  "related_beliefs": ["bel_user_lang_rust_001"],
  "tags": ["engineering", "backend", "rust"],
  "embedding": [...]
}
```

#### Event Lifecycle

```
created → in_progress → completed
                     → failed
                     → abandoned
```

Events are **never overwritten** — only outcomes are appended. This preserves a full activity log for reflection and insight generation.

---

### 6. Skill Memory

Stores **reusable procedural knowledge** — the most differentiated component of this architecture.

#### Schema

```json
{
  "skill_id": "skl_deploy_rust_api",
  "name": "Deploy Rust API to Linux server",
  "trigger_conditions": [
    "user is deploying a Rust service",
    "target is a Linux system"
  ],
  "steps": [
    "Run `cargo build --release`",
    "Copy binary to /usr/local/bin/",
    "Configure systemd service file",
    "Run `systemctl enable && systemctl start <service>`"
  ],
  "confidence": 0.82,
  "success_count": 4,
  "failure_count": 1,
  "failure_notes": ["Failed when binary linked against glibc not present on target"],
  "last_used": "2026-02-28T09:10:00Z",
  "source": "user_instruction",
  "embedding": [...]
}
```

#### Skill Learning Sources

- Previous successful task executions
- Explicit user instructions
- External documentation
- Tool execution logs
- Agent-synthesized procedures from event patterns

#### Skill Feedback Loop

```
Skill executed
     │
     ├─ Success → success_count++, confidence += 0.05 (max 0.99)
     │
     └─ Failure → failure_count++, confidence -= 0.10
                  failure_notes.append(error_context)
                  if confidence < 0.3 → flag for review
                  if confidence < 0.15 → status = "deprecated"
```

Skills **never deleted on failure** — failure notes are retained as training data for refinement.

---

### 7. World Model

Stores the **current structured state of the agent's environment**.

Unlike beliefs (which are relational facts), the world model is a **live, mutable snapshot** of system state.

#### Schema

```json
{
  "model_id": "wm_project_systems_rewrite",
  "entity_type": "project",
  "name": "systems-rewrite",
  "properties": {
    "language": "Rust",
    "framework": "Axum",
    "database": "PostgreSQL",
    "deployment": "Linux VPS",
    "status": "in_progress"
  },
  "last_updated": "2026-03-10T14:22:00Z",
  "related_beliefs": ["bel_user_lang_rust_001"],
  "related_events": ["evt_20260310_proj_start"],
  "embedding": [...]
}
```

The world model is the **only memory type that allows direct property mutation** — because it represents live state, not historical knowledge. All mutations are logged with a timestamp and prior value.

---

## Unified Memory Store

All four memory types share a common physical store.

#### Record schema (all types)

```
memory_id       UUID
memory_type     belief | event | skill | world_model
content         JSONB
embedding       vector(1536)
confidence      float
status          active | superseded | deprecated | archived
decay_rate      float
created_at      timestamp
updated_at      timestamp
metadata        JSONB
```

#### Recommended stack

| Component | Recommended Options |
|---|---|
| Primary store | PostgreSQL with pgvector |
| Vector index | HNSW index via pgvector, or Qdrant for scale |
| Embedding model | OpenAI `text-embedding-3-small` or `nomic-embed-text` (local) |
| Cache layer | Redis (hot belief cache for active sessions) |
| Agent framework | Custom agent loop or LangGraph |

---

## Vector Retrieval Engine

Retrieves relevant memory given a query context.

#### Pipeline

```
Query text
     │
     ▼
Embed query → vector(1536)
     │
     ▼
Similarity search over Unified Memory Store
(filter by status = "active")
     │
     ▼
Score candidates:
  final_score = (0.65 × semantic_similarity)
              + (0.20 × confidence)
              + (0.10 × recency_score)
              + (0.05 × usage_frequency)

  where:
    recency_score  = e^(−λ × days_since_last_accessed)  [λ = 0.01]
    usage_frequency = log(1 + access_count) / log(1 + max_access_count)
     │
     ▼
Return top-k records (k = 8, configurable)
Grouped by memory type for structured context injection
```

---

## Reasoning Engine

Assembles retrieved memories into a structured context block for the LLM.

#### Context injection format

```
=== MEMORY CONTEXT ===

[BELIEFS]
- User uses Rust (confidence: 0.78, domain: systems)
- User is building a systems rewrite project (confidence: 0.85)

[WORLD MODEL]
- Project: systems-rewrite | Language: Rust | DB: PostgreSQL | Status: in_progress

[RECENT EVENTS]
- 2026-03-10: User started systems rewrite in Rust

[RELEVANT SKILLS]
- Deploy Rust API to Linux server (confidence: 0.82)

=== END MEMORY CONTEXT ===
```

The reasoning engine may use:

- Chain-of-thought reasoning
- Tool invocation
- Multi-step planning
- Skill retrieval and execution

---

## Memory Evolution

### Temporal Updates

Facts are versioned, not replaced. When a belief changes:

```
Before: user → uses → Python  (status: active)
After update:
  user → uses → Python  (status: superseded, superseded_by: bel_002)
  user → uses → Rust    (status: active)
```

Both records are retained. Queries default to `status = active` but can query history.

### Confidence Decay

All beliefs decay over time if not reinforced:

```
confidence_t = confidence_0 × e^(−decay_rate × Δt_days)

Decay rates by memory type:
  Belief (stable fact)   → 0.005  (slow decay)
  Belief (volatile fact) → 0.02   (medium decay)
  Event                  → 0.0    (events do not decay — they happened)
  Skill                  → 0.01   (decays if not used)
  World Model property   → 0.03   (decays quickly — state changes often)

Archival threshold: confidence < 0.15 → status = "archived"
```

### Pruning and Eviction

To prevent memory bloat in long-running agents:

```
Scheduled pruning (weekly):
  1. Archive beliefs with confidence < 0.15
  2. Archive events older than 90 days with no linked beliefs
  3. Deprecate skills with confidence < 0.15 or unused > 60 days
  4. Compact world model: remove properties not updated in 30 days
```

Archived records are never deleted — they are moved to cold storage and remain queryable for reflection tasks.

---

## Reflection and Insight Generation

Periodic reflection synthesizes higher-order knowledge from raw memory.

#### Pipeline

```
Trigger: scheduled (daily) or threshold (N new events)
     │
     ▼
Retrieve memory cluster (topic-grouped)
     │
     ▼
Pattern detection:
  - recurring entities → candidate belief
  - repeated event type → behavioral pattern
  - common skill triggers → skill generalization
     │
     ▼
Generate insight (via LLM)
     │
     ▼
Store as new belief with:
  source = "agent_reflection"
  confidence = 0.5 (starts lower, rises with reinforcement)
```

**Example:**

```
Raw events:
  2026-01: User deployed Rust service
  2026-02: User optimized Rust binary size
  2026-03: User switched Python project to Rust

Insight generated:
  "User systematically migrates projects from Python to Rust"
  → stored as belief, confidence: 0.5
```

---

## Privacy and Multi-Tenant Isolation

For systems serving multiple users, memory must be strictly partitioned.

#### Isolation requirements

- All memory records carry a `user_id` or `agent_id` partition key
- Vector searches are always filtered by partition key before similarity scoring
- Cross-user belief inference is **prohibited** — no shared belief pools
- World models are never shared between tenants
- Skill memory **may** be shared across agents in a read-only global skill library, if explicitly configured

---

## Design Principles

| Principle | Description |
|---|---|
| **Atomic storage** | One fact per belief record — no compound sentences |
| **Non-destructive** | Facts are versioned and superseded, never deleted |
| **Evidence-backed** | Every belief cites its source interactions |
| **Conflict-aware** | Contradictions are resolved explicitly, not silently |
| **Temporally indexed** | All records carry creation, update, and last-seen timestamps |
| **Decay-driven** | Memory fades without reinforcement |
| **Retrieval-driven reasoning** | Memory is assembled at query time, not pre-baked into prompts |
| **Failure-retaining** | Failed skills and superseded beliefs are archived, not erased |

---

## Implementation Roadmap

### Phase 1 — Foundation
- [ ] Interaction Layer + Fact Extraction (regex + LLM hybrid)
- [ ] Belief Memory with PostgreSQL + pgvector
- [ ] Basic retrieval engine
- [ ] Simple update engine (create / update, no conflict resolution yet)

### Phase 2 — Robustness
- [ ] Conflict Detection and Resolution Protocol
- [ ] Event Memory
- [ ] Confidence decay + archival scheduler
- [ ] World Model

### Phase 3 — Intelligence
- [ ] Skill Memory with feedback loop
- [ ] Reflection and insight generation
- [ ] Multi-tenant isolation
- [ ] Redis cache layer for hot beliefs

### Phase 4 — Scale
- [ ] Migrate vector index to Qdrant or Weaviate
- [ ] Async memory updates (non-blocking writes)
- [ ] Memory analytics dashboard
- [ ] Benchmark against Mem0 / Zep

---

## Summary

This architecture gives an AI agent the ability to:

- **Know** — through structured, evidence-backed belief memory
- **Remember what happened** — through timestamped event memory
- **Learn how to act** — through skill memory with failure feedback
- **Understand its environment** — through a live world model
- **Forget gracefully** — through confidence decay and archival
- **Resolve contradiction** — through explicit conflict adjudication
- **Grow smarter over time** — through periodic reflection and insight synthesis

The combination of these capabilities supports agents that operate reliably across **weeks, months, and years** of continuous interaction.
