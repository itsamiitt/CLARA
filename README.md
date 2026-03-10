<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/v/clara-memory?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/pyversions/clara-memory?label=Python" alt="Python 3.10+"></a>
  <a href="https://github.com/itsamiitt/CLARA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/itsamiitt/CLARA"><img src="https://img.shields.io/badge/docs-github-blue" alt="Documentation"></a>
</p>

<h1 align="center">CLARA</h1>
<p align="center"><b>Cognitive Living Architecture for Reliable Agents</b></p>

---

```bash
pip install clara-memory
```

## Why CLARA?

Existing agent memory systems like Mem0 and Zep treat memory as a flat key-value cache or a simple vector log — they store what was said, but they don't *understand* it. CLARA is fundamentally different: it separates memory into four cognitive types (beliefs, events, skills, and a live world model), versions every fact instead of overwriting it, resolves contradictions through an explicit conflict-adjudication protocol, and lets stale knowledge **decay** naturally over time rather than accumulating forever. The result is an agent that doesn't just *retrieve* past context — it **knows**, **learns**, **forgets gracefully**, and **grows smarter** across weeks and months of continuous interaction.

## 30-Second Quickstart

```python
import asyncio
from clara.agent import ClaraMemory

async def main():
    agent = await ClaraMemory.create(
        db_url="postgresql+asyncpg://user:pass@localhost:5432/clara",
        embedding_backend="openai",   # or "local"
        llm_provider="openai",        # or "anthropic"
    )

    # Store facts — extraction, classification & conflict resolution happen automatically
    await agent.remember("I switched from Python to Rust for my systems work.")

    # Retrieve ranked, type-grouped memories
    results = await agent.recall("What language does the user prefer?")

    # Get a ready-to-inject context block for your LLM prompt
    ctx = await agent.context_for("Help me deploy my service.")
    print(ctx)

    await agent.close()

asyncio.run(main())
```

**Output:**

```
=== MEMORY CONTEXT ===

[BELIEFS]
- user uses Rust (confidence: 0.78, domain: systems)

[WORLD MODEL]
- (none)

[RECENT EVENTS]
- 2026-03-10: user switched_to Rust

[RELEVANT SKILLS]
- (none)

=== END MEMORY CONTEXT ===
```

## Comparison

| Feature | **CLARA** | Mem0 | Zep | LangMem | Naive RAG |
|---|---|---|---|---|---|
| **Knowledge Depth** | 4 typed memory stores (belief, event, skill, world model) with Bayesian confidence scoring | Flat key-value facts | Summaries + raw messages | Graph triples | Chunked documents |
| **Skill Learning** | ✅ Procedural memory with success/failure feedback loop & auto-deprecation | ❌ | ❌ | ❌ | ❌ |
| **Conflict Resolution** | ✅ Explicit protocol: temporal ordering → source weighting → supersession with full audit trail | ❌ Silent overwrite | ❌ Last-write wins | Partial (graph merge) | ❌ Duplicates |
| **Cost Model** | Self‑hosted Postgres + pgvector; pay only for embeddings & LLM calls | SaaS per-request | SaaS subscription | LLM calls per update | Embedding + retrieval |
| **Non-destructive Updates** | ✅ All facts versioned — superseded, never deleted | ❌ | ❌ | ✅ | ❌ |
| **Temporal Decay** | ✅ Confidence decays via `e^(−λt)` with configurable rates per memory type | ❌ | ❌ | ❌ | ❌ |
| **Multi-tenant Isolation** | ✅ Partition-key isolation; cross-user inference prohibited | ✅ | ✅ | ❌ | ❌ |

## Architecture

```
User / Environment / Tools
          │
          ▼
   ┌──────────────┐
   │  Interaction  │   ← Raw input: messages, API results, tool outputs
   │    Layer      │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │     Fact      │   ← Entity/relation extraction, negation detection
   │  Extraction   │     Confidence scoring, domain tagging
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────────────────────┐
   │            Memory Update Engine                   │
   │  Embed → Similarity Search → Type Classification │
   │  → Conflict Detection → Resolution → Write       │
   └──────┬──────────┬──────────────┬────────────┬────┘
          │          │              │             │
          ▼          ▼              ▼             ▼
       Belief      Event         Skill         World
       Memory      Memory        Memory        Model
          │          │              │             │
          └──────────┴──────────────┴─────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │  Unified Memory  │   ← PostgreSQL + pgvector
                   │     Store        │     JSONB content + vector(1536)
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Vector Retrieval │   ← Semantic similarity + recency
                   │     Engine       │     + confidence + usage ranking
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  Decay Scheduler │   ← APScheduler cron jobs
                   │  (Background)    │     Daily decay · Weekly pruning
                   └──────────────────┘
```

## Project Structure

```
clara/
├── agent.py              # ClaraMemory — top-level async façade
├── db/
│   ├── models.py         # SQLAlchemy models (Memory, enums, Base)
│   └── migrations/       # Alembic migrations
├── extraction/
│   └── extractor.py      # LLM-based fact extraction
├── memory/
│   └── belief.py         # Belief memory operations & Bayesian updates
├── retrieval/
│   ├── embeddings.py     # Embedding backends (OpenAI / local)
│   └── engine.py         # Ranked vector retrieval engine
├── update/
│   └── engine.py         # Memory update pipeline & conflict resolution
└── scheduler/
    └── decay.py          # DecayScheduler (confidence decay + pruning)
```

## Installation

### 1. PostgreSQL + pgvector

CLARA requires PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.

```bash
# Ubuntu / Debian
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-16-pgvector   # match your PG version

# macOS (Homebrew)
brew install postgresql@16
brew install pgvector

# Docker (fastest)
docker run -d --name clara-pg \
  -e POSTGRES_USER=clara \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=clara \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Install CLARA

```bash
pip install clara-memory
```

Or from source:

```bash
git clone https://github.com/itsamiitt/CLARA.git
cd CLARA
pip install -e ".[dev]"
```

### 3. Environment Variables

```bash
# Required
export DATABASE_URL="postgresql+asyncpg://clara:secret@localhost:5432/clara"

# Embedding backend (pick one)
export OPENAI_API_KEY="sk-..."          # for embedding_backend="openai"
# — or —
# No key needed for embedding_backend="local" (uses sentence-transformers)

# LLM provider for fact extraction (pick one)
export OPENAI_API_KEY="sk-..."          # for llm_provider="openai"
export ANTHROPIC_API_KEY="sk-ant-..."   # for llm_provider="anthropic"
```

### 4. Verify

```python
import asyncio
from clara.agent import ClaraMemory

async def check():
    agent = await ClaraMemory.create(
        db_url="postgresql+asyncpg://clara:secret@localhost:5432/clara",
        start_scheduler=False,
    )
    print("✓ CLARA is ready")
    await agent.close()

asyncio.run(check())
```

## Core Concepts

| Concept | Description |
|---|---|
| **Belief Memory** | Atomic subject–relation–object triples with Bayesian confidence. Versioned, never deleted. |
| **Event Memory** | Time-indexed occurrences (started, completed, failed). Immutable once created. |
| **Skill Memory** | Procedural knowledge with trigger conditions, steps, and a success/failure feedback loop. |
| **World Model** | Live, mutable snapshot of environment state. The only memory type that allows in-place mutation. |
| **Conflict Resolution** | Temporal ordering → source-weight arbitration → supersession. Old beliefs retained for audit. |
| **Confidence Decay** | `confidence_t = confidence_0 × e^(−decay_rate × Δt)` — stale facts fade, reinforced facts strengthen. |
| **Pruning** | Weekly scheduler archives beliefs < 0.15 confidence, deprecates unused skills, compacts world model. |

## Documentation

See the codebase for full details and examples.

## License

CLARA is released under the [MIT License](LICENSE).

