<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/v/clara-memory?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/pyversions/clara-memory?label=Python" alt="Python 3.10+"></a>
  <a href="https://github.com/itsamiitt/CLARA"><img src="https://img.shields.io/badge/tests-362%20passed-success" alt="Test status"></a>
</p>

<h1 align="center">CLARA</h1>
<p align="center"><b>Cognitive Living Architecture for Reliable Agents</b></p>

CLARA is a structured memory system for agents. It converts raw text into typed memories, stores them with provenance and embeddings, and retrieves them as ranked context.

## What CLARA Is

CLARA is designed for agent memory, not generic document storage.

Today it stores four memory types in one unified store:

- belief
- event
- skill
- world_model

Each stored memory row contains:

- structured JSON content
- an embedding for retrieval
- confidence
- status
- decay rate
- timestamps
- metadata and provenance

The public facade is:

```python
await agent.remember(text)
await agent.recall(query, top_k=8)
await agent.context_for(query, top_k=8)
await agent.interact(message, user_id="alice")
```

## What It Does Well

- extracts facts from raw text
- classifies memories into belief, event, skill, and world model
- stores embeddings with each memory
- retrieves memories by semantic similarity plus confidence, recency, and usage
- reinforces or supersedes beliefs instead of blindly overwriting them
- supports local SQLite testing and PostgreSQL + pgvector production use

## Current Scope

Implemented well:

- belief memory lifecycle
- event, skill, and world-model typed storage
- retrieval and context formatting
- reasoning loop with memory-grounded response generation
- reflection-driven insight synthesis from recent memories
- decay and pruning
- optional retrieval-result caching with in-memory or Redis backends
- FastAPI service layer for interaction and memory queries
- admin/reporting endpoints for stats, conflicts, decay, health, and skill ranking
- tenant-scoped retrieval, updates, and reflection runs
- SQLite test/dev fallback

Not implemented yet:

- first-class document storage
- chunk store for long documents
- rich procedural skill graphs
- true mutable world-model merge logic

If you need document ingestion, the right design is to add separate `documents` and `document_chunks` tables and let memories reference them for provenance.

## Quickstart

```python
import asyncio
from clara.agent import ClaraMemory


async def main():
    agent = await ClaraMemory.create(
        db_url="postgresql+asyncpg://user:pass@localhost:5432/clara",
        embedding_backend="openai",   # or "local"
        llm_provider="openai",        # or "anthropic"
    )

    await agent.remember("I use Rust for systems work.")

    result = await agent.recall("What language does the user use?", top_k=5)
    print(result.total)

    ctx = await agent.context_for("Help the user deploy a Rust service.", top_k=5)
    print(ctx)

    reply = await agent.interact(
        "What language does the user use for systems work?",
        user_id="alice",
    )
    print(reply["response"])

    await agent.close()


asyncio.run(main())
```

## Storage Model

CLARA stores extracted memory records in the `memories` table.

Key fields:

- `memory_type`
- `content`
- `embedding`
- `confidence`
- `status`
- `decay_rate`
- `created_at`
- `updated_at`
- `metadata`

Raw source text is kept as provenance in metadata:

- beliefs keep an evidence trail
- other memory types keep `raw_text` and `source_type`

This means CLARA currently stores distilled memory, not full source documents.

## Memory Flow

1. `remember(text)` runs fact extraction.
2. Each extracted fact is classified into a memory type.
3. The fact is embedded.
4. Similar memories are searched.
5. CLARA creates, reinforces, supersedes, or retains both.
6. `recall()` ranks results across memory types.
7. `context_for()` renders the retrieval result for prompt injection.
8. `interact()` runs retrieval, builds memory context, calls the reasoning model, and stores any facts extracted from the response.

## API

CLARA also exposes a FastAPI service layer.

Core routes:

- `POST /interact`
- `POST /memory/learn`
- `GET /memory/search`
- `GET /memory/timeline`
- `GET /memory/beliefs`
- `GET /memory/{memory_id}`

Run locally:

```bash
uvicorn clara.main:app --reload
```

## Production and Local Use

### Production

Use PostgreSQL with pgvector.

Example database URL:

```bash
export DATABASE_URL="postgresql+asyncpg://clara:secret@localhost:5432/clara"
```

### Local development and tests

SQLite is supported for local development and automated tests.

On SQLite:

- schema creation works
- retrieval uses a Python cosine-similarity fallback instead of pgvector SQL

PostgreSQL + pgvector is still the intended deployment target.

## Installation

### From source

```bash
git clone https://github.com/itsamiitt/CLARA.git
cd CLARA
pip install -e ".[dev]"
```

### Optional runtime choices

- `embedding_backend="openai"` requires `OPENAI_API_KEY`
- `embedding_backend="local"` requires `sentence-transformers`
- `llm_provider="openai"` requires `OPENAI_API_KEY`
- `llm_provider="anthropic"` requires `ANTHROPIC_API_KEY`

For local SQLite testing, install `aiosqlite`.

## Repository Layout

```text
clara/
  agent.py
  db/
    models.py
    migrations/
  extraction/
    extractor.py
  memory/
    belief.py
  retrieval/
    embeddings.py
    engine.py
  scheduler/
    decay.py
  update/
    engine.py
tests/
README.md
pyproject.toml
```

## GitHub Hygiene

### Keep on GitHub

- source code
- tests
- migrations
- `pyproject.toml`
- `README.md`
- sanitized docs and architecture notes

### Keep out of GitHub

- `.env` files
- credentials and API keys
- local databases
- generated test output
- logs, caches, temp files
- private user data
- raw local notes that are not meant for distribution

Important:

- adding a file to `.gitignore` stops new untracked copies from being added
- if a file is already tracked by Git, it must also be removed from the index separately if you want it gone from future commits

## Verification

Current verified state in this repo:

- full suite passes: `371 passed`
- facade-level smoke tests pass for belief, event, skill, and world-model storage and retrieval

Run locally with:

```bash
pytest --tb=short -q
```

## License

MIT
