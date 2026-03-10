# CLARA

CLARA is a structured memory layer for agents. It takes raw text, extracts facts, stores them as typed memories, and retrieves them later as ranked context.

The public API is small:

- `remember(text)`
- `recall(query, top_k=...)`
- `context_for(query, top_k=...)`

## What CLARA Stores

CLARA does not currently store full documents as first-class documents.

It stores extracted memory records in a single `memories` table:

- `belief`
- `event`
- `skill`
- `world_model`

Each row contains:

- structured `content` JSON
- an `embedding`
- `confidence`
- `status`
- `decay_rate`
- timestamps
- auxiliary `metadata`

Raw source text is kept only as provenance in metadata:

- beliefs store an evidence trail
- other memory types store `raw_text` and `source_type`

If you need document storage, chunking, or document-level retrieval, that should be added as a separate document layer rather than forcing everything into the memory table.

## Current Implementation Status

What is implemented today:

- belief memory has the richest behavior: reinforcement, confidence updates, superseding, evidence trail
- event, skill, and world-model memories are stored and retrieved correctly as typed rows
- semantic retrieval works with PostgreSQL + pgvector and now also has a SQLite fallback for local testing
- confidence decay and pruning are implemented

What is not implemented yet:

- first-class document and chunk storage
- procedural skill objects with step graphs or execution traces
- true mutable world-model merge semantics

## Quickstart

### Production path

Use PostgreSQL with pgvector.

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
    result = await agent.recall("What language does the user use?")
    ctx = await agent.context_for("Help the user deploy a Rust service.")

    print(result.total)
    print(ctx)

    await agent.close()

asyncio.run(main())
```

### Local development and tests

SQLite is supported for local validation and tests:

- schema creation works on SQLite
- retrieval falls back to Python cosine similarity instead of pgvector SQL

This is useful for tests and local smoke checks. PostgreSQL + pgvector is still the intended production path.

## Installation

### From source

```bash
git clone https://github.com/itsamiitt/CLARA.git
cd CLARA
pip install -e ".[dev]"
```

### Runtime dependencies

Core package:

- `sqlalchemy`
- `asyncpg`
- `pgvector`
- `apscheduler`
- `openai`

Optional:

- `sentence-transformers` for `embedding_backend="local"`
- `anthropic` for `llm_provider="anthropic"`
- `aiosqlite` for local SQLite tests/dev

## Environment

For PostgreSQL:

```bash
export DATABASE_URL="postgresql+asyncpg://clara:secret@localhost:5432/clara"
```

For OpenAI embeddings or extraction:

```bash
export OPENAI_API_KEY="sk-..."
```

For Anthropic extraction:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## How Memory Flows Work

1. `remember(text)` calls the extractor and turns text into `ExtractedFact` objects.
2. Each fact is classified into `belief`, `event`, `skill`, or `world_model`.
3. The fact is embedded.
4. Similar memories are searched.
5. CLARA either creates, reinforces, supersedes, or retains both.
6. `recall()` ranks results by similarity, confidence, recency, and usage.
7. `context_for()` formats the ranked memories into an LLM-friendly context block.

## Storage Model

Main table:

- `memories`

Important fields:

- `memory_type`
- `content`
- `embedding`
- `confidence`
- `status`
- `decay_rate`
- `created_at`
- `updated_at`
- `metadata`

Notes:

- memories are versioned through status changes, not hard deletes
- beliefs keep evidence history
- events do not decay
- skills and world-model entries currently use the generic typed-memory pipeline

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
pyproject.toml
README.md
```

## What Should Be On GitHub

Commit these:

- source code under `clara/`
- tests under `tests/`
- `pyproject.toml`
- migrations
- documentation like `README.md`, architecture notes, implementation plans
- small deterministic fixtures or redacted example data

Keep the repo clean:

- commit code, schema, tests, docs, and reproducible examples
- do not commit local runtime state

## What Should Not Be On GitHub

Do not commit:

- `.env` files
- API keys, tokens, credentials, certificates
- local database files and dumps
- user conversation logs or private raw documents
- generated embeddings or memory snapshots containing real user data
- virtual environments
- coverage output, test caches, editor folders, temp files

If you need example memory data in the repo, use sanitized fixtures only.

## Testing

Current verification in this repo:

- full test suite passes: `207 passed`
- facade-level smoke tests cover storing and retrieving belief, event, skill, and world-model memories

Run the suite with:

```bash
pytest -q
```

## Practical Notes

- PostgreSQL + pgvector is the real deployment target.
- SQLite support exists to make tests and local validation easy.
- CLARA is a memory system, not yet a full document store.
- Belief memory is the most mature subsystem right now.

## License

MIT
