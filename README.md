<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/v/clara-memory?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/clara-memory/"><img src="https://img.shields.io/pypi/pyversions/clara-memory?label=Python" alt="Python 3.10+"></a>
  <a href="https://github.com/itsamiitt/CLARA"><img src="https://img.shields.io/badge/tests-384%20passed-success" alt="Test status"></a>
</p>

<h1 align="center">CLARA</h1>
<p align="center"><b>Cognitive Living Architecture for Reliable Agents</b></p>

CLARA is a structured memory system for agents. It extracts facts from text, stores them as typed memories, retrieves them by relevance, and builds memory-grounded context for downstream reasoning.

## Overview

CLARA is built around four memory types:

- `belief`
- `event`
- `skill`
- `world_model`

The public interface is intentionally small:

```python
await agent.remember(text)
await agent.recall(query, top_k=8)
await agent.context_for(query, top_k=8)
await agent.interact(message, user_id="alice")
```

What CLARA does well:

- extracts structured facts from raw text
- classifies memories into typed records
- reinforces or supersedes beliefs instead of overwriting blindly
- retrieves memories with semantic similarity plus confidence, recency, and usage
- keeps relational metadata in SQLite and vector search in embedded LanceDB
- supports fully local operation with Ollama

## Easiest Installation

The simplest supported setup is:

- SQLite for relational metadata
- LanceDB for vector search
- Ollama for local LLM + embeddings

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install CLARA with Ollama support:

```bash
python -m pip install --upgrade pip
pip install "clara-memory[ollama]"
ollama pull llama3.2
ollama pull nomic-embed-text
```

Minimal zero-key startup:

```python
import asyncio
from clara.agent import ClaraMemory


async def main():
    agent = await ClaraMemory.create(
        embedding_backend="ollama",
        llm_provider="ollama",
    )

    await agent.remember("I prefer concise answers.")
    result = await agent.recall("what does the user prefer?")
    print(result.total)

    await agent.close()


asyncio.run(main())
```

By default this creates:

- `clara.db`
- `./clara_vectors`

No PostgreSQL server, pgvector extension, or API key is required for storage.

## Installation Options

Runtime packages:

- `pip install clara-memory`
  Uses SQLite + LanceDB storage with OpenAI available as the default LLM/embedding path.
- `pip install "clara-memory[ollama]"`
  Best local-first install. Recommended for zero-key usage.
- `pip install "clara-memory[local]"`
  Enables `sentence-transformers` embeddings for local embedding-only setups.
- `pip install "clara-memory[anthropic]"`
  Enables Anthropic as the LLM provider.
- `pip install "clara-memory[api]"`
  Adds FastAPI and Uvicorn for the HTTP service.
- `pip install -e ".[dev]"`
  Contributor install from source.

Source install:

```bash
git clone https://github.com/itsamiitt/CLARA.git
cd CLARA
pip install -e ".[dev]"
```

## Quickstart

Explicit local setup:

```python
import asyncio
from clara.agent import ClaraMemory


async def main():
    agent = await ClaraMemory.create(
        db_url="sqlite+aiosqlite:///clara.db",
        lance_path="./clara_vectors",
        embedding_backend="ollama",
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.2",
        ollama_embed_model="nomic-embed-text",
        start_scheduler=False,
    )

    await agent.remember("Alicia uses Rust for payments systems.")
    await agent.remember("Alicia knows Kubernetes.")

    result = await agent.recall("What does Alicia use for payments systems?", top_k=5)
    print(result.total)

    context = await agent.context_for("Summarize Alicia's stack.", top_k=5)
    print(context)

    reply = await agent.interact("What stack does Alicia use?", user_id="alice")
    print(reply["response"])

    await agent.close()


asyncio.run(main())
```

## Storage Architecture

CLARA now uses a two-layer embedded storage design:

- SQLite via `aiosqlite` for relational memory metadata
- LanceDB for vector indexing and search

SQLite stores:

- `memory_id`
- `user_id`
- `memory_type`
- `content`
- `confidence`
- `status`
- `decay_rate`
- `created_at`
- `updated_at`
- `metadata`

LanceDB stores:

- `memory_id`
- `vector`
- `user_id`
- `memory_type`
- `status`

This keeps setup simple while avoiding Python-side full-table scans for semantic retrieval.

## Memory Flow

1. `remember(text)` extracts structured facts.
2. Facts are classified into one of the supported memory types.
3. Each fact is embedded.
4. CLARA searches for similar active memories.
5. It creates, reinforces, supersedes, or retains both depending on conflict rules.
6. `recall()` ranks results by similarity, confidence, recency, and usage.
7. `context_for()` renders retrieval results into a prompt-ready memory block.
8. `interact()` retrieves context, generates a response, and stores facts extracted from that response.

## Provider Matrix

Embedding backends:

- `openai`
- `local`
- `ollama`

LLM providers:

- `openai`
- `anthropic`
- `ollama`

Requirements by mode:

- `embedding_backend="openai"` requires `OPENAI_API_KEY`
- `embedding_backend="local"` requires `sentence-transformers`
- `embedding_backend="ollama"` requires `pip install "clara-memory[ollama]"` and a local Ollama server
- `llm_provider="openai"` requires `OPENAI_API_KEY`
- `llm_provider="anthropic"` requires `ANTHROPIC_API_KEY`
- `llm_provider="ollama"` requires `pip install "clara-memory[ollama]"` and a local Ollama server

## Environment Variables

Core settings:

- `CLARA_DB_URL`
- `CLARA_LANCE_PATH`
- `CLARA_EMBEDDING_BACKEND`
- `CLARA_LLM_PROVIDER`
- `CLARA_START_SCHEDULER`
- `CLARA_CACHE_URL`

OpenAI settings:

- `OPENAI_API_KEY`
- `CLARA_OPENAI_MODEL`
- `CLARA_OPENAI_EMBEDDING_MODEL`

Anthropic settings:

- `ANTHROPIC_API_KEY`
- `CLARA_ANTHROPIC_MODEL`

Ollama settings:

- `CLARA_OLLAMA_BASE_URL`
- `CLARA_OLLAMA_MODEL`
- `CLARA_OLLAMA_EMBED_MODEL`

Example local-first shell config:

```bash
export CLARA_DB_URL="sqlite+aiosqlite:///clara.db"
export CLARA_LANCE_PATH="./clara_vectors"
export CLARA_EMBEDDING_BACKEND="ollama"
export CLARA_LLM_PROVIDER="ollama"
export CLARA_OLLAMA_MODEL="llama3.2"
export CLARA_OLLAMA_EMBED_MODEL="nomic-embed-text"
```

## API

Install the API extra:

```bash
pip install "clara-memory[api]"
```

Run locally:

```bash
uvicorn clara.main:app --reload
```

Core routes:

- `POST /interact`
- `POST /memory/learn`
- `GET /memory/search`
- `GET /memory/timeline`
- `GET /memory/beliefs`
- `GET /memory/{memory_id}`

Admin routes:

- stats
- conflicts
- decay
- health
- skill ranking

## Current Scope

Implemented:

- belief memory lifecycle
- event, skill, and world-model storage
- semantic retrieval and context formatting
- reasoning loop with memory-grounded responses
- reflection-driven insight synthesis
- decay and pruning
- optional in-memory or Redis retrieval caching
- tenant-scoped retrieval and updates
- FastAPI service layer

Not implemented yet:

- first-class document storage
- chunked document ingestion
- rich procedural skill graphs
- mutable world-model merge semantics

If you need document ingestion, the right next step is separate `documents` and `document_chunks` tables with memories referencing them for provenance.

## Troubleshooting

If `ollama` is selected and not installed:

- install with `pip install "clara-memory[ollama]"`

If Ollama models are missing:

- `ollama pull llama3.2`
- `ollama pull nomic-embed-text`

If you want an explicit database path:

- pass `db_url="sqlite+aiosqlite:///your-path.db"` to `ClaraMemory.create()`

If you want a separate vector directory:

- pass `lance_path="./your_vectors"`

## Verification

Current verified state in this working tree:

- default suite: `384 passed, 1 deselected`
- stress suite: previously verified during migration work

Run locally:

```bash
pytest --tb=short -q
```

## Repository Layout

```text
clara/
  agent.py
  config.py
  db/
    models.py
  extraction/
    extractor.py
  memory/
  reasoning/
  reflection/
  retrieval/
    embeddings.py
    engine.py
  scheduler/
  update/
tests/
scripts/
README.md
pyproject.toml
```

## License

MIT
