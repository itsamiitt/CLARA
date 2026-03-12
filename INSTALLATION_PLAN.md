# CLARA Installation Plan

## Goal

Make the first successful CLARA run require the fewest possible decisions.

Recommended default:

- SQLite
- LanceDB
- Ollama

This path avoids API keys, external database setup, and cloud dependencies.

## Default User Path

1. Create a virtual environment.
2. Install `clara-memory[ollama]`.
3. Pull the two default Ollama models.
4. Run CLARA with:
   - `db_url="sqlite+aiosqlite:///clara.db"`
   - `lance_path="./clara_vectors"`
   - `embedding_backend="ollama"`
   - `llm_provider="ollama"`

Copy-paste commands:

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

Install and prepare models:

```bash
python -m pip install --upgrade pip
pip install "clara-memory[ollama]"
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Recommended Documentation Order

1. Fastest local install
2. Minimal startup example
3. Optional cloud/API-key installs
4. Source install for contributors
5. Troubleshooting

## Packaging Guidance

Keep the simplest runtime path obvious:

- `clara-memory[ollama]` for easiest local install
- `clara-memory` for OpenAI-backed runtime
- `clara-memory[local]` for local embeddings only
- `clara-memory[anthropic]` for Anthropic-backed runtime
- `clara-memory[dev]` for contributors

## Verification Checklist

For the easiest install path, verify:

1. `pip install "clara-memory[ollama]"` succeeds in a clean venv.
2. `ollama pull llama3.2` succeeds.
3. `ollama pull nomic-embed-text` succeeds.
4. `ClaraMemory.create(...)` succeeds with SQLite + LanceDB + Ollama defaults.
5. `remember()` stores at least one fact.
6. `recall()` returns at least one result.
7. `close()` exits cleanly.

## Short-Term Recommendation

Use the Ollama install path as the primary README path and treat all other
install modes as advanced or optional.
