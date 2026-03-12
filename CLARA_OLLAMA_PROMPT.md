## Context

You are an expert Python engineer specializing in AI systems and local LLM integration. I have a Python memory library called **CLARA** (Cognitive Living Architecture for Reliable Agents). It currently supports two LLM providers — `openai` and `anthropic` — and two embedding backends — `openai` and `local` (sentence-transformers). I need you to add **Ollama** as a third option for both the LLM provider and the embedding backend, exactly the way mem0 implements it.

The result should be that CLARA works completely locally with zero API keys — just like mem0 does with Ollama.

---

## How mem0 Does It (Reference Implementation)

mem0's Ollama LLM adapter does the following:

```python
from ollama import Client

client = Client(host="http://localhost:11434")  # default Ollama address

# For LLM calls — uses native Ollama chat API
response = client.chat(
    model="llama3.2",
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": query}],
    options={"temperature": 0.2, "num_predict": 2048},
    format="json",   # when JSON output is required
)
text = response.message.content

# For embeddings — uses native Ollama embeddings API
response = client.embeddings(model="nomic-embed-text", prompt=text)
vector = response["embedding"]
```

mem0 also auto-pulls the model if it is not already downloaded:
```python
local_models = client.list()["models"]
if not any(m.get("model") == model_name for m in local_models):
    client.pull(model_name)
```

CLARA must follow this exact same pattern. The `ollama` Python package communicates with the Ollama server running locally at `http://localhost:11434` — no API key, no internet connection needed after the model is downloaded.

---

## Where CLARA Calls LLMs Today

There are exactly **three files** that make LLM calls. All three need an Ollama path added:

### 1. `clara/extraction/extractor.py`

Called by `agent.remember()`. Parses raw text into structured `ExtractedFact` objects (subject / relation / object / confidence / domain). Currently calls OpenAI or Anthropic with `response_format={"type": "json_object"}` to force JSON output.

Key function signature:
```python
def _call_openai(text: str, model: str) -> str: ...
def _call_anthropic(text: str, model: str) -> str: ...
# Need to add:
def _call_ollama(text: str, model: str, base_url: str) -> str: ...
```

The system prompt (`SYSTEM_PROMPT`) is already defined in the file and must be passed unchanged to Ollama.

### 2. `clara/reasoning/engine.py`

Called by `agent.interact()`. Generates responses using retrieved memory context. Currently calls OpenAI or Anthropic with a system prompt + user query. Returns plain text (not JSON).

Key method signatures:
```python
def _call_openai(self, system_prompt: str, query: str) -> str: ...
def _call_anthropic(self, system_prompt: str, query: str) -> str: ...
# Need to add:
def _call_ollama(self, system_prompt: str, query: str) -> str: ...
```

### 3. `clara/reflection/pipeline.py`

Called by the background scheduler. Detects patterns across memories and generates meta-insights. Currently calls OpenAI or Anthropic. Returns plain text.

Key method signatures:
```python
def _call_openai(self, prompt: str, pattern: PatternCandidate) -> str: ...
def _call_anthropic(self, prompt: str, pattern: PatternCandidate) -> str: ...
# Need to add:
def _call_ollama(self, prompt: str, pattern: PatternCandidate) -> str: ...
```

---

## Where CLARA Calls Embeddings Today

One file handles all embeddings:

### `clara/retrieval/embeddings.py`

Two backends exist:
- `_OpenAIBackend` — calls `openai.OpenAI().embeddings.create()`, produces 1536-dim vectors
- `_LocalBackend` — uses `sentence-transformers/all-MiniLM-L6-v2`, produces 384-dim vectors

Need to add:
- `_OllamaBackend` — calls `ollama.Client().embeddings()`, produces variable-dim vectors depending on model

**Critical dimension issue:** `nomic-embed-text` produces 768-dim vectors. `mxbai-embed-large` produces 1024-dim. CLARA's database schema uses `VECTOR_DIMENSIONS = 1536`. The existing `normalize_embedding_dimensions()` function already handles padding/truncating — it must be used for the Ollama embedding backend too, just like the local backend uses it.

---

## Target: Zero-Setup Usage After This Change

```python
# Install
# pip install "clara-memory[ollama]"
# ollama pull llama3.2
# ollama pull nomic-embed-text

agent = await ClaraMemory.create(
    db_url="sqlite+aiosqlite:///clara.db",
    embedding_backend="ollama",    # nomic-embed-text, runs locally
    llm_provider="ollama",         # llama3.2, runs locally
)

# Zero API keys. Zero internet after model download. Full CLARA semantics.
await agent.remember("I prefer Python for data work.")
results = await agent.recall("what does the user prefer?")
await agent.close()
```

Or with custom models and Ollama host:
```python
agent = await ClaraMemory.create(
    db_url="sqlite+aiosqlite:///clara.db",
    embedding_backend="ollama",
    llm_provider="ollama",
    ollama_base_url="http://localhost:11434",   # default, can be overridden
    ollama_llm_model="mistral",                 # default: llama3.2
    ollama_embed_model="nomic-embed-text",      # default: nomic-embed-text
)
```

---

## Files to Modify

Only touch these files. Everything else stays identical.

### 1. `pyproject.toml`

Add an `ollama` optional extras group:

```toml
[project.optional-dependencies]
ollama = [
    "ollama>=0.3",    # the official Ollama Python client
]
```

Also update the `all` extras to include `ollama`:
```toml
all = [
    "clara-memory[api,cache,local,anthropic,ollama,dev]",
]
```

Do **not** add `ollama` to core dependencies — it must remain optional. If `llm_provider="ollama"` is used but `ollama` is not installed, raise a clear `ImportError` with installation instructions, exactly like the existing OpenAI and Anthropic guards do.

### 2. `clara/retrieval/embeddings.py`

Add `_OllamaBackend` class alongside the existing `_OpenAIBackend` and `_LocalBackend`:

```python
try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib = None


class _OllamaBackend(_EmbeddingBackend):
    """Wraps Ollama's local embedding API. Zero API key required.

    Default model: nomic-embed-text (768 dims, auto-pulled if missing).
    Vectors are normalized to VECTOR_DIMENSIONS via normalize_embedding_dimensions().
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        if _ollama_lib is None:
            raise ImportError(
                "The 'ollama' package is required for the Ollama embedding backend. "
                "Install it with: pip install 'clara-memory[ollama]'"
            )
        self._model = model or self.DEFAULT_MODEL
        self._client = _ollama_lib.Client(host=base_url or self.DEFAULT_BASE_URL)
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Pull the model from Ollama if not already downloaded."""
        local = self._client.list().get("models", [])
        names = {m.get("name", "") for m in local} | {m.get("model", "") for m in local}
        if self._model not in names:
            self._client.pull(self._model)

    @property
    def dimensions(self) -> int:
        return VECTOR_DIMENSIONS   # always normalized to 1536

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings(model=self._model, prompt=text)
        raw = response["embedding"]
        return normalize_embedding_dimensions(raw, target_dimensions=VECTOR_DIMENSIONS)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
```

Update `build_embedding_engine()` (or wherever the backend is selected) to handle `"ollama"`:

```python
def build_embedding_engine(
    backend: str = "openai",
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> EmbeddingEngine:
    match backend.strip().lower():
        case "openai":
            return EmbeddingEngine(_OpenAIBackend())
        case "local":
            return EmbeddingEngine(_LocalBackend())
        case "ollama":
            return EmbeddingEngine(_OllamaBackend(model=ollama_model, base_url=ollama_base_url))
        case _:
            raise ValueError(
                f"Unknown embedding backend {backend!r}. "
                "Choose 'openai', 'local', or 'ollama'."
            )
```

### 3. `clara/extraction/extractor.py`

Add guarded import at the top alongside the existing OpenAI and Anthropic imports:

```python
try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib: Any = None
```

Add environment variable constants:
```python
ENV_OLLAMA_BASE_URL = "CLARA_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "CLARA_OLLAMA_MODEL"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
```

Add `_call_ollama` function:
```python
def _call_ollama(text: str, model: str, base_url: str) -> str:
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama extraction provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )
    client = _ollama_lib.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input:\n{text}\n\nRespond with valid JSON only."},
        ],
        options={"temperature": 0.1, "num_predict": 2048},
        format="json",   # Ollama native JSON mode — equivalent to response_format=json_object
    )
    return response.message.content or ""
```

Update `FactExtractor.__init__` to accept and store `ollama_base_url` and `ollama_model`:
```python
def __init__(
    self,
    provider: str | None = None,
    model: str | None = None,
    ollama_base_url: str | None = None,
) -> None:
    self._provider = (provider or os.environ.get(ENV_LLM_PROVIDER, DEFAULT_PROVIDER)).lower()
    self._ollama_base_url = (
        ollama_base_url
        or os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
    )
    if self._provider == "openai":
        self._model = model or os.environ.get(ENV_OPENAI_MODEL, DEFAULT_OPENAI_MODEL)
        self._call_fn = lambda text, m: _call_openai(text, m)
    elif self._provider == "anthropic":
        self._model = model or os.environ.get(ENV_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_MODEL)
        self._call_fn = lambda text, m: _call_anthropic(text, m)
    elif self._provider == "ollama":
        self._model = model or os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL)
        self._call_fn = lambda text, m: _call_ollama(text, m, self._ollama_base_url)
    else:
        raise ValueError(
            f"Unknown LLM provider {self._provider!r}. "
            "Choose 'openai', 'anthropic', or 'ollama'."
        )
```

### 4. `clara/reasoning/engine.py`

Add guarded import:
```python
try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib: Any = None
```

Add environment variable constants:
```python
ENV_OLLAMA_BASE_URL = "CLARA_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "CLARA_OLLAMA_MODEL"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
```

Add `_call_ollama` method to `ReasoningEngine`:
```python
def _call_ollama(self, system_prompt: str, query: str) -> str:
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama reasoning provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )
    base_url = os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
    client = _ollama_lib.Client(host=base_url)
    response = client.chat(
        model=self._model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.2, "num_predict": 2048},
    )
    return response.message.content or ""
```

Update `_model_name()` to return the Ollama model when provider is ollama:
```python
def _model_name(self) -> str:
    provider = self._llm_provider.strip().lower()
    if provider == "openai":
        return os.environ.get("CLARA_OPENAI_MODEL", "gpt-4o-mini")
    if provider == "anthropic":
        return os.environ.get("CLARA_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    if provider == "ollama":
        return os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL)
    return "gpt-4o-mini"
```

Update `_generate_response` to route to `_call_ollama`:
```python
async def _generate_response(self, system_prompt: str, query: str) -> str:
    provider = self._llm_provider.strip().lower()
    if provider == "openai":
        return await asyncio.get_event_loop().run_in_executor(
            None, self._call_openai, system_prompt, query
        )
    if provider == "anthropic":
        return await asyncio.get_event_loop().run_in_executor(
            None, self._call_anthropic, system_prompt, query
        )
    if provider == "ollama":
        return await asyncio.get_event_loop().run_in_executor(
            None, self._call_ollama, system_prompt, query
        )
    raise ValueError(f"Unknown reasoning provider {self._llm_provider!r}.")
```

Note: This also fixes **Bug #1** from the audit — all three providers now use `run_in_executor` instead of blocking the event loop directly.

### 5. `clara/reflection/pipeline.py`

Same pattern as `reasoning/engine.py`. Add:

```python
try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib: Any = None
```

Add `_call_ollama` method to `ReflectionEngine`:
```python
def _call_ollama(self, prompt: str, pattern: PatternCandidate) -> str:
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama reflection provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )
    base_url = os.environ.get("CLARA_OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("CLARA_OLLAMA_MODEL", "llama3.2")
    client = _ollama_lib.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_predict": 1024},
    )
    return response.message.content or ""
```

Update `_generate_insight` (or equivalent dispatch method) to route to `_call_ollama`:
```python
provider = self._llm_provider.strip().lower()
if provider == "openai":
    return await loop.run_in_executor(None, self._call_openai, prompt, pattern)
if provider == "anthropic":
    return await loop.run_in_executor(None, self._call_anthropic, prompt, pattern)
if provider == "ollama":
    return await loop.run_in_executor(None, self._call_ollama, prompt, pattern)
raise ValueError(f"Unknown reflection provider {self._llm_provider!r}.")
```

### 6. `clara/config.py`

Add Ollama configuration fields to `ClaraConfig`:

```python
@dataclass
class ClaraConfig:
    # ... existing fields ...

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
```

Update `from_env()` to read these from environment:
```python
@classmethod
def from_env(cls) -> "ClaraConfig":
    return cls(
        # ... existing fields ...
        ollama_base_url=_str("CLARA_OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_llm_model=_str("CLARA_OLLAMA_MODEL", "llama3.2"),
        ollama_embed_model=_str("CLARA_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )
```

### 7. `clara/agent.py`

Add `ollama_base_url`, `ollama_llm_model`, `ollama_embed_model` parameters to `ClaraMemory.create()`:

```python
@classmethod
async def create(
    cls,
    db_url: str,
    embedding_backend: str = "openai",
    llm_provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
    ollama_llm_model: str = "llama3.2",
    ollama_embed_model: str = "nomic-embed-text",
    # ... other existing params ...
) -> "ClaraMemory":
```

Pass `ollama_base_url` and `ollama_embed_model` through to `build_embedding_engine()`:
```python
embedding_engine = build_embedding_engine(
    backend=embedding_backend,
    ollama_base_url=ollama_base_url,
    ollama_model=ollama_embed_model,
)
```

Pass `ollama_base_url` and `ollama_llm_model` through to `FactExtractor`, `ReasoningEngine`, and `ReflectionEngine` constructors.

---

## Environment Variables Reference

After this change, the full set of Ollama-related env vars:

| Variable | Default | Purpose |
|---|---|---|
| `CLARA_LLM_PROVIDER` | `openai` | Set to `ollama` to use Ollama for all LLM calls |
| `CLARA_EMBEDDING_BACKEND` | `openai` | Set to `ollama` to use Ollama for embeddings |
| `CLARA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `CLARA_OLLAMA_MODEL` | `llama3.2` | Model for extraction, reasoning, reflection |
| `CLARA_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |

Usage via environment (no code change needed):
```bash
export CLARA_LLM_PROVIDER=ollama
export CLARA_EMBEDDING_BACKEND=ollama
export CLARA_OLLAMA_MODEL=mistral
export CLARA_OLLAMA_EMBED_MODEL=mxbai-embed-large

python my_agent.py
```

---

## Recommended Ollama Models

| Use case | Model | Command | Size |
|---|---|---|---|
| Extraction + reasoning (default) | `llama3.2` | `ollama pull llama3.2` | 2GB |
| Extraction + reasoning (better) | `mistral` | `ollama pull mistral` | 4GB |
| Extraction + reasoning (best local) | `llama3.1:8b` | `ollama pull llama3.1:8b` | 5GB |
| Embeddings (default) | `nomic-embed-text` | `ollama pull nomic-embed-text` | 274MB |
| Embeddings (higher quality) | `mxbai-embed-large` | `ollama pull mxbai-embed-large` | 670MB |

Minimum setup for zero-API-key CLARA:
```bash
ollama pull llama3.2        # 2GB download, one time
ollama pull nomic-embed-text # 274MB download, one time
```

---

## Constraints

- The `ollama` Python package must be an **optional dependency** — never imported at module level without a guard. If it is missing and `llm_provider="ollama"` is used, raise `ImportError` with the exact install command
- All Ollama calls are **synchronous** (`ollama.Client` has no async API). They must be wrapped in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the event loop — this also fixes Bug #1 from the audit for the existing OpenAI and Anthropic paths at the same time
- **Auto-pull behaviour**: if the specified model is not found in `client.list()`, call `client.pull(model)` automatically before the first use — exactly like mem0 does. This must happen in `_OllamaBackend.__init__` for embeddings and lazily (on first call) for the LLM provider
- The `format="json"` parameter must be passed to Ollama for extraction calls (where JSON output is required) but **not** for reasoning or reflection calls (where plain text is expected)
- Existing `openai` and `anthropic` providers must continue to work exactly as before — this is an additive change only
- All **378 existing passing tests** must still pass — the Ollama provider is additive and does not change any existing code paths
- The `_cosine_similarity` and all retrieval scoring logic stays completely unchanged

---

## New Tests to Add

Add `tests/test_ollama.py`:

```python
# tests/test_ollama.py
import pytest
from unittest.mock import MagicMock, patch


class TestOllamaEmbeddingBackend:

    def test_ollama_backend_raises_without_package(self):
        with patch("clara.retrieval.embeddings._ollama_lib", None):
            from clara.retrieval.embeddings import _OllamaBackend
            with pytest.raises(ImportError, match="ollama"):
                _OllamaBackend()

    def test_ollama_embed_normalizes_dimensions(self):
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"model": "nomic-embed-text"}]}
        mock_client.embeddings.return_value = {"embedding": [0.1] * 768}  # nomic produces 768

        with patch("clara.retrieval.embeddings._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            from clara.retrieval.embeddings import _OllamaBackend, VECTOR_DIMENSIONS
            backend = _OllamaBackend(model="nomic-embed-text")
            result = backend.embed("test text")

        assert len(result) == VECTOR_DIMENSIONS   # must be padded to 1536

    def test_ollama_embed_pulls_missing_model(self):
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}  # model not present

        with patch("clara.retrieval.embeddings._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            from clara.retrieval.embeddings import _OllamaBackend
            _OllamaBackend(model="nomic-embed-text")

        mock_client.pull.assert_called_once_with("nomic-embed-text")


class TestOllamaExtractor:

    def test_extractor_raises_without_package(self):
        with patch("clara.extraction.extractor._ollama_lib", None):
            from clara.extraction.extractor import FactExtractor
            with pytest.raises(ImportError, match="ollama"):
                extractor = FactExtractor(provider="ollama")
                extractor.extract("I use Python.")

    def test_extractor_calls_ollama_with_json_format(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(content='{"facts": [{"subject": "user", "relation": "uses", '
                                      '"object": "Python", "domain": null, '
                                      '"source_type": "user_direct", "confidence": 0.9, '
                                      '"is_negation": false}]}')
        )
        with patch("clara.extraction.extractor._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            from clara.extraction.extractor import FactExtractor
            extractor = FactExtractor(provider="ollama", model="llama3.2")
            facts = extractor.extract("I use Python for data work.")

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"   # must request JSON mode
        assert len(facts) == 1
        assert facts[0].subject == "user"
        assert facts[0].object == "Python"


class TestOllamaReasoningEngine:

    @pytest.mark.asyncio
    async def test_reasoning_engine_uses_run_in_executor(self, db_session, fake_embedder):
        """Ollama call must not block the event loop."""
        import asyncio
        from unittest.mock import AsyncMock
        from clara.reasoning.engine import ReasoningEngine

        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(content="Response using Ollama.")
        )

        with patch("clara.reasoning.engine._ollama_lib") as mock_lib:
            mock_lib.Client.return_value = mock_client
            engine = ReasoningEngine(
                db_session, fake_embedder, extractor=None,
                llm_provider="ollama",
            )
            response = await engine.respond("What do I use?", user_id="alice")

        assert "Ollama" in response.text or response.text != ""
        # Verify it did not block — mock_client.chat was called in executor, not directly
        mock_client.chat.assert_called_once()
```

---

## Do Not Touch

- `clara/memory/belief.py`
- `clara/memory/event.py`
- `clara/memory/skill.py`
- `clara/memory/world_model.py`
- `clara/retrieval/cache.py`
- `clara/retrieval/engine.py`
- `clara/scheduler/decay.py`
- `clara/interaction/layer.py`
- `clara/integrations/openclaw_bridge.py`
- `clara/api/` (all files)
- `clara/core/` (all files)
- `clara/db/models.py`
- All existing test files except adding the new `tests/test_ollama.py`

---

## Final State After Both Prompts Applied

```
PostgreSQL  ──→  removed entirely
pgvector    ──→  removed entirely
asyncpg     ──→  removed entirely

SQLite      ←─  relational metadata (zero config)
LanceDB     ←─  vector search (zero config)
Ollama      ←─  LLM + embeddings (zero config after model pull)

OpenAI      ←─  still works (optional)
Anthropic   ←─  still works (optional)
sentence-transformers ←─ still works (optional)
```

Zero API keys. Zero servers. Zero setup. Full CLARA memory semantics.
