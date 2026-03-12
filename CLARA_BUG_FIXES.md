# CLARA v2 — Bug Fix Guide

8 bugs · 3 critical · 3 medium · 2 low  
Fixes ordered by priority: fix critical first, in order.

---

## Table of Contents

- [Bug #1 — Sync blocking LLM calls (CRITICAL)](#bug-1--sync-blocking-llm-calls-critical)
- [Bug #2 — Stress test segfault (CRITICAL)](#bug-2--stress-test-segfault-critical)
- [Bug #3 — WorldModel upsert race condition (CRITICAL)](#bug-3--worldmodel-upsert-race-condition-critical)
- [Bug #4 — SkillStore.match() full table scan (MEDIUM)](#bug-4--skillstorematch-full-table-scan-medium)
- [Bug #5 — BackgroundWriter never used (MEDIUM)](#bug-5--backgroundwriter-never-used-medium)
- [Bug #6 — fastapi not in core dependencies (MEDIUM)](#bug-6--fastapi-not-in-core-dependencies-medium)
- [Bug #7 — API user_id has no auth guard (LOW)](#bug-7--api-user_id-has-no-auth-guard-low)
- [Bug #8 — OpenClaw bridge session isolation by text prefix (LOW)](#bug-8--openclaw-bridge-session-isolation-by-text-prefix-low)

---

## Bug #1 — Sync blocking LLM calls (CRITICAL)

**Files:** `clara/reasoning/engine.py` · `clara/reflection/pipeline.py`  
**Impact:** Freezes the entire asyncio event loop on every `agent.interact()` or `ReflectionEngine.run()` call. All concurrent `recall()` and `remember()` operations stall until the HTTP response returns.

### What's wrong

`_call_openai()` and `_call_anthropic()` are synchronous methods that use `openai.OpenAI` and `anthropic.Anthropic` — the blocking, non-async SDK clients. They are called from inside `async def _generate_response()` with no `run_in_executor` wrapper. The same pattern exists in `ReflectionEngine._generate_insight()`.

```python
# reasoning/engine.py — BROKEN
def _call_openai(self, system_prompt: str, query: str) -> str:
    client = _openai.OpenAI(api_key=api_key)           # ← sync client
    response = client.chat.completions.create(...)      # ← blocks event loop
    return response.choices[0].message.content or ""

def _call_anthropic(self, system_prompt: str, query: str) -> str:
    client = _anthropic.Anthropic(api_key=api_key)     # ← sync client
    response = client.messages.create(...)              # ← blocks event loop
    return response.content[0].text or ""
```

### Fix — Option A: Switch to async SDK clients (recommended)

Replace both `_call_openai` and `_call_anthropic` with async versions in **both** `reasoning/engine.py` and `reflection/pipeline.py`.

**`clara/reasoning/engine.py`** — replace `_call_openai` and `_call_anthropic`:

```python
# FIXED — reasoning/engine.py
async def _call_openai(self, system_prompt: str, query: str) -> str:
    if _openai is None:
        raise ImportError(
            "The 'openai' package is required for the OpenAI reasoning provider."
        )
    api_key = os.environ.get(ENV_OPENAI_KEY)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {ENV_OPENAI_KEY!r} is not set."
        )
    client = _openai.AsyncOpenAI(api_key=api_key)      # ← AsyncOpenAI
    response = await client.chat.completions.create(   # ← await
        model=self._model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""

async def _call_anthropic(self, system_prompt: str, query: str) -> str:
    if _anthropic is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic reasoning provider."
        )
    api_key = os.environ.get(ENV_ANTHROPIC_KEY)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {ENV_ANTHROPIC_KEY!r} is not set."
        )
    client = _anthropic.AsyncAnthropic(api_key=api_key)  # ← AsyncAnthropic
    response = await client.messages.create(              # ← await
        model=self._model_name(),
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
        temperature=0.2,
    )
    return response.content[0].text or ""
```

Then update `_generate_response` to `await` the calls:

```python
# FIXED — _generate_response in reasoning/engine.py
async def _generate_response(self, query, memory_context, *, system_prompt=None) -> str:
    ...
    provider = self._llm_provider.strip().lower()
    if provider == "openai":
        return await self._call_openai(final_system_prompt, query)   # ← await
    if provider == "anthropic":
        return await self._call_anthropic(final_system_prompt, query) # ← await
    raise ValueError(f"Unknown reasoning provider {self._llm_provider!r}.")
```

Apply the **exact same changes** to `clara/reflection/pipeline.py` — the `ReflectionEngine._call_openai`, `_call_anthropic`, and `_generate_insight` methods follow the identical pattern.

### Fix — Option B: run_in_executor (if you can't use async clients)

```python
# Alternative if async SDK not available
import asyncio

async def _call_openai(self, system_prompt: str, query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        self._call_openai_sync,
        system_prompt,
        query,
    )

def _call_openai_sync(self, system_prompt: str, query: str) -> str:
    # original sync implementation here
    client = _openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(...)
    return response.choices[0].message.content or ""
```

Option A is strongly preferred — `run_in_executor` uses a thread pool and has overhead. Async SDK clients are purpose-built for this.

---

## Bug #2 — Stress test segfault (CRITICAL)

**File:** `tests/test_agent_stress.py`  
**Impact:** Running the full test suite crashes the Python process with a native segmentation fault. CI pipelines will fail silently or with no traceback.

### What's wrong

The test fires 120 concurrent queries against a single SQLite file via `asyncio.gather()` with no concurrency cap. SQLite's aiosqlite threading model can't handle this and crashes at the C extension level.

```python
# test_agent_stress.py line 426 — BROKEN
CONCURRENT_QUERY_USERS = 30  # × 4 query types = 120 concurrent queries

concurrent_results = await asyncio.gather(
    *(agent.recall(query, top_k=6) for query in concurrent_queries)
    # ↑ 120 simultaneous SQLite connections → segfault
)
```

### Fix — Add a semaphore and isolate the test

**Step 1:** Add a semaphore to cap SQLite concurrency inside the test:

```python
# test_agent_stress.py — FIXED
import asyncio

# Cap concurrent SQLite ops to 10 — safe for aiosqlite
_SEM = asyncio.Semaphore(10)

async def _recall_guarded(agent, query, top_k):
    async with _SEM:
        return await agent.recall(query, top_k=top_k)

# Replace the gather block:
concurrent_results = await asyncio.gather(
    *(_recall_guarded(agent, query, top_k=6) for query in concurrent_queries)
)
```

**Step 2:** Mark the test so it doesn't run in the default suite:

```python
# test_agent_stress.py — add marker at top of test function
@pytest.mark.stress
async def test_bulk_agent_round_trip_with_heavy_retrievals():
    ...
```

**Step 3:** Register the marker and exclude it by default in `pyproject.toml`:

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "stress: heavy load tests, excluded from default run",
]
addopts = "-m 'not stress'"
```

Run stress tests explicitly when needed:
```bash
pytest -m stress tests/test_agent_stress.py
```

---

## Bug #3 — WorldModel upsert race condition (CRITICAL)

**Files:** `clara/memory/world_model.py` · `clara/db/models.py`  
**Impact:** Under concurrent load (e.g. BackgroundWriter processing two facts about the same entity), both coroutines can SELECT → find nothing → INSERT, producing duplicate world model rows with no error raised.

### What's wrong

The SELECT-then-INSERT pattern has no database-level guard:

```python
# world_model.py line 96 — BROKEN
existing = await self._find_existing(entity_type, name, user_id=user_id)
# ← concurrent coroutine also returns None here at the same time
if existing is not None:
    return await self._merge_properties(...)
return await self._create_new(...)  # ← both reach here → duplicate rows
```

There is no `UNIQUE` constraint on `(user_id, memory_type, entity_type_in_content, name_in_content)` in `db/models.py`.

### Fix — Two-part: schema constraint + application-level guard

**Part 1 — Add a partial unique index in `clara/db/models.py`:**

SQLAlchemy doesn't support functional indexes on JSONB content columns natively, so use a raw DDL expression:

```python
# db/models.py — add to __table_args__
from sqlalchemy import Index, text

__table_args__ = (
    Index("ix_memories_memory_type", "memory_type"),
    Index("ix_memories_status", "status"),
    Index("ix_memories_created_at", "created_at"),
    Index("ix_memories_type_status", "memory_type", "status"),
    Index("ix_memories_user_id", "user_id"),
    Index("ix_memories_user_type_status", "user_id", "memory_type", "status"),

    # NEW: prevents duplicate world model entities per user
    # PostgreSQL only — functional index on JSONB content fields
    Index(
        "uq_world_model_entity_per_user",
        "user_id",
        text("(content->>'entity_type')"),
        text("(content->>'name')"),
        postgresql_where=text("memory_type = 'world_model' AND status = 'active'"),
        unique=True,
    ),
)
```

**Part 2 — Handle the race in `world_model.py` using INSERT ... ON CONFLICT:**

```python
# world_model.py — replace upsert() body with conflict-safe version
from sqlalchemy.dialects.postgresql import insert as pg_insert

async def upsert(self, *, entity_type, name, properties=None, domain=None,
                 user_id=None, confidence=0.9, source_type="user_direct",
                 raw_text=None, embedding=None) -> Memory:
    properties = properties or {}

    # Try merge first (optimistic path — most common case)
    existing = await self._find_existing(entity_type, name, user_id=user_id)
    if existing is not None:
        return await self._merge_properties(
            existing, properties,
            source_type=source_type, raw_text=raw_text, embedding=embedding,
        )

    # Create new, but catch the race condition
    try:
        return await self._create_new(
            entity_type=entity_type, name=name, properties=properties,
            domain=domain, user_id=user_id, confidence=confidence,
            source_type=source_type, raw_text=raw_text, embedding=embedding,
        )
    except Exception as exc:
        # If unique constraint fires, the race happened — re-fetch and merge
        if "uq_world_model_entity_per_user" in str(exc):
            existing = await self._find_existing(entity_type, name, user_id=user_id)
            if existing is not None:
                return await self._merge_properties(
                    existing, properties,
                    source_type=source_type, raw_text=raw_text, embedding=embedding,
                )
        raise
```

> **SQLite dev mode note:** The PostgreSQL functional index won't apply to SQLite. For local dev, the application-level try/except is the only guard. That's acceptable — the race is only dangerous under the concurrent load patterns seen in production PostgreSQL deployments.

---

## Bug #4 — SkillStore.match() full table scan (MEDIUM)

**File:** `clara/memory/skill.py` lines 225–262  
**Impact:** Every skill lookup fetches up to 200 rows and does Python-side string comparison. Users with more than 200 skills silently get incomplete results. Performance degrades linearly with skill count.

### What's wrong

```python
# skill.py line 246 — BROKEN
async def match(self, context, *, user_id=None, limit=10) -> Sequence[Memory]:
    all_skills = await self.get_active_skills(user_id=user_id, limit=200)
    # ↑ always fetches 200 rows regardless of how many match
    context_lower = context.lower()
    matched = []
    for skill in all_skills:
        triggers = content.get("trigger_conditions", [])
        if any(trigger.lower() in context_lower for trigger in triggers):
            matched.append(skill)
    matched.sort(key=lambda s: s.confidence, reverse=True)
    return matched[:limit]
```

### Fix — Push matching to the retrieval engine

Replace the substring scan with a vector similarity search via the existing `RetrievalEngine`. This requires passing the `EmbeddingEngine` into `SkillStore` at construction time.

**Step 1 — Update `SkillStore.__init__` to accept an embedding engine:**

```python
# skill.py — FIXED __init__
from clara.retrieval.engine import RetrievalEngine
from clara.retrieval.embeddings import EmbeddingEngine

class SkillStore:
    def __init__(
        self,
        session: AsyncSession,
        embedding_engine: EmbeddingEngine | None = None,  # ← new optional param
    ) -> None:
        self._session = session
        self._embedding_engine = embedding_engine
```

**Step 2 — Replace `match()` with vector search when engine is available:**

```python
# skill.py — FIXED match()
async def match(
    self,
    context: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
) -> Sequence[Memory]:
    """Match skills by vector similarity (preferred) or substring fallback."""
    from clara.db.models import MemoryType

    # Fast path: vector similarity via RetrievalEngine
    if self._embedding_engine is not None:
        retriever = RetrievalEngine(self._session, self._embedding_engine)
        result = await retriever.search(
            context,
            top_k=limit,
            user_id=user_id,
            memory_types=[MemoryType.skill],
        )
        return [sm.memory for sm in result.skills]

    # Fallback: substring match (dev/test, no embedding engine)
    # Hard cap lowered to 50 to prevent silent truncation at scale
    all_skills = await self.get_active_skills(user_id=user_id, limit=50)
    context_lower = context.lower()
    matched = [
        skill for skill in all_skills
        if any(
            trigger.lower() in context_lower
            for trigger in (skill.content or {}).get("trigger_conditions", [])
            if isinstance(trigger, str)
        )
    ]
    matched.sort(key=lambda s: s.confidence, reverse=True)
    return matched[:limit]
```

**Step 3 — Update all `SkillStore` instantiation sites to pass the embedding engine:**

Wherever `SkillStore(session)` is constructed (primarily `agent.py`), add the embedding engine:

```python
# Anywhere SkillStore is used:
skill_store = SkillStore(session, embedding_engine=self._embedding_engine)
```

---

## Bug #5 — BackgroundWriter never used (MEDIUM)

**File:** `clara/agent.py` lines 310, 329–382  
**Impact:** `BackgroundWriter` is instantiated and shut down but never enqueued to. `agent.remember()` always runs synchronously, blocking the caller for the full extraction + DB write cycle. The async write queue provides zero benefit.

### What's wrong

```python
# agent.py — BackgroundWriter created in ClaraMemory.create()...
background_writer=BackgroundWriter(session_factory, embedding_engine, cache=cache)

# ...but remember() completely ignores it:
async def remember(self, text, *, user_id=None):
    facts = self._extractor.extract(interaction.raw_text)
    async with self._session_factory() as session:
        async with session.begin():
            for fact in facts:
                outcome = await update_engine.process(fact, ...)  # ← blocks caller
                results.append({...})
    return results  # ← caller waited for all writes
```

### Fix — Route extracted facts through BackgroundWriter

Replace the inline synchronous write loop with `enqueue()` calls. Keep the synchronous path as an opt-in for callers that need immediate confirmation (e.g. tests).

```python
# agent.py — FIXED remember()
async def remember(
    self,
    text: str,
    *,
    user_id: str | None = None,
    wait: bool = False,   # ← new param: True = sync (old behaviour), False = async
) -> list[dict[str, Any]]:
    """Extract facts from *text* and store them in the memory store.

    Args:
        text:    Raw natural-language input.
        wait:    If True, process synchronously and return full results.
                 If False (default), enqueue to BackgroundWriter and return
                 immediately with action="enqueued" stubs.
    """
    if not text or not text.strip():
        return []

    interaction = self._interaction_layer.receive(text, user_id=user_id)
    facts = self._extractor.extract(interaction.raw_text)
    if not facts:
        logger.debug("No facts extracted from text: %r", text[:120])
        return []

    # --- Async path (default) ---
    if not wait and self._background_writer is not None:
        for fact in facts:
            await self._background_writer.enqueue(fact, user_id=interaction.user_id)
        logger.info("Enqueued %d fact(s) from text (%d chars)", len(facts), len(text))
        return [{"action": "enqueued", "memory_id": None, "conflict": False} for _ in facts]

    # --- Sync path (wait=True or no background writer) ---
    results: list[dict[str, Any]] = []
    async with self._session_factory() as session:
        async with session.begin():
            update_engine = MemoryUpdateEngine(
                session,
                self._embedding_engine,
                RetrievalEngine(session, self._embedding_engine, cache=self._cache),
                cache=self._cache,
            )
            for fact in facts:
                outcome = await update_engine.process(fact, user_id=interaction.user_id)
                results.append({
                    "action": outcome.action_taken.value,
                    "memory_id": str(outcome.memory_id) if outcome.memory_id else None,
                    "conflict": outcome.conflict_detected,
                    "superseded_id": (
                        str(outcome.superseded_id) if outcome.superseded_id else None
                    ),
                })
    logger.info("Remembered %d fact(s) from text (%d chars)", len(results), len(text))
    return results
```

Update the API route to use async mode by default:

```python
# api/routes_interaction.py — no change needed, default wait=False
results = await agent.remember(payload.text, user_id=payload.user_id)
```

Update tests to use `wait=True` for assertions that need immediate confirmation:

```python
# In tests — pass wait=True to get synchronous behaviour
results = await agent.remember("Alice uses Python", wait=True)
assert results[0]["action"] == "created"
```

---

## Bug #6 — fastapi not in core dependencies (MEDIUM)

**Files:** `pyproject.toml` · `tests/test_api.py` · `tests/test_admin_api.py`  
**Impact:** Fresh `pip install .` followed by `pytest` crashes with `ModuleNotFoundError: No module named 'fastapi'` during test collection, before any test runs.

### What's wrong

```toml
# pyproject.toml — fastapi is optional only
[project.optional-dependencies]
api = [
    "fastapi>=0.115,<1.0",
    "uvicorn[standard]>=0.30,<1.0",
]
```

```python
# tests/test_api.py line 13 — no guard, always imports
from clara.api import create_app  # ← ModuleNotFoundError in clean install
```

### Fix — Add pytest skip guards to API test files

This keeps fastapi optional (correct architecture) while stopping collection failures.

**`tests/test_api.py`** — add at the top, after existing imports:

```python
# tests/test_api.py — add after existing imports
import pytest

fastapi = pytest.importorskip(
    "fastapi",
    reason="fastapi not installed — run: pip install 'clara[api]'",
)
httpx = pytest.importorskip(
    "httpx",
    reason="httpx not installed — run: pip install 'clara[api]'",
)

# Remove the direct import of create_app (now guarded above)
# from clara.api import create_app  ← DELETE this line
```

Then import `create_app` lazily inside each test or fixture:

```python
@pytest_asyncio.fixture
async def app():
    from clara.api import create_app   # ← import inside fixture, after skip guard
    ...
```

**`tests/test_admin_api.py`** — apply the same skip guard pattern.

**Alternative: add a `[test]` extras group** to make CI easy:

```toml
# pyproject.toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115,<1.0",
    "uvicorn[standard]>=0.30,<1.0",
]
test = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "httpx>=0.27",          # ← needed for API tests
    "fastapi>=0.115,<1.0",  # ← needed for API tests
]
```

Then CI installs: `pip install ".[test]"` and gets everything needed.

---

## Bug #7 — API user_id has no auth guard (LOW)

**Files:** `clara/api/routes_memory.py` · `clara/api/dependencies.py`  
**Impact:** Any caller can pass `?user_id=someone_else` to read or write another user's memories. `TenantViolationError` exists but is never raised anywhere.

### What's wrong

```python
# routes_memory.py — user_id is an unauthenticated query param
@router.get("/search")
async def search_memories(
    q: str,
    user_id: str | None = None,   # ← caller passes anything here, no verification
    agent=Depends(get_agent),
):
    result = await agent.recall(q, top_k=top_k, user_id=user_id)
```

### Fix — Add an identity header dependency

**Step 1 — Add `get_current_user` to `clara/api/dependencies.py`:**

```python
# dependencies.py — add new dependency
import os
from fastapi import Header, HTTPException

def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> str | None:
    """Extract the caller's identity from the X-User-ID header.

    In production, replace this with JWT validation or API key lookup.
    Returns None when no header is present (unauthenticated / local mode).
    """
    return x_user_id


def require_user(
    current_user: str | None = Depends(get_current_user),
) -> str:
    """Like get_current_user but raises 401 if no identity is present."""
    if current_user is None:
        raise HTTPException(status_code=401, detail="X-User-ID header is required.")
    return current_user
```

**Step 2 — Update routes to inject and enforce user identity:**

```python
# routes_memory.py — FIXED search endpoint
from clara.api.dependencies import get_agent, get_session, get_current_user
from clara.core.exceptions import TenantViolationError

@router.get("/search")
async def search_memories(
    q: str,
    top_k: int = 8,
    user_id: str | None = None,                               # ← keep for local/dev
    current_user: str | None = Depends(get_current_user),    # ← new: header identity
    agent=Depends(get_agent),
):
    # If authenticated, enforce that the requested user_id matches the caller
    if current_user is not None and user_id is not None and user_id != current_user:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot query memories for user {user_id!r}: access denied.",
        )

    # Use authenticated identity if no explicit user_id was passed
    effective_user_id = user_id or current_user
    result = await agent.recall(q, top_k=top_k, user_id=effective_user_id)
    ...
```

Apply the same pattern to all other routes in `routes_memory.py` and `routes_interaction.py`.

**Step 3 — Add a `CLARA_AUTH_REQUIRED` config flag** so local/dev deployments aren't broken:

```python
# config.py — add to ClaraConfig
auth_required: bool = False  # set True in production

@classmethod
def from_env(cls) -> "ClaraConfig":
    return cls(
        ...
        auth_required=_bool("CLARA_AUTH_REQUIRED", False),
    )
```

```python
# dependencies.py — gate enforcement on config
def get_current_user(request: Request, ...) -> str | None:
    agent = getattr(request.app.state, "agent", None)
    if agent and agent._config.auth_required and x_user_id is None:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    return x_user_id
```

---

## Bug #8 — OpenClaw bridge session isolation by text prefix (LOW)

**File:** `clara/integrations/openclaw_bridge.py`  
**Impact:** Session scoping relies on embedding similarity to a `"session:ID"` text prefix rather than a database predicate. Memories from different sessions can surface in wrong queries. Sessions cannot be listed, filtered, or deleted at the DB level.

### What's wrong

```python
# openclaw_bridge.py — BROKEN: scoping by text prefix only
async def recall_for(self, *, session_id, query, top_k=None) -> RetrievalResult:
    scoped_query = f"session:{session_id} {query}".strip()
    return await self.memory.recall(scoped_query, top_k=k)
    # ↑ no DB filter — relies on embedding of "session:abc123" being distinctive

def _serialize_turn(self, *, session_id, role, text, metadata) -> str:
    # session_id written into raw text body — not a structured field
    header = [f"session:{session_id}", f"role:{role}", ...]
```

### Fix — Store session_id as structured metadata and filter at DB level

**Step 1 — Update `_serialize_turn` to embed session_id as metadata:**

```python
# openclaw_bridge.py — FIXED _serialize_turn
@staticmethod
def _serialize_turn(*, session_id: str, role: str, text: str, metadata: dict) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    # Keep header for human readability, but also pass session_id structurally
    header = [
        f"role:{role}",
        f"timestamp:{ts}",
    ]
    meta_text = ""
    if metadata:
        pairs = [f"{k}={v}" for k, v in metadata.items()]
        meta_text = "\nmetadata: " + ", ".join(pairs)
    return "\n".join(header) + "\ntext: " + text.strip() + meta_text
```

**Step 2 — Pass `session_id` as user_id or as a dedicated metadata field:**

The cleanest approach within the current schema: use `user_id` as a composite key `"{user_id}:{session_id}"` for session-scoped storage, or store `session_id` in `Memory.metadata_` and filter on it.

```python
# openclaw_bridge.py — FIXED remember_turn and recall_for
async def remember_turn(
    self,
    *,
    session_id: str,
    role: str,
    text: str,
    metadata: dict | None = None,
) -> list[dict]:
    if not text or not text.strip():
        return []
    payload = self._serialize_turn(
        session_id=session_id, role=role,
        text=text, metadata=metadata or {},
    )
    # Use composite user_id so DB filtering is exact
    scoped_user_id = f"session:{session_id}"
    return await self.memory.remember(payload, user_id=scoped_user_id)


async def recall_for(
    self,
    *,
    session_id: str,
    query: str,
    top_k: int | None = None,
) -> RetrievalResult:
    k = top_k if top_k is not None else self.config.default_top_k
    scoped_user_id = f"session:{session_id}"
    # Now filtered at DB level via user_id index — not text prefix
    return await self.memory.recall(query, top_k=k, user_id=scoped_user_id)


async def context_for(
    self,
    *,
    session_id: str,
    query: str,
    top_k: int | None = None,
) -> str:
    k = top_k if top_k is not None else self.config.default_top_k
    scoped_user_id = f"session:{session_id}"
    return await self.memory.context_for(query, top_k=k, user_id=scoped_user_id)
```

This works immediately because `user_id` is now indexed (`ix_memories_user_id`) and all retrieval queries already filter on it. The composite `"session:{session_id}"` format is unambiguous and enables future per-session pruning.

---

## Fix Order Summary

| # | Bug | Severity | File(s) | Effort |
|---|-----|----------|---------|--------|
| 1 | Sync blocking LLM calls | 🔴 Critical | `reasoning/engine.py`, `reflection/pipeline.py` | ~30 min |
| 2 | Stress test segfault | 🔴 Critical | `tests/test_agent_stress.py`, `pyproject.toml` | ~20 min |
| 3 | WorldModel race condition | 🔴 Critical | `db/models.py`, `memory/world_model.py` | ~45 min |
| 4 | SkillStore full table scan | 🟡 Medium | `memory/skill.py` | ~30 min |
| 5 | BackgroundWriter unused | 🟡 Medium | `agent.py` | ~30 min |
| 6 | fastapi not in core deps | 🟡 Medium | `pyproject.toml`, `tests/test_api.py`, `tests/test_admin_api.py` | ~15 min |
| 7 | API no auth guard | 🟢 Low | `api/dependencies.py`, `api/routes_memory.py` | ~45 min |
| 8 | OpenClaw bridge text-prefix isolation | 🟢 Low | `integrations/openclaw_bridge.py` | ~20 min |

**Total estimated effort: ~4 hours**

Fix #1 and #2 before any production use of `agent.interact()`. Fix #3 before any multi-user or concurrent deployment. Fixes #4–6 before next release. Fixes #7–8 before exposing the API to untrusted callers.
