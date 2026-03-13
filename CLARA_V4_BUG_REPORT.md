# CLARA v4 — Complete Bug Report

**Codebase audited:** `CLARA-main__4_.zip`  
**Audit date:** 2026-03-13  
**Total bugs found:** 13 (3 from original audit fixed · 5 original still open · 5 new bugs introduced)

---

## Summary Table

| # | Bug | Severity | Status | File(s) |
|---|---|---|---|---|
| 1 | Sync blocking LLM calls | Critical | ✅ Fixed | `reasoning/engine.py`, `reflection/pipeline.py` |
| 2 | Stress test segfault | Critical | ✅ Fixed | `tests/test_agent_stress.py` |
| 3 | WorldModelStore TOCTOU race | Critical | ❌ Not fixed | `memory/world_model.py` |
| 4 | SkillStore.match() full table scan | Medium | ❌ Not fixed | `memory/skill.py` |
| 5 | BackgroundWriter never used | Medium | ❌ Not fixed | `agent.py` |
| 6 | fastapi optional dep breaks CI | Medium | ✅ Fixed | `pyproject.toml`, test files |
| 7 | API user_id unauthenticated | Low | ❌ Not fixed | `api/routes_memory.py` |
| 8 | OpenClaw session isolation text-prefix only | Low | ❌ Not fixed | `integrations/openclaw_bridge.py` |
| 9 | **NEW** LanceDB search is a full in-memory scan | Critical | ❌ New | `retrieval/engine.py` |
| 10 | **NEW** Decay scheduler bypasses LanceDB sync | Critical | ❌ New | `scheduler/decay.py` |
| 11 | **NEW** `after_commit` listener routes to wrong LanceDB instance | High | ❌ New | `retrieval/engine.py` |
| 12 | **NEW** `_embedding_cache` not cleared on ORM expiry | Medium | ❌ New | `db/models.py` |
| 13 | **NEW** `pytest-cov` missing — no coverage enforcement | Low | ❌ New | `pyproject.toml` |

---

## Original Bugs — Status

---

### ✅ Bug #1 — Sync Blocking LLM Calls — FIXED

**Files:** `clara/reasoning/engine.py`, `clara/reflection/pipeline.py`

Both files now use `AsyncOpenAI` / `AsyncAnthropic` with proper `await`, and the new Ollama path correctly uses `run_in_executor`. The event loop is no longer blocked on LLM calls.

---

### ✅ Bug #2 — Stress Test Segfault — FIXED

**File:** `tests/test_agent_stress.py`, `pyproject.toml`

`_recall_guarded()` helper with `asyncio.Semaphore(MAX_CONCURRENT_RECALLS)` exists at line 238. The test is marked `@pytest.mark.stress` and `pyproject.toml` excludes it from the default run via `addopts = "-m 'not stress'"`. No more segfault in CI.

---

### ❌ Bug #3 — WorldModelStore TOCTOU Race — NOT FIXED

**File:** `clara/memory/world_model.py` lines 95–107

The `upsert()` method still does SELECT → INSERT with zero concurrency protection:

```python
# world_model.py lines 95-107
existing = await self._find_existing(entity_type, name, user_id=user_id)

if existing is not None:
    return await self._merge_properties(...)

return await self._create_new(...)   # ← two concurrent calls both reach here
```

No `IntegrityError` handler, no unique database constraint on `(user_id, entity_type, name)`, no `ON CONFLICT DO UPDATE`. Under concurrent load two agents writing the same entity simultaneously will produce duplicate rows.

**Fix required:**

```python
# clara/db/models.py — add unique index
from sqlalchemy import UniqueConstraint

__table_args__ = (
    # ... existing indexes ...
    Index(
        "uq_world_model_entity",
        "user_id",
        postgresql_where=text("memory_type = 'world_model' AND status = 'active'"),
    ),
)
```

```python
# clara/memory/world_model.py — wrap _create_new with conflict handler
from sqlalchemy.exc import IntegrityError

try:
    return await self._create_new(...)
except IntegrityError:
    await self._session.rollback()
    existing = await self._find_existing(entity_type, name, user_id=user_id)
    if existing is not None:
        return await self._merge_properties(existing, properties, ...)
    raise
```

---

### ❌ Bug #4 — SkillStore.match() Full Table Scan — NOT FIXED

**File:** `clara/memory/skill.py` line 246

```python
all_skills = await self.get_active_skills(user_id=user_id, limit=200)
# Then Python-side substring loop over all 200 rows
```

The docstring even acknowledges the problem: *"For production use, this should be augmented with embedding-based similarity search."* Every call to `match()` loads up to 200 rows into Python memory and iterates them with string comparison.

**Fix required:** Route through `RetrievalEngine.search()` with `memory_types=[MemoryType.skill]`, fall back to substring only if vector search returns zero results.

---

### ❌ Bug #5 — BackgroundWriter Never Used — NOT FIXED

**File:** `clara/agent.py` lines 259–271, 365, 384–450

`BackgroundWriter` is instantiated at line 365 and stored at `self._background_writer` at line 271. But `remember()` (starting line 384) processes everything inline through `MemoryUpdateEngine` synchronously — `enqueue()` is never called. The writer is created, stored, properly stopped in `close()`, but never does any work.

**Fix required:** Add a `wait: bool = True` parameter to `remember()`. When `wait=False`, route facts through `self._background_writer.enqueue()` instead of the inline update engine.

---

### ✅ Bug #6 — fastapi Optional Dep Breaks CI — FIXED

`fastapi` is correctly in `[api]` extras only. Both `test_api.py` and `test_admin_api.py` have `pytest.importorskip("fastapi")` and `pytest.importorskip("httpx")` guards. `aiosqlite` promoted to core deps.

---

### ❌ Bug #7 — API user_id Unauthenticated — NOT FIXED

**File:** `clara/api/routes_memory.py`, `clara/api/dependencies.py`

`dependencies.py` only contains `get_agent` and `get_session`. No `X-User-ID` header dependency, no authentication middleware, no `CLARA_AUTH_REQUIRED` flag. Any caller can pass `?user_id=anyone` and access another tenant's memories.

**Fix required:** Add `get_current_user` dependency to `dependencies.py` that reads from an `X-User-ID` header and raises `HTTP 401` when `CLARA_AUTH_REQUIRED=true` and the header is absent.

---

### ❌ Bug #8 — OpenClaw Session Isolation Text-Prefix Only — NOT FIXED

**File:** `clara/integrations/openclaw_bridge.py` lines 52–54, 65–66

`remember_turn()` calls `self.memory.remember(payload)` with no `user_id` argument. All sessions share the same tenant namespace in the database. The "isolation" is a query string prefix bias (`session:{session_id}`), not a hard database partition. Session A can retrieve Session B's memories if the query is semantically similar.

**Fix required:**

```python
# openclaw_bridge.py — pass user_id to remember() and recall()
return await self.memory.remember(payload, user_id=f"session:{session_id}")

# And in recall_for:
return await self.memory.recall(query, top_k=k, user_id=f"session:{session_id}")
```

---

## New Bugs Introduced in v4

---

### ❌ Bug #9 — LanceDB Search Is a Full In-Memory Scan — CRITICAL

**File:** `clara/retrieval/engine.py` lines 351–430

This is the most significant new bug. Despite the migration to LanceDB, vector search does **not** use LanceDB's native ANN (Approximate Nearest Neighbor) index. Instead `_search_candidates_sync()` loads all records into a Python dict (`self._records`) and does a full Python-side cosine similarity scan:

```python
# retrieval/engine.py lines 414-429
def _search_candidates_sync(self, ...):
    self._ensure_records_loaded_sync()    # loads ALL records into self._records dict
    with self._records_lock:
        records = list(self._records.values())    # copy entire dict to list

    ranked = sorted(
        (
            (record.memory_id, RetrievalEngine._cosine_similarity(query_vector, record.vector))
            for record in records          # iterates every single record
            if record.vector is not None
            and record.status == MemoryStatus.active.value
            and self._matches_filters(...)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:n_candidates]
```

`_ensure_records_loaded_sync()` at line 351 calls `table.to_arrow().to_pylist()` which reads the **entire LanceDB table into Python memory** on first call. After that, every query iterates `self._records` in Python. This is identical to the old SQLite full-scan fallback — just with an extra deserialization step on startup.

**Why this happened:** The LanceDB search API (`table.search(vector).metric("cosine").where(...).limit(n)`) was never called. The code loads data from LanceDB once, then ignores LanceDB for all subsequent searches.

**Impact:** O(n) query time at any scale. 10,000 memories = 10,000 Python cosine computations per query. The entire point of using LanceDB — its HNSW index and sub-linear ANN search — is completely unused.

**Fix required:** Replace `_search_candidates_sync` with actual LanceDB native search:

```python
def _search_candidates_sync(self, *, query_vector, n_candidates, memory_types, user_id):
    table = self._ensure_table_sync()

    where_parts = ["status = 'active'"]
    if user_id is not None:
        where_parts.append(f"user_id = '{self._escape(user_id)}'")
    if memory_types:
        values = ", ".join(f"'{mt.value}'" for mt in memory_types)
        where_parts.append(f"memory_type IN ({values})")
    where_clause = " AND ".join(where_parts)

    results = (
        table.search(query_vector)
        .metric("cosine")
        .where(where_clause, prefilter=True)
        .limit(n_candidates)
        .select(["memory_id", "_distance"])
        .to_list()
    )
    # cosine distance → similarity
    return [(r["memory_id"], 1.0 - r["_distance"]) for r in results]
```

Also remove `_records`, `_records_loaded`, `_merge_records`, `_ensure_records_loaded_sync`, `_matches_filters` — they are all part of the in-memory scan and are no longer needed once native LanceDB search is used.

---

### ❌ Bug #10 — Decay Scheduler Bypasses LanceDB Sync — CRITICAL

**File:** `clara/scheduler/decay.py` lines 177–295

The decay scheduler changes `Memory.status` to `archived` or `deprecated` using SQLAlchemy bulk `UPDATE` statements:

```python
# scheduler/decay.py lines 186-214
async with self._session_factory() as session:
    async with session.begin():
        stmt = select(Memory).where(...)
        result = await session.execute(stmt)
        for record in result.scalars():
            record.status = MemoryStatus.archived   # ← ORM attribute change
            record.confidence = new_confidence
```

The global `before_flush` / `after_commit` SQLAlchemy event listener installed in `retrieval/engine.py` **should** catch these ORM-level changes. However there is a critical gap: `_needs_lance_sync()` only triggers LanceDB sync if:

1. `is_new=True` (new record), OR
2. `memory._embedding_cache` is not None (embedding was set), OR
3. `status`, `user_id`, or `memory_type` has SQLAlchemy history changes

For condition 3 to work, the ORM object must have been loaded into the session with change tracking enabled. The scheduler loads records with `select(Memory)` in the same session — this works for ORM-tracked changes, so condition 3 **should** fire.

However the weekly pruning uses bulk `UPDATE` that completely bypasses the ORM:

```python
# scheduler/decay.py lines 277-288
stmt = (
    update(Memory)
    .where(Memory.memory_type == MemoryType.skill)
    .where(Memory.updated_at < skill_cutoff)
    .values(status=MemoryStatus.deprecated, updated_at=now)
)
await session.execute(stmt)   # ← bulk SQL UPDATE, ORM never sees this
```

This bulk UPDATE never triggers `before_flush` because no ORM objects are loaded — it goes straight to SQL. After this runs, archived/deprecated skills remain in LanceDB with `status='active'` and continue appearing in search results forever.

**Impact:** Stale memories that should be invisible continue polluting every search result. Decay is supposed to remove low-confidence memories from search — but they remain searchable indefinitely.

**Fix required:** After every bulk UPDATE in the scheduler, explicitly sync the affected records to LanceDB:

```python
# After bulk UPDATE in run_weekly_pruning:
affected_ids = [str(skill.memory_id) for skill in deprecated_skills]
lance = LanceRetrievalEngine.get_default()
lance.sync_records_sync([
    LanceMemoryRecord(
        memory_id=mid, vector=None,
        user_id="", memory_type="skill",
        status=MemoryStatus.deprecated.value
    )
    for mid in affected_ids
])
```

---

### ❌ Bug #11 — `after_commit` Listener Routes to Wrong LanceDB Instance — HIGH

**File:** `clara/retrieval/engine.py` line 845

The `after_commit` SQLAlchemy event listener is installed globally on the `Session` class and hardcodes `LanceRetrievalEngine.get_default()`:

```python
@event.listens_for(Session, "after_commit")
def _sync_lance_after_commit(session) -> None:
    snapshots = list(session.info.pop("_lance_pending_snapshots", {}).values())
    ...
    LanceRetrievalEngine.get_default().enqueue_records(snapshots)  # ← always default
```

`get_default()` returns the engine for `CLARA_LANCE_PATH` or `cls._default_path`. This is fine for a single agent. But if two `ClaraMemory` instances are created with different `lance_path` values — e.g. one for testing and one for production, or two different tenants with isolated vector stores — both will have their commits routed to whichever instance is currently registered as "default".

The test `conftest.py` uses `monkeypatch.setenv("CLARA_LANCE_PATH", ...)` and `LanceRetrievalEngine.reset_defaults()` to work around this, but application code has no protection.

**Impact:** In multi-agent or multi-tenant deployments using different lance paths, writes silently go to the wrong vector store. No error is raised. Data silently diverges between SQLite (correct) and LanceDB (wrong instance).

**Fix required:** Store the specific `LanceRetrievalEngine` instance in `session.info` at session creation time, and read it back in the listener:

```python
# In RetrievalEngine.__init__ or the session factory:
session.info["_lance_engine"] = self._lance

# In the after_commit listener:
lance = session.info.get("_lance_engine") or LanceRetrievalEngine.get_default()
lance.enqueue_records(snapshots)
```

---

### ❌ Bug #12 — `_embedding_cache` Not Cleared on ORM Expiry — MEDIUM

**File:** `clara/db/models.py` lines 203–221

The `embedding` property on `Memory` is implemented as a hybrid property with an instance-level `_embedding_cache`:

```python
@hybrid_property
def embedding(self) -> list[float] | None:
    cached = getattr(self, "_embedding_cache", None)
    if cached is None:
        return None
    return [float(value) for value in cached]

@embedding.setter
def embedding(self, value):
    self._embedding_cache = [float(item) for item in value] if value else None
```

SQLAlchemy expires ORM objects after commit by default (`expire_on_commit=True`). When a `Memory` object is expired and then re-accessed, SQLAlchemy reloads all mapped columns from the database — but `_embedding_cache` is an unmapped Python instance attribute. It is **never cleared** during expiry, refresh, or detach.

**Consequences:**

1. A `Memory` object loaded in session A has its embedding set (`_embedding_cache = [...]`)
2. Session A commits → object is expired
3. Any code that checks `memory.embedding is not None` (including `_needs_lance_sync`) will still see the old embedding value, even after the object has been expired and reloaded
4. If the embedding was updated in the database by another session, the stale `_embedding_cache` will be used for the LanceDB sync, writing the wrong vector

**Fix required:** Register a SQLAlchemy `after_bulk_update` or instance-expiry listener that clears `_embedding_cache`:

```python
from sqlalchemy import event as sa_event
from sqlalchemy.orm import InstanceEvents

@sa_event.listens_for(Memory, "expire")
def _clear_embedding_cache(target, attrs):
    target._embedding_cache = None
```

Or use `@reconstructor` to ensure `_embedding_cache` is always initialized to `None` on load:

```python
from sqlalchemy.orm import reconstructor

@reconstructor
def _init_on_load(self):
    self._embedding_cache = None
```

---

### ❌ Bug #13 — `pytest-cov` Missing — No Coverage Enforcement — LOW

**File:** `pyproject.toml`

`pytest-cov` is not in `[dev]` dependencies. Coverage has never been measured, no threshold is enforced, and the CI workflow has no `--cov` flag. The test suite runs but nobody knows what percentage of the codebase is actually exercised.

```toml
# Current dev deps — pytest-cov is absent
dev = [
    "pytest>=8.0,<10.0",
    "pytest-asyncio>=0.23,<1.0",
    "pytest-mock>=3.14,<4.0",
    "httpx>=0.27,<1.0",
    "hypothesis>=6.0,<7.0",
]
```

**Fix required:**

```toml
dev = [
    "pytest>=8.0,<10.0",
    "pytest-asyncio>=0.23,<1.0",
    "pytest-mock>=3.14,<4.0",
    "pytest-cov>=5.0,<7.0",         # ADD
    "httpx>=0.27,<1.0",
    "hypothesis>=6.0,<7.0",
]

[tool.coverage.run]
source = ["clara"]
branch = true
omit = ["clara/db/migrations/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 70
show_missing = true
```

Update CI:
```yaml
- name: Run tests
  run: pytest --cov=clara --cov-report=term-missing --cov-fail-under=70 -q
```

---

## Fix Priority Order

Fix in this order — each level unblocks the next.

### Level 1 — Fix first (data correctness at risk)

| Bug | Why first |
|---|---|
| #9 — LanceDB full in-memory scan | Every search is O(n). Defeats the entire purpose of the LanceDB migration. |
| #10 — Decay scheduler bypasses LanceDB | Archived memories stay searchable forever. Decay is silently broken. |
| #11 — `after_commit` wrong LanceDB instance | Multi-agent deployments silently write vectors to wrong store. |

### Level 2 — Fix next (functional correctness)

| Bug | Why second |
|---|---|
| #3 — WorldModelStore TOCTOU | Data duplication under concurrent load. |
| #12 — `_embedding_cache` not cleared | Stale embeddings written to LanceDB after session expiry. |
| #5 — BackgroundWriter never used | Feature was built and wired but does nothing. |

### Level 3 — Fix last (quality and completeness)

| Bug | Why last |
|---|---|
| #4 — SkillStore.match() full scan | Performance issue, not correctness. |
| #7 — API user_id unauthenticated | Security — important before any public deployment. |
| #8 — OpenClaw session isolation | Functional gap but only affects multi-session bridge use. |
| #13 — pytest-cov missing | Quality gate — straightforward to add. |

---

## What Was Fixed vs What Was Added

### Correctly fixed from original audit
- ✅ Bug #1 — Async LLM calls (`AsyncOpenAI` / `AsyncAnthropic` + `run_in_executor` for Ollama)
- ✅ Bug #2 — Stress test semaphore + `@pytest.mark.stress` exclusion
- ✅ Bug #6 — `fastapi` optional dep + `importorskip` guards in test files

### New functionality correctly added in v4
- ✅ LanceDB migration — storage architecture is right, sync hooks exist
- ✅ Ollama support in extractor, reasoning, reflection, embeddings
- ✅ `conftest.py` with shared `lance_fixture` for test isolation
- ✅ `pyproject.toml` cleaned up — `asyncpg` and `pgvector` removed

### What was added but has bugs
- ❌ LanceDB search — architecture is correct but search path never reaches LanceDB's ANN index
- ❌ LanceDB sync — ORM hook works, but bulk UPDATEs in scheduler bypass it
- ❌ Global event listener — correct approach but hardcodes `get_default()` instead of the session's specific engine
