# CLARA — Test Suite Review (2026-07-23)

This critique was produced by a **separate agent** told to attack the tests
(most of which I, the reviewer, wrote or extended in 0.2.0) with a skeptical
eye. I have verified its headline claims against the source. Baseline: `pytest
-q` → 737 passed; `pytest --cov` → 81.35%. **Passing ≠ correct** — this file is
about what the green run does *not* prove.

Ground truth that colors everything: **no test anywhere uses a real embedding
model or a real LLM.** Every extraction/embedding path uses a hash-based fake
backend or a stub extractor (`test_agent_stress.py:65`, `test_retrieval_stress.py:29`,
`test_api.py:30`, `test_reasoning.py:28`, `test_integration.py:45`,
`test_decay_search_integration.py:31`).

## The 10 most load-bearing gaps

1. **`test_decay.py` validates decay against a mock that re-implements the code
   under test.** `_make_session_factory` (`tests/test_decay.py:63`) routes queries
   by substring-matching compiled SQL (`"decay_rate >" in where`, `"'event'" in
   where`, `"'skill'" in where`) and **re-computes the answer in Python** (event
   cutoff at `:174`, unlinked-belief guard `not (r.metadata_ or {}).get("related_beliefs")`
   at `:179`, the UPDATE application in `_apply_update`/`_apply_one`). Every
   `TestRunDailyDecay` / `TestRunWeeklyPruning` case runs against this mock. The
   real-SQLite integration test (`test_decay_search_integration.py:128`) only
   asserts `summary["archived"] >= 1` and enum/membership — **it never checks a
   decayed confidence value.** A wrong real WHERE/UPDATE passes green because the
   mock computes the right answer itself. *This is the single most important test
   to rewrite against a real store.*

2. **No test runs two processes / two event loops on one SQLite file.** The
   read-only-reader + WAL-writer + cross-process `.maintenance.lock` design — the
   product's entire concurrency story — is only exercised in-process via
   `asyncio.gather` (`test_reliability.py:86`). `test_fastpath.py` always closes
   the writer *before* the subprocess reader runs (`:89→:91`), so the reader
   never reads a live-written file. *(This review's own probes filled part of
   this gap out-of-band — 4-process writers, SIGKILL, 2-process maintenance all
   pass — but the suite itself asserts none of it.)*

3. **`test_retrieval_stress.py:112` interleaved-write concurrency test is a
   tautology:** the reader does `ok += 1 if result.total >= 0` — a count is
   always ≥ 0 — then asserts the loop ran 15 times. It verifies nothing about
   concurrent correctness.

4. **LLM extraction quality is never tested.** `FactExtractor.extract` is 100%
   mocked; only the JSON post-parser `_parse_llm_response` is real (that part is
   genuinely well-tested). The "prompt enforces rules" tests are tautologies:
   `assert "JSON" in SYSTEM_PROMPT` (`:243`), `assert word in SYSTEM_PROMPT`
   (`:248`). Any regression in extraction *quality* — wrong relation, missed
   negation, hedge leakage, hallucination — is invisible.

5. **MCP: zero stdio / JSON-RPC.** `test_mcp_server.py` gates on
   `importorskip("mcp")`; the two tests call `list_tools()` and `call_tool()`
   in-process and do `assert "disabled" in str(payload)`. No wire framing, no
   `initialize` handshake, no tool-schema validation, no serialization of a
   non-trivial result. A broken stdio server ships green.

6. **LanceDB ANN ranking quality untested.** Unit tests mock the vector search
   wholesale; the one "native" test uses a fake table recording query-builder
   calls; `TestLanceCommitRouting` asserts `enqueue_records.assert_called_once()`
   (a mock-call count). Integration tests use 8-dim hash vectors with
   membership-only assertions — order/precision never checked.

7. **Stress tier tops out at ~324 rows.** `test_agent_stress.py` is the suite's
   best test (real SQLite, asserts exact action tallies and row counts) but
   scale is tiny and "concurrent" is one-loop `asyncio.gather`.
   `test_retrieval_stress.py` is a no-crash test. No thousand/million-row
   degradation coverage.

8. **Route/protocol coverage hides behind optional extras.** `test_api.py`,
   `test_admin_api.py` (`importorskip fastapi`/`httpx`), `test_mcp_server.py`
   (`importorskip mcp`), and every bash-gated hook test (`_working_bash()` in
   `test_plugin_layout.py:36`, also in `test_verdicts.py:321`,
   `test_governance.py:311`) **silently skip** on a minimal-install CI and the
   run still goes green. The `windows-native` CI job and full-extra `test.yml`
   mitigate this in *practice*, but a bare `pip install -e .` + `pytest` proves
   far less than the green suggests.

9. **Module blind spots:** `clara/main.py` (no direct test), `reasoning/context.py`
   (one fully-mocked test), `routes_interaction.py` (agent `AsyncMock`ed).

10. **Net:** strong on pure-function math and deterministic post-processing
    (confidence formula, JSON parsing, sanitizer, simhash, repoid, the new
    `store.py`/`security.py`/`porting.py`/`lexical_unicode.py` — all genuinely
    good bad-input coverage). Weak wherever the *real* boundary lives — LLM, ANN,
    cross-process SQLite, MCP wire — which is exactly where production breaks.

## Assessment of the 0.2.0 test additions (same skeptical standard)

- `test_store.py` — **good**: real `git init`/worktree, resolution matrix,
  three-way parity (`:112`). Gap: single-orphan only; no permission-denied/symlink paths.
- `test_porting.py` — **good**: byte-equal round-trip, dedup, dry-run, secret
  skip, backup rotation+integrity. Gap: no truncated/corrupt mid-stream line, no unicode round-trip.
- `test_security.py` — **good**: real positive+negative pattern tables, size
  caps. Gap: no homoglyph secrets, no multi-secret redaction ordering.
- `test_lexical_unicode.py` — **best bad-input coverage in the suite**:
  diacritics, eszett, CJK bigrams, mixed script. Gap: trigram assertion behind a runtime skip.
- `test_bridge.py` — covers markers/conflict/idempotency/round-trip + 500-record
  budget. Gap: no unicode/RTL in fence bodies, no concurrent export races.
- `test_reliability.py` — proves read-only fastpath and WAL-writer **in
  isolation** but never simultaneously; the maintenance lock is single-loop only (gap #2).

## Recommended test work, ranked

1. Rewrite `test_decay.py` daily/weekly cases against a **real file-backed
   store** and assert decayed **confidence values**, not just `archived >= 1`.
2. Add a real two-process concurrency test (spawn 2+ `python -c` writers on one
   DB; assert row count + `integrity_check` + FTS==base). *The probes in
   `REPRODUCTIONS.md` are ready to promote into the suite.*
3. One real end-to-end MCP stdio test (spawn `clara-mcp`, do an `initialize` +
   `tools/call`, assert framed JSON-RPC).
4. De-tautologize `test_retrieval_stress.py:112` — assert the recalled set
   matches what was written, not `total >= 0`.
5. Promote the review probes (F1/F2 already added as `test_review_regressions.py`;
   add SIGKILL-integrity and negative-input fuzz).
