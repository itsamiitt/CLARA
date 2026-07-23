# CLARA — Honest Assessment (2026-07-23)

**Reviewer conflict, stated plainly:** I wrote most of v0.2.0. Discount my
judgment; trust the commands. Every verdict here has a repro in the sibling
files.

## Would I rely on CLARA to hold my memory today?

**Yes, with caveats — for the zero-backend tier on a normal machine.**

The core storage engine is genuinely solid under the abuse I threw at it:
4 concurrent OS processes on one SQLite file → 40/40 rows, integrity ok; SIGKILL
mid-write → reopens clean with FTS in sync; FTS5 operator/SQL-injection queries
→ no crash, no table drop; the prompt-injection sanitizer actually neutralizes
forged context fences; supersede/forget never delete; concurrent world_model
upserts converge to one row with zero IntegrityError leaks; confidence clamps to
[0,1]. These are the properties a memory store lives or dies on, and they held.

The caveats are about the **edges**, not the core:
- Two real MEDIUM bugs existed in the 0.2.0 code I shipped (both now fixed +
  regression-tested): hooks aborted on unset `$HOME`, and negative `top_k`
  silently returned the wrong set. Neither corrupts data; both were reachable.
- The whole **LLM extraction tier is unverified** here (no key) and, per the
  test review, unverified *anywhere* — every extraction test is mocked. If you
  rely on CLARA to turn prose into correct facts automatically, that quality is
  untested. The zero-backend "the host model calls `memory_save`" path (which I
  did exercise) does not depend on it.
- The README tells new users to `pip install clara-memory`, which **fails** —
  the package isn't on PyPI yet.

## The 3 scariest things I found

1. **Nothing data-corrupting — and I tried.** The scariest *possible* class
   (silent corruption, wrong memory returned as authoritative, MCP-stream
   poisoning) I could not produce. That is a real, stated-with-evidence result,
   not flattery: SIGKILL, 4-process contention, corrupt-file, injection, and
   clock-skew probes all either degraded cleanly or held integrity.
2. **The decay scheduler's correctness is tested against a mock that
   re-implements it** (`TEST_SUITE_REVIEW.md` #1). The real SQL that decays and
   archives your memories has its *values* asserted nowhere. It happens to be
   correct (my probes over a real store showed sane decay + correct archival),
   but the suite would not catch a regression in that SQL. This is the finding
   most likely to bite a *future* change.
3. **The "no daemon, opportunistic maintenance" model works, but its only
   cross-process safety test is single-process.** I verified out-of-band that
   two real processes race the lock correctly (exactly one runs) — but that
   guarantee rests on a probe I ran, not a test that will re-run in CI.

## The single most important fix

**Rewrite `test_decay.py` (and add one real two-process concurrency test)
against a real file-backed store, asserting actual confidence/status values.**
The engine is correct today; the *test coverage of the engine's real SQL* is the
weakest load-bearing thing in the repo. The probes in `REPRODUCTIONS.md` are
ready to promote. (The two behavioral bugs, F1/F2, are already fixed — fixing
the test blind spot is what protects the next change.)

## What I'd test next that I couldn't test here

- **A real Claude Code session driving the MCP server over stdio** — the wire
  protocol, tool-schema validation, and that the shim spawn + PowerShell hook
  chain actually fire in a live session (I verified the pieces, not the whole).
- **The LLM tier with a real key** — extraction quality, negation handling,
  hedge rejection, and the `degraded_heuristic` fallback when the LLM host is
  down (C4's LLM half).
- **macOS/BSD `sh`** — the scripts are POSIX but I only have Git Bash on Windows.
- **A second machine** — JSONL export→import across hosts, and whether repo_id
  collision on a shared root commit actually merges two repos' ledgers.
- **Scale** — everything I ran topped out in the low hundreds to 10k rows;
  100k–1M behavior is projected from code, not measured end-to-end.

## Confidence in my own review

- **High** on the SQLite/FTS/concurrency/injection findings — those are real
  runs with verbatim output on this box, reproducible by anyone.
- **High** on F1/F2 being real and now fixed (re-ran both).
- **Medium** on the doc-truthfulness and config findings (simple, but
  `pip index` could in principle be a network artifact — I checked `git tag` is
  empty to corroborate).
- **Low / explicitly absent** on: the LLM tier (never ran it), the live MCP wire
  (never drove it from Claude Code), macOS/BSD, and scale beyond 10k. Treat any
  impression I have of those as a guess, not a finding.

## Bottom line

CLARA's memory core survived an honest adversarial pass: no CRITICAL, no HIGH,
no data loss, no injection. Two MEDIUM edge bugs in my own recent code, both
fixed and regression-tested. The real risk is not in the running code — it's in
the **test suite's blind spots** (mocked LLM, mocked decay SQL, single-process
concurrency), which mean the *next* change is less protected than the 737-green
number suggests. Fix the decay/concurrency test coverage and CLARA is
trustworthy to a degree few memory plugins could demonstrate under this much
abuse.
