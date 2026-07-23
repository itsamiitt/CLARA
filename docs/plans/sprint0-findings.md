# Sprint 0 findings (S1–S5)

Recorded 2026-07-22 on the primary dev machine (Windows Server 2025 Datacenter,
Python 3.12.10, Claude Code CLI 2.1.218, node 24 / npm 11). Docs citations are
from https://code.claude.com/docs/en/hooks fetched 2026-07-22.

## S1 — How does a PostToolUse hook return text Claude sees?

**Question.** JSON `additionalContext`? stderr + exit code? What is the exact
mechanism?

**Answer.** Two mechanisms exist; the primary one is stdout JSON on exit 0:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "text Claude will see"
  }
}
```

`additionalContext` is injected as a system reminder at the tool-result
location in the conversation; Claude reads it on the next model request. The
same `hookSpecificOutput` envelope also supports `updatedToolOutput`, which
rewrites the tool's result before Claude sees it. Plain stdout (non-JSON, exit
0) goes to the debug log only — it is NOT shown to Claude (the stdout-as-context
exception covers only `UserPromptSubmit`, `UserPromptExpansion`, and
`SessionStart`). The secondary mechanism: exit code 2 shows stderr to Claude
as an error message — non-blocking for PostToolUse since the tool already ran.
Top-level `decision: "block"` + `reason` also feeds `reason` back to Claude.

**Evidence.**
- Docs: https://code.claude.com/docs/en/hooks (fetched 2026-07-22): "For most
  events, stdout is written to the debug log but not shown in the transcript.
  The exceptions are `UserPromptSubmit`, `UserPromptExpansion`, and
  `SessionStart` ..."; PostToolUse `hookSpecificOutput` example with
  `additionalContext`; exit 2 documented as non-blocking for PostToolUse with
  stderr shown to Claude.
- Live minimal example (CLI 2.1.218): a scratchpad project with
  `.claude/settings.json`:

  ```json
  {
    "hooks": {
      "PostToolUse": [
        {
          "matcher": "Read",
          "hooks": [{"type": "command", "command": "python <abs>/hook.py"}]
        }
      ]
    }
  }
  ```

  where `hook.py` reads stdin and prints the `hookSpecificOutput` JSON above
  with the token `CLARA-SPIKE-TOKEN-9317`. Running
  `claude -p "Use the Read tool to read x.txt, then repeat any token you were
  given." --allowedTools "Read"` produced a reply containing
  `Token from hook: CLARA-SPIKE-TOKEN-9317` — the model demonstrably received
  the injected context.

**Implication.** The curator/graph layers can push per-tool feedback (doc
staleness notes, graph hits) into the conversation from a PostToolUse hook via
`hookSpecificOutput.additionalContext`, exit 0. Reserve exit 2 for errors.

## S2 — Can a plugin hook watch `*.md` repo-wide, and at what cost?

**Question.** Is there a file-watch event? If unsupported/expensive, git
dirty-check remains the default.

**Answer.** Supported — the prompt's premise is outdated. A `FileChanged` hook
event exists: "fires when a watched file changes on disk. The `matcher` field
specifies which filenames to watch." The matcher is NOT a glob: patterns made
only of letters, digits, `_`, and `|` are exact filename matches; anything
else makes the pattern an unanchored regex, and only `|` separates
alternatives. Repo-wide Markdown watching is therefore the regex matcher
`\.md$` (a glob like `**/*.md` would be interpreted as regex, not glob).
Cost: one hook process spawn per change event — with a Python hook that is
~69 ms per event on this machine (see S3 cold number), which is fine for
occasional saves but noisy for bulk operations (branch switches touching many
files fire many events).

**Evidence.** Same docs fetch: FileChanged event table entry; "`FileChanged`
and `StopFailure` use a narrower exact-match set of letters, digits, `_`, and
`|` only"; "The `FileChanged` event doesn't follow these rules when building
its watch list."

**Implication.** The plan's default stands **by choice, not necessity**: git
dirty-check at session boundaries remains the default signal for the curator
(cheap, batched, no per-event process spawns); `FileChanged` with matcher
`\.md$` is available as an opt-in for teams that want immediate signals.

## S3 — Latency baseline: SQLite indexed read, stdlib only

**Question.** Cold interpreter+query time and warm time for an indexed read on
a 10k-row SQLite DB on this machine.

**Answer** (Python 3.12.10, Windows Server 2025, local NVMe temp dir):

| Measurement | Time |
| --- | --- |
| Cold: full `python bench_s3.py` process (interpreter + connect + indexed query) | 65–72 ms, median **69 ms** (5 runs) |
| First indexed query in-process (page cache cold-ish) | **0.559 ms** |
| Warm indexed query in-process (median of 100) | **0.025 ms** |

Interpreter startup dominates the cold path (~68 of 69 ms); the query itself
is sub-millisecond even cold.

**Evidence.** Script run from the session scratchpad (throwaway, not
committed), embedded verbatim:

```python
import os
import sqlite3
import sys
import time

db = sys.argv[1]
fresh = not os.path.exists(db)
conn = sqlite3.connect(db)
if fresh:
    conn.executescript(
        "CREATE TABLE mem (id INTEGER PRIMARY KEY, key TEXT, val TEXT);"
        "CREATE INDEX ix_mem_key ON mem(key);"
    )
    conn.executemany(
        "INSERT INTO mem(key, val) VALUES (?, ?)",
        [(f"k{i:05d}", f"value-{i}" * 10) for i in range(10_000)],
    )
    conn.commit()
t0 = time.perf_counter()
conn.execute("SELECT val FROM mem WHERE key = ?", ("k07777",)).fetchone()
t1 = time.perf_counter()
warm = []
for _ in range(100):
    a = time.perf_counter()
    conn.execute("SELECT val FROM mem WHERE key = ?", ("k07777",)).fetchone()
    warm.append(time.perf_counter() - a)
print(f"first-query-in-process: {(t1 - t0) * 1000:.3f} ms")
print(f"warm-median: {sorted(warm)[50] * 1000:.4f} ms")
```

Cold timing: 5× `subprocess.run([sys.executable, script, db])` wall-clock from
a wrapper (72, 72, 69, 69, 65 ms).

**Implication.** Hook budgets: any Python-based hook pays ~70 ms interpreter
tax before doing anything; DB access adds well under 1 ms. Per-prompt hooks
are fine; per-file-change hooks multiply the 70 ms by event count (see S2).

## S4 — Target platform statement

**Answer.** Recorded in `README.md` (new section "Supported platforms",
between "Reliability" and "Development"): the plugin surface targets
**macOS, Linux, and WSL** at launch; native Windows is best-effort — the
library, CLI, and test suite run on Windows (this sprint's suite was
developed and passes on Windows Server 2025), but plugin hooks are exercised
only on POSIX shells.

## S5 — Publish path: `claude plugin validate`

**Question.** Run `claude plugin validate --strict .` (may fail — no manifest
yet); record the command in a CI workflow stub.

**Answer.** Ran at the repo root with CLI 2.1.218:

```text
$ claude plugin validate --strict .
Validating plugin manifest: C:\Users\Administrator\Downloads\CLARA\CLARA

✘ Found 1 error:

  ❯ directory: No manifest found in directory. Expected .claude-plugin/marketplace.json or .claude-plugin/plugin.json

✘ Validation failed
(exit code 1)
```

Expected failure: the manifest lands in milestone 02. `--strict` treats
warnings as errors — the CI-appropriate mode. The command is wired into
`.github/workflows/plugin.yml`: a `validate` job (ubuntu-latest) that checks
out the repo, installs node 22 + `@anthropic-ai/claude-code`, and runs
`claude plugin validate --strict .` — with every step gated on
`hashFiles('.claude-plugin/plugin.json') != ''` so the workflow is a green
no-op until the manifest exists, then validates automatically from milestone
02 onward.

**Implication.** No CI changes needed when the manifest lands — the gate
opens by itself.
