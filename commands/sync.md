---
description: Sync CLARA memory with Claude Code native memory files (CLAUDE.md / auto-memory)
argument-hint: "[export|import|status]"
---

Run the CLARA ⇄ native-memory bridge and report what happened.

1. Run `clara sync $ARGUMENTS` with Bash (no argument = import then export,
   which is the safe default order).
2. Relay the summary: how many memories were imported from native files,
   whether MEMORY.md's CLARA section and the `clara-memory.md` topic file
   were refreshed, and any hand-edited-fence conflict notice.
3. If lines were skipped as "did not extract", tell the user they can re-run
   with `clara sync --verbatim` to keep them as plain note beliefs.
4. Never edit MEMORY.md or CLAUDE.md by hand here — the bridge owns the
   fenced section, and everything else belongs to the user (or to Claude's
   own auto-memory writes).
