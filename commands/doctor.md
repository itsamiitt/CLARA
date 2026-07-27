---
description: Check CLARA's health — store, schema, venv, and why memory might be missing
---

Diagnose CLARA and report plainly whether memory is working.

**Finding the CLI.** A plugin-only install does not put `clara` on `PATH` —
the bootstrap keeps it in the plugin's own venv and shims it to
`$CLAUDE_PLUGIN_DATA/shim/clara` (default `~/.clara/plugin/shim/clara`, plus
`clara.exe` on Windows). Try `clara doctor` first; if that reports "command
not found", re-run with that path. Do not report the tool as broken because a
bare `clara` is not on `PATH` — for the recommended install it never is.

1. Run `clara doctor` with Bash. It exits 0 (healthy), 1 (degraded), or 2
   (unusable), and prints the resolved store path, schema version, kill-switch
   flags, ledger counts, the active venv, and the tail of the install log.
2. Also call the `memory_stats` MCP tool. If the tool answers but `clara
   doctor` cannot be found, memory is working and only the CLI path is
   unknown — say that, rather than implying memory is down.
3. Translate the output instead of pasting it:
   - **healthy** — say so, give the store path and the memory count.
   - **schema newer than this build** — the store was written by a newer
     CLARA, so it is open read-only and writes are refused. The fix is
     `pip install -U clara-memory`; reads keep working meanwhile.
   - **installing** — the background build is still running. Memory arrives
     next session; nothing is broken.
   - **no store yet** — expected before the first memory is saved.
4. If the store is unhealthy, mention that `<store dir>/backups/` holds
   timestamped copies and that `clara restore <file>` reverses a bad state,
   taking a pre-restore backup first. Never run `restore` or
   `uninstall --purge-memories` without the user asking for it explicitly.
