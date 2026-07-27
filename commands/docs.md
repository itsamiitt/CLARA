---
description: Document standing or repo rot report from the ledger
argument-hint: [path]
---

Check the CLARA document ledger for: $ARGUMENTS

If a path is given, call `docs_status` with it and relay the standing in one
or two sentences — tier, lifecycle, and any warning signals (staleness, dead
references, near-duplicates, supersession). If the document is superseded or
quarantined, say what replaced it and advise against following it.

If no path is given, run `clara docs report` via Bash (or summarize from
`docs_status` calls if Bash is unavailable) and present the rot report:
stale documents, dead-reference documents, duplicate clusters, and archive
candidates. These are proposals only — never move or edit files without the
user's explicit approval. If the ledger is empty, suggest `clara docs scan`.

**Finding the CLI.** A plugin-only install does not put `clara` on `PATH` —
the bootstrap keeps it in the plugin's own venv and shims it to
`$CLAUDE_PLUGIN_DATA/shim/clara` (default `~/.clara/plugin/shim/clara`, plus
`clara.exe` on Windows). If a bare `clara ...` call reports "command not
found", re-run it with that path instead of telling the user it is broken.
