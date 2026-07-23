---
description: Check document standing or the repo rot report from the CLARA ledger
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
