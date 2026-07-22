---
description: Retire a memory from CLARA (never hard-deletes)
argument-hint: <memory id or description of what to forget>
---

Forget from CLARA memory: $ARGUMENTS

If given a memory id, call `memory_forget` with it directly. Otherwise call
`memory_search` to find matching memories; if exactly one clear match, forget
it; if several plausible matches, list them with ids and ask which to retire
before calling `memory_forget`. Memories are retired (status change), never
hard-deleted — say so if the user asks about recovery. Confirm what was
retired in one line.
