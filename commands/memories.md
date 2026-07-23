---
description: Show an overview of recent CLARA memories and store stats
argument-hint: [how many, default 10]
---

Show the CLARA memory overview. Call `memory_stats` for store totals and
`memory_recent` (n = $ARGUMENTS if given, else 10) for the latest entries.
Present a compact list — one line per memory with type, content summary, and
confidence — followed by one line of totals per type. Offer to expand or
search if the user wants more.
