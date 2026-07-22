---
description: Search CLARA memory and summarize what is known about a topic
argument-hint: <topic or question>
---

Recall from CLARA memory: $ARGUMENTS

If a topic is given, call `memory_search` with its key terms (try a second
phrasing if the first returns nothing). If no topic is given, call
`memory_recent`. Answer the question from the hits in a few sentences —
lead with the answer, mention confidence or age only when it matters, and
say plainly when memory has nothing relevant.
