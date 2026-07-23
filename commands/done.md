---
description: Close out a completed plan — distill it into memory
argument-hint: [plan path]
---

Close out completed plan work: $ARGUMENTS

1. Identify the plan document: the given path, or the plan-type documents
   touched this session (check `docs_status` on candidates; the session may
   have shown a "looks complete" nudge naming one).
2. Verify it is actually done — checkboxes, the work you performed, merged
   commits/PRs. If it is not complete, say what remains and stop.
3. Draft 1-5 distilled durable facts from the plan's OUTCOMES: decisions
   made, constraints adopted, standards set. Not implementation trivia.
   Shape each as a memory_save fact (belief subject/relation/object,
   world_model entity state, or skill).
4. Show the drafted facts to the user for confirmation; adjust as asked.
5. Call `docs_fulfill(path, distilled, evidence)` with the confirming
   commit/PR ref as evidence. Report the memory ids saved.
6. If this plan replaced an older plan, also call `docs_supersede` for the
   old one. Never claim anything was deleted — documents are quarantined,
   memories are retired, nothing is destroyed.
