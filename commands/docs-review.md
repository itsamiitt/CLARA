---
description: Review doc rot proposals; apply approved ones as one commit
argument-hint: [scope]
---

Run a CLARA documentation review: $ARGUMENTS

**Nothing executes without the user's explicit approval. Collect all
decisions first, execute once, produce ONE reviewable commit.**

1. Fetch `docs_report`. If empty or the ledger is unscanned, say so and
   offer `clara docs scan`. (Large repo? The `curator-reviewer` subagent
   can run this whole workflow in an isolated worktree and report back.)
2. Walk proposals in three groups, one decision per item
   (approve / skip / edit):
   a. **Archive candidates** — quote the evidence sentence from the report
      (lifecycle, staleness vs window, checkbox completion).
   b. **Duplicate clusters** — suggest which doc to keep (higher
      link_indegree, better tier, newer); the others become superseded.
   c. **Promotion suggestions** — plans that look complete (all checkboxes,
      merge evidence) but are not yet fulfilled: draft 1-5 distilled
      durable facts each, show them for editing.
3. Execute approved actions in one pass:
   - promotions → `docs_fulfill(path, distilled, evidence)`
   - duplicates → `docs_supersede(loser, keeper, rationale)`
   - archives → `clara docs archive <path>` via Bash (git-mv mode stages
     the moves)
4. Create ONE commit on a review branch (`docs-review/<date>`); when `gh`
   is available, offer to push and open a PR instead. The commit/PR body
   must list every action with its evidence sentence and the promoted
   memory ids. Example body line:
   `archive docs/plans/old.md — stale plan (94d > 60d), checkboxes 8/8`.
5. Report the summary: actions taken, skipped items, the commit/PR ref.
   Remind that `git revert` + `clara docs scan` undoes the repo side while
   promoted memories intentionally persist.
