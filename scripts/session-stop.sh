#!/bin/sh
# CLARA Stop hook: one debounced nudge when a plan doc looks complete.
#
# Pure sh, zero subprocesses, always exits 0, fail-open on every branch.
# Fuel: $BASE/proposals/<repo_id>.txt (written by the fastpath). repo_id
# comes from .git/clara-marker (fastpath-stamped; immune to path-spelling
# differences), falling back to $BASE/proposals/index.tsv for worktrees
# (whose .git is a file). Debounce: one nudge per session via
# $BASE/session-flags/<session>.done. Stop-hook stdout is not model-visible
# feedback (sprint0-findings S1); plain stdout per the hooks doc is the
# intended channel here.

set -u

# Kill switch: the proposals nudge is doc-curator behavior.
case "${CLARA_DOCS_ENABLED:-1}" in
  0 | false | no | off) exit 0 ;;
esac

BASE="${CLARA_HOME:-$HOME/.clara}"
[ -d "$BASE/proposals" ] || exit 0

FLAG_DIR="$BASE/session-flags"
FLAG="$FLAG_DIR/${CLAUDE_SESSION_ID:-pid$$}.done"
[ -e "$FLAG" ] && exit 0

# Repo root: walk up from the hook cwd looking for .git (no subprocess).
dir=$PWD
root=''
while [ -n "$dir" ] && [ "$dir" != "/" ]; do
  if [ -e "$dir/.git" ]; then
    root=$dir
    break
  fi
  next=${dir%/*}
  [ "$next" = "$dir" ] && break
  dir=$next
done
[ -n "$root" ] || exit 0

rid=''
if [ -f "$root/.git/clara-marker" ]; then
  IFS= read -r rid <"$root/.git/clara-marker" || rid=''
fi
if [ -z "$rid" ] && [ -f "$BASE/proposals/index.tsv" ]; then
  while IFS='	' read -r idx_root idx_rid; do
    if [ "$idx_root" = "$root" ]; then
      rid=$idx_rid
      break
    fi
  done <"$BASE/proposals/index.tsv"
fi
[ -n "$rid" ] || exit 0

PROPOSALS="$BASE/proposals/$rid.txt"
[ -s "$PROPOSALS" ] || exit 0

IFS='	' read -r first_path _rest <"$PROPOSALS" || exit 0
[ -n "$first_path" ] || exit 0

printf 'CLARA: %s looks complete — consider /clara:done\n' "$first_path"
mkdir -p "$FLAG_DIR" 2>/dev/null || exit 0
: >"$FLAG" 2>/dev/null || true
exit 0
