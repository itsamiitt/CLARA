#!/bin/sh
# CLARA PostToolUse(Read) hook: annotate reads of quarantined documents.
#
# stdin: the hook JSON (tool_input.file_path). Output on hit: the S1-verified
# channel — stdout JSON {"hookSpecificOutput":{"hookEventName":"PostToolUse",
# "additionalContext":"CLARA: ..."}} on exit 0 (see docs/plans/
# sprint0-findings.md). Pure sh + one awk; jq used for parsing when present.
# repo_id + canonical root come from .git/clara-marker (fastpath-stamped),
# so path-spelling differences (8.3 vs long, MSYS vs native) cannot break
# the lookup. Fail-open everywhere: any miss or parse failure exits 0
# silently. Per-file once per session via $BASE/session-flags/<session>.read/.

set -u

BASE="${CLARA_HOME:-$HOME/.clara}"

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
if [ -z "$root" ] || [ ! -f "$root/.git/clara-marker" ]; then
  cat >/dev/null 2>&1
  exit 0
fi
rid=''
marker_root=''
{
  IFS= read -r rid
  IFS= read -r marker_root
} <"$root/.git/clara-marker" || exit 0
MANIFEST="$BASE/quarantine/$rid.tsv"
if [ ! -f "$MANIFEST" ]; then
  cat >/dev/null 2>&1
  exit 0
fi

INPUT=$(cat 2>/dev/null) || exit 0
[ -n "$INPUT" ] || exit 0

if command -v jq >/dev/null 2>&1; then
  FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || FILE=''
else
  # Conservative fallback: first "file_path":"..." value, JSON escapes undone.
  FILE=$(printf '%s' "$INPUT" | awk '
    match($0, /"file_path"[[:space:]]*:[[:space:]]*"[^"]*"/) {
      v = substr($0, RSTART, RLENGTH)
      sub(/^"file_path"[[:space:]]*:[[:space:]]*"/, "", v)
      sub(/"$/, "", v)
      gsub(/\\\\/, "\\", v)
      gsub(/\\\//, "/", v)
      print v
      exit
    }' 2>/dev/null) || FILE=''
fi
[ -n "$FILE" ] || exit 0

SESSION="${CLAUDE_SESSION_ID:-pid$$}"
FLAG_DIR="$BASE/session-flags/$SESSION.read"
mkdir -p "$FLAG_DIR" 2>/dev/null || exit 0

# One awk: relativize against either root spelling, manifest match, per-file
# debounce (getline probes the flag file), JSON emission. Values travel via
# ENVIRON, never -v: awk -v escape-processes backslashes and would corrupt
# Windows paths.
CLARA_HK_FILE="$FILE" CLARA_HK_ROOT_A="$marker_root" CLARA_HK_ROOT_B="$PWD" \
CLARA_HK_MANIFEST="$MANIFEST" CLARA_HK_FLAGDIR="$FLAG_DIR" \
awk '
BEGIN {
  file = ENVIRON["CLARA_HK_FILE"]
  root_a = ENVIRON["CLARA_HK_ROOT_A"]
  root_b = ENVIRON["CLARA_HK_ROOT_B"]
  manifest = ENVIRON["CLARA_HK_MANIFEST"]
  flagdir = ENVIRON["CLARA_HK_FLAGDIR"]
  gsub(/\\/, "/", file)
  rel = ""
  if (root_a != "" && substr(file, 1, length(root_a) + 1) == root_a "/")
    rel = substr(file, length(root_a) + 2)
  else if (root_b != "" && substr(file, 1, length(root_b) + 1) == root_b "/")
    rel = substr(file, length(root_b) + 2)
  else if (file !~ /^([A-Za-z]:)?\//)
    rel = file
  if (rel == "") exit 0

  flagname = rel
  gsub(/[\/:]/, "_", flagname)
  flag = flagdir "/" flagname
  if ((getline dummy < flag) >= 0) { close(flag); exit 0 }

  note = ""
  while ((getline line < manifest) > 0) {
    tab = index(line, "\t")
    if (tab == 0) continue
    if (substr(line, 1, tab - 1) == rel) {
      rest = substr(line, tab + 1)
      tab2 = index(rest, "\t")
      status = (tab2 > 0) ? substr(rest, 1, tab2 - 1) : rest
      extra = (tab2 > 0) ? substr(rest, tab2 + 1) : ""
      note = rel " is " status
      if (extra != "") note = note " (" extra ")"
      note = note " — treat as historical record, not current guidance"
      break
    }
  }
  close(manifest)
  if (note == "") exit 0

  gsub(/\\/, "\\\\", note); gsub(/"/, "\\\"", note)
  printf "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"CLARA: %s\"}}\n", note
  print "" > flag
  close(flag)
}' 2>/dev/null
exit 0
