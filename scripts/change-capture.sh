#!/bin/sh
# CLARA PostToolUse hook: journal file/repo changes for the code index.
#
# stdin: hook JSON ({tool_name, tool_input, cwd, session_id, ...}). stdout
# stays empty — PostToolUse output is not something a turn should ever see.
# Fires on every Edit/Write/MultiEdit/NotebookEdit and every Bash command, so
# the wrapper drops what cannot possibly enqueue before paying for an
# interpreter: a Bash payload with no git/package-manager keyword is over in
# pure sh. Fail-open on every branch: no store, no interpreter, any failure
# => exit 0 silently; the daily maintenance walk reconciles anything missed.

set -u

case "${CLARA_MEMORY_ENABLED:-1}" in
  0 | false | no | off)
    cat >/dev/null 2>&1
    exit 0
    ;;
esac

payload=$(cat 2>/dev/null || true)
[ -n "$payload" ] || exit 0

# Keyword pre-filter for Bash events (the noisiest matcher). Keys on the
# tool_name FIELD, not a bare "Bash" substring — an edited file's content may
# legitimately contain the word Bash and must not push a file event through
# the keyword gate. Coarse on purpose: a false positive costs one
# interpreter start, a false negative waits for the daily walk.
case "$payload" in
  *'"tool_name":"Bash"'* | *'"tool_name": "Bash"'*)
    case "$payload" in
      *git*|*npm*|*yarn*|*bun*|*pip*|*poetry*|*cargo*|*'go get'*|*'uv '*) : ;;
      *) exit 0 ;;
    esac
    ;;
esac

DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-${HOME:-/tmp}/.clara}/plugin}"

# Keep in sync with prompt-recall.sh / session-start.sh / bootstrap.sh.
find_bin() {
  for _cand in "$1/bin/$2" "$1/Scripts/$2" "$1/Scripts/$2.exe"; do
    if [ -x "$_cand" ]; then
      printf '%s' "$_cand"
      return 0
    fi
  done
  return 1
}

PYBIN=''
if ! PYBIN=$(find_bin "$DATA_DIR/current" python); then
  PYBIN=''
  if [ -f "$DATA_DIR/current.path" ]; then
    _venv=''
    # `|| true`, not `|| _venv=''`: the pointer file has no trailing
    # newline, so read exits nonzero at EOF with the value already in hand.
    IFS= read -r _venv <"$DATA_DIR/current.path" || true
    _venv=$(printf '%s' "$_venv" | tr '\\' '/')
    if [ -n "$_venv" ]; then
      PYBIN=$(find_bin "$_venv" python) || PYBIN=''
    fi
  fi
fi

[ -n "$PYBIN" ] || exit 0

printf '%s' "$payload" | "$PYBIN" -m clara.fastpath.change_capture >/dev/null 2>&1 || true
exit 0
