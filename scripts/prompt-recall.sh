#!/bin/sh
# CLARA UserPromptSubmit hook: recall stored memories that match this prompt.
#
# stdin: hook JSON ({prompt, cwd, session_id, ...}); stdout on exit 0 is
# added to the model's context (verified against the hooks documentation —
# same channel as SessionStart). Silence is the default: no store, no match,
# no interpreter, any failure at all => exit 0 with no output. This runs on
# EVERY prompt, so the wrapper stays free of subprocesses until it knows an
# interpreter exists to run.

set -u

case "${CLARA_MEMORY_ENABLED:-1}" in
  0 | false | no | off)
    cat >/dev/null 2>&1
    exit 0
    ;;
esac

SCRIPT_DIR=$(dirname "$0")
DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-${HOME:-/tmp}/.clara}/plugin}"

# Keep in sync with session-start.sh / bootstrap.sh.
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
    IFS= read -r _venv <"$DATA_DIR/current.path" || _venv=''
    _venv=$(printf '%s' "$_venv" | tr '\\' '/')
    if [ -n "$_venv" ]; then
      PYBIN=$(find_bin "$_venv" python) || PYBIN=''
    fi
  fi
fi

if [ -z "$PYBIN" ]; then
  # Not installed yet (first session). Recall is a nicety; never a nag.
  cat >/dev/null 2>&1
  exit 0
fi

"$PYBIN" -m clara.fastpath.prompt_recall || true
exit 0
