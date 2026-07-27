#!/bin/sh
# CLARA plugin SessionStart hook: inject the memory context block.
#
# Contract: ALWAYS exits 0 — memory must never block a session. stdout is the
# context Claude sees; stderr carries diagnostics. Runs exactly one fastpath
# invocation and never execs before it.

set -u

# Keep in sync with bootstrap.sh.
find_bin() {
  for _cand in "$1/bin/$2" "$1/Scripts/$2" "$1/Scripts/$2.exe"; do
    if [ -x "$_cand" ]; then
      printf '%s' "$_cand"
      return 0
    fi
  done
  return 1
}

# ${HOME:-/tmp} guards against `set -u` aborting when HOME is unset (rare, but
# then the hook must still exit 0 per its contract, not die "unbound variable").
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-${HOME:-/tmp}/.clara}/plugin}"
BASE="${CLARA_HOME:-${HOME:-/tmp}/.clara}"

# Session-cwd hint: lets the long-lived MCP server resolve the store for THIS
# session's directory even when its own process cwd is stale. Best-effort.
if [ -n "${CLAUDE_SESSION_ID:-}" ]; then
  mkdir -p "$BASE/session-cwd" 2>/dev/null \
    && printf '%s' "$PWD" >"$BASE/session-cwd/$CLAUDE_SESSION_ID" 2>/dev/null \
    || true
fi

sh "$SCRIPT_DIR/bootstrap.sh"
rc=$?

if [ "$rc" -eq 3 ]; then
  # A previous attempt that failed changes what is true here. Retrying is
  # right -- most failures are transient -- but reporting only "available next
  # session" after a recorded failure tells someone with no network to keep
  # waiting for something that will not arrive. Say both: it is retrying, and
  # the last attempt failed.
  if [ -f "$DATA_DIR/install.failed" ]; then
    printf '%s\n' "CLARA is retrying its background install (the last attempt failed on $(cat "$DATA_DIR/install.failed" 2>/dev/null))."
    printf '%s\n' "If this repeats, the log says why: $DATA_DIR/install.log"
    printf '%s\n' "Most often this is no network access to PyPI, or a proxy that blocks it."
  else
    printf '%s\n' 'CLARA is installing in the background — memory will be available next session.'
  fi
  exit 0
fi

# Do NOT bail on a non-zero bootstrap: if a usable venv exists, inject from it
# anyway. Bootstrap failing (e.g. no system Python on PATH for an upgrade) is
# not a reason to withhold memory that is already installed and working.
# Bootstrap has already explained itself on stderr; the session is never blocked.
PYBIN=''
if ! PYBIN=$(find_bin "$DATA_DIR/current" python); then
  PYBIN=''
  if [ -f "$DATA_DIR/current.path" ]; then
    _ptr=$(cat "$DATA_DIR/current.path" 2>/dev/null || true)
    if [ -n "$_ptr" ] && [ -d "$_ptr" ]; then
      PYBIN=$(find_bin "$_ptr" python 2>/dev/null || true)
    fi
  fi
fi

if [ -n "$PYBIN" ]; then
  "$PYBIN" -m clara.fastpath.context --cwd "$PWD" || true
fi
exit 0
