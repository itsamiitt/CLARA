#!/bin/sh
# CLARA plugin MCP launcher: bootstrap the environment, then exec clara-mcp.
#
# stdout discipline: nothing but the MCP protocol ever touches stdout — the
# bootstrap writes to stderr only, and this script's own messages go to
# stderr. On bootstrap exit 3 (background install) we poll up to 20s for the
# server binary before giving up with a pointer at the install log.

set -u

log() { printf 'clara: %s\n' "$*" >&2; }

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

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-$HOME/.clara}/plugin}"

sh "$SCRIPT_DIR/bootstrap.sh"
rc=$?

if [ "$rc" -eq 3 ]; then
  waited=0
  while [ "$waited" -lt 20 ]; do
    if find_bin "$DATA_DIR/current" clara-mcp >/dev/null; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
fi

if BIN=$(find_bin "$DATA_DIR/current" clara-mcp); then
  exec "$BIN"
fi

log "memory server not ready — install still running or failed; see $DATA_DIR/install.log"
exit 1
