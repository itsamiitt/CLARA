#!/bin/sh
# CLARA plugin bootstrap: ensure a virtualenv with clara-memory[mcp] exists.
#
# Exit codes: 0 = environment ready · 1 = unrecoverable (no Python 3.10+)
#             3 = install running in the background, try again later.
# stdout discipline: this script writes ONLY to stderr — callers own stdout.
#
# Layout under $CLAUDE_PLUGIN_DATA (default ~/.clara/plugin):
#   venv-<hash>/  one venv per pyproject.toml content hash (12 hex chars)
#   current       symlink to the active venv (atomic swap via ln -sfn)
#   shim/         stable clara-mcp entry the MCP server config spawns
#   .installing   flag while a background install runs (stale after 15 min)
#   .lock/        mkdir-based mutex so concurrent callers spawn one install
#   install.log   background install output
#
# Python floor: 3.10, matching pyproject.toml requires-python — the library,
# CI matrix, and this gate must agree (tests/test_installation_defaults.py).

set -u

log() { printf 'clara: %s\n' "$*" >&2; }

# Venv binary layouts: POSIX bin/, Windows Scripts/. macOS, Linux, WSL, and
# native Windows are all first-class (native Windows uses the .ps1 scripts;
# this bash path also runs under Git Bash). See README "Supported platforms".
# Keep in sync with clara-mcp-launch.sh / session-start.sh.
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
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(dirname -- "$SCRIPT_DIR")}"
DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-${HOME:-/tmp}/.clara}/plugin}"

# Keep $DATA/shim/clara-mcp pointing at the active venv's real binary. The
# plugin's mcpServers command spawns this path directly (no shell, no
# bootstrap subshell), so it must be refreshed whenever the venv moves.
ensure_shim() {
  _venv=$1
  _data=$2
  _bin=$(find_bin "$_venv" clara-mcp) || return 1
  mkdir -p "$_data/shim" 2>/dev/null || return 1
  case "$_bin" in
    *.exe) cp -f "$_bin" "$_data/shim/clara-mcp.exe" 2>/dev/null || return 1 ;;
    *) ln -sfn "$_bin" "$_data/shim/clara-mcp" 2>/dev/null \
         || cp -f "$_bin" "$_data/shim/clara-mcp" 2>/dev/null || return 1 ;;
  esac
  return 0
}

# ---------------------------------------------------------------------------
# Detached worker (re-invoked by the spawn below; stdout goes to install.log)
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--install-worker" ]; then
  PY="${CLARA_BS_PY:?}"
  VENV="${CLARA_BS_VENV:?}"
  ROOT="${CLARA_BS_ROOT:?}"
  DATA="${CLARA_BS_DATA:?}"
  FLAG="$DATA/.installing"
  LOCK="$DATA/.lock"
  CURRENT="$DATA/current"

  echo "=== clara install started: $(date) ==="
  echo "python: $PY  venv: $VENV"
  status=1
  rm -rf "$VENV"
  # Install target: NEVER pass "$ROOT[mcp]" to pip. On native Windows under
  # Git Bash, $ROOT is an MSYS path (/c/Users/...) that the venv's *Windows*
  # Python does not recognize as a filesystem path — pip then parses it as a
  # PEP 508 requirement and fails ("Invalid requirement: Expected package
  # name"). Installing from inside $ROOT with the relative ".[mcp]" spec
  # resolves via the OS on every platform.
  if command -v uv >/dev/null 2>&1; then
    if uv venv --python "$PY" "$VENV"; then
      VPY=$(find_bin "$VENV" python) \
        && ( cd "$ROOT" && uv pip install --python "$VPY" ".[mcp]" ) \
        && status=0
    fi
  else
    if "$PY" -m venv "$VENV"; then
      VPY=$(find_bin "$VENV" python) \
        && "$VPY" -m pip install --quiet --upgrade pip \
        && ( cd "$ROOT" && "$VPY" -m pip install --quiet ".[mcp]" ) \
        && status=0
    fi
  fi

  if [ "$status" -eq 0 ] && find_bin "$VENV" clara-mcp >/dev/null; then
    ln -sfn "$VENV" "$CURRENT"
    ensure_shim "$VENV" "$DATA" || echo "shim refresh failed (non-fatal)"
    # GC: keep the two newest venvs (the one just installed + one fallback).
    # venv-* basenames are fixed-format hex, so parsing ls -dt is safe here.
    # shellcheck disable=SC2012
    ls -dt "$DATA"/venv-*/ 2>/dev/null | tail -n +3 | while IFS= read -r old_venv; do
      rm -rf "$old_venv"
    done
    echo "=== clara install complete: $(date) ==="
  else
    status=1
    echo "=== clara install FAILED (see messages above): $(date) ==="
  fi
  rm -f "$FLAG"
  rmdir "$LOCK" 2>/dev/null || true
  exit "$status"
fi

# ---------------------------------------------------------------------------
# Foreground path
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR" 2>/dev/null || {
  log "cannot create data dir $DATA_DIR"
  exit 1
}

# Python gate: first interpreter >= 3.10 wins (the library's requires-python).
PY=''
for _cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 \
    && "$_cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
      >/dev/null 2>&1; then
    PY="$_cand"
    break
  fi
done
if [ -z "$PY" ]; then
  log 'no Python >= 3.10 on PATH (tried python3.13 python3.12 python3.11 python3.10 python3 python).'
  log 'install Python 3.10+ (https://www.python.org/downloads/) and start a new session.'
  exit 1
fi

# Versioned venv: hash of pyproject.toml selects the environment.
HASH=$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:12])' "$PLUGIN_ROOT/pyproject.toml" 2>/dev/null || true)
if [ -z "$HASH" ]; then
  log "cannot hash $PLUGIN_ROOT/pyproject.toml"
  exit 1
fi
VENV="$DATA_DIR/venv-$HASH"
CURRENT="$DATA_DIR/current"

# Fast path: the venv for the current hash is ready. Heal the `current`
# pointer if it is missing or a symlink aimed elsewhere (a non-symlink
# `current` — MSYS fallback copy — is left for the worker to replace).
if find_bin "$VENV" clara-mcp >/dev/null; then
  current_target=$(readlink "$CURRENT" 2>/dev/null || true)
  if [ ! -e "$CURRENT" ]; then
    ln -sfn "$VENV" "$CURRENT" 2>/dev/null || true
  elif [ -n "$current_target" ] && [ "$current_target" != "$VENV" ]; then
    ln -sfn "$VENV" "$CURRENT" 2>/dev/null || true
  fi
  # Heal the MCP shim too (dangling after venv GC, missing on upgrade).
  if [ ! -e "$DATA_DIR/shim/clara-mcp" ] && [ ! -e "$DATA_DIR/shim/clara-mcp.exe" ]; then
    ensure_shim "$VENV" "$DATA_DIR" || true
  elif [ -L "$DATA_DIR/shim/clara-mcp" ] && [ ! -e "$DATA_DIR/shim/clara-mcp" ]; then
    ensure_shim "$VENV" "$DATA_DIR" || true
  fi
  exit 0
fi

# Install needed (first run or pyproject hash change).
FLAG="$DATA_DIR/.installing"
LOCK="$DATA_DIR/.lock"
INSTALL_LOG="$DATA_DIR/install.log"

if [ -e "$FLAG" ]; then
  age=$("$PY" -c 'import os, sys, time; print(int(time.time() - os.path.getmtime(sys.argv[1])))' "$FLAG" 2>/dev/null || printf '0')
  if [ "${age:-0}" -lt 900 ]; then
    exit 3
  fi
  log 'previous install looks stale (>15 min); retrying.'
  rm -f "$FLAG"
  rmdir "$LOCK" 2>/dev/null || true
fi

if ! mkdir "$LOCK" 2>/dev/null; then
  # Another caller is spawning the install right now.
  exit 3
fi
: >"$FLAG"
log "installing the CLARA memory environment in the background (log: $INSTALL_LOG)"

CLARA_BS_PY="$PY" CLARA_BS_VENV="$VENV" CLARA_BS_ROOT="$PLUGIN_ROOT" CLARA_BS_DATA="$DATA_DIR" \
  nohup sh "$SCRIPT_DIR/bootstrap.sh" --install-worker >>"$INSTALL_LOG" 2>&1 &
exit 3
