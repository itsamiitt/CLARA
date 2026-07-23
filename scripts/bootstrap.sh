#!/bin/sh
# CLARA plugin bootstrap: ensure a virtualenv with clara-memory[mcp] exists.
#
# Exit codes: 0 = environment ready · 1 = unrecoverable (no Python 3.11+)
#             3 = install running in the background, try again later.
# stdout discipline: this script writes ONLY to stderr — callers own stdout.
#
# Layout under $CLAUDE_PLUGIN_DATA (default ~/.clara/plugin):
#   venv-<hash>/  one venv per pyproject.toml content hash (12 hex chars)
#   current       symlink to the active venv (atomic swap via ln -sfn)
#   .installing   flag while a background install runs (stale after 15 min)
#   .lock/        mkdir-based mutex so concurrent callers spawn one install
#   install.log   background install output

set -u

log() { printf 'clara: %s\n' "$*" >&2; }

# Venv binary layouts: POSIX bin/, Windows Scripts/ (best-effort — the plugin
# targets macOS/Linux/WSL; see README "Supported platforms").
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
DATA_DIR="${CLAUDE_PLUGIN_DATA:-${CLARA_HOME:-$HOME/.clara}/plugin}"

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
  if command -v uv >/dev/null 2>&1; then
    if uv venv --python "$PY" "$VENV"; then
      VPY=$(find_bin "$VENV" python) \
        && uv pip install --python "$VPY" "${ROOT}[mcp]" \
        && status=0
    fi
  else
    if "$PY" -m venv "$VENV"; then
      VPY=$(find_bin "$VENV" python) \
        && "$VPY" -m pip install --quiet --upgrade pip \
        && "$VPY" -m pip install --quiet "${ROOT}[mcp]" \
        && status=0
    fi
  fi

  if [ "$status" -eq 0 ] && find_bin "$VENV" clara-mcp >/dev/null; then
    ln -sfn "$VENV" "$CURRENT"
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

# Python gate: first interpreter >= 3.11 wins.
PY=''
for _cand in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 \
    && "$_cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
      >/dev/null 2>&1; then
    PY="$_cand"
    break
  fi
done
if [ -z "$PY" ]; then
  log 'no Python >= 3.11 on PATH (tried python3.13 python3.12 python3.11 python3 python).'
  log 'install Python 3.11+ (https://www.python.org/downloads/) and start a new session.'
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
