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

# Record the active venv in a plain-text pointer beside `current`.
#
# `current` is a symlink here but an NTFS junction in bootstrap.ps1, and the
# Windows hooks read `current.path` whenever the link cannot be resolved. This
# script previously only *read* that file, so installing from Git Bash on
# Windows left it naming a venv that had already been garbage-collected, and
# session-start.ps1 fell back to a dangling path. Writing it keeps both
# platforms' views of "the active venv" in agreement.
#
# The path is written in the native form the Windows scripts expect: under
# MSYS/Cygwin, `cygpath -w` converts /c/... to C:\... — elsewhere the path is
# already native.
write_current_path() {
  _venv=$1
  _data=$2
  _native="$_venv"
  if command -v cygpath >/dev/null 2>&1; then
    _native=$(cygpath -w "$_venv" 2>/dev/null || printf '%s' "$_venv")
  fi
  printf '%s' "$_native" >"$_data/current.path" 2>/dev/null || true
}

# Hash the package sources with $1; empty output on failure.
#
# The venv is keyed on pyproject.toml, which describes *dependencies*. Editing
# clara/*.py does not change that key, so the fast path below used to conclude
# "venv is current" and keep serving a stale copy of the package: users never
# received code-only updates. Dependencies are expensive to rebuild and change
# rarely; the package itself is cheap to reinstall and changes constantly, so
# the two are tracked separately. ~30 ms for ~90 files.
hash_sources() {
  "$1" - "$PLUGIN_ROOT/clara" <<'PYEOF' 2>/dev/null || true
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:12])
PYEOF
}

# Reinstall just the package (not its dependencies) into the active venv.
ensure_current_sources() {
  _venv=$1
  _data=$2
  _vpy=$(find_bin "$_venv" python) || return 0
  _want=$(hash_sources "$_vpy")
  [ -n "$_want" ] || return 0
  _have=''
  [ -f "$_data/source.hash" ] && _have=$(cat "$_data/source.hash" 2>/dev/null || true)
  [ "$_want" = "$_have" ] && return 0
  log 'plugin code changed — refreshing the installed package'
  if ( cd "$PLUGIN_ROOT" && "$_vpy" -m pip install --quiet --no-deps . ) >/dev/null 2>&1; then
    printf '%s' "$_want" >"$_data/source.hash" 2>/dev/null || true
  else
    log 'package refresh failed; continuing with the previously installed code'
  fi
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
    # Before the GC below deletes the previous venv, so the Windows hooks are
    # never left reading a pointer to a directory that no longer exists.
    write_current_path "$VENV" "$DATA"
    # Record what was just installed so the fast path can tell when the
    # package has fallen behind the checkout.
    _vpy_done=$(find_bin "$VENV" python) || _vpy_done=""
    [ -n "$_vpy_done" ] && hash_sources "$_vpy_done" >"$DATA/source.hash" 2>/dev/null
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

CURRENT="$DATA_DIR/current"

# Hash pyproject with $1; empty output on failure.
hash_pyproject() {
  "$1" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:12])' \
    "$PLUGIN_ROOT/pyproject.toml" 2>/dev/null || true
}

# An already-installed venv is self-sufficient: it ships its own interpreter,
# so a healthy install must NOT depend on a system Python still being on PATH.
# (It previously did — the probe below ran first and exited 1, which silently
# disabled memory injection on any machine whose PATH lost Python after
# install, even though the venv was intact.) Try the installed interpreter
# first; only fall back to probing PATH when there is nothing installed yet.
INSTALLED_PY=''
if [ -n "$CURRENT" ] && [ -e "$CURRENT" ]; then
  INSTALLED_PY=$(find_bin "$CURRENT" python 2>/dev/null || true)
fi
if [ -z "$INSTALLED_PY" ] && [ -f "$DATA_DIR/current.path" ]; then
  _ptr=$(cat "$DATA_DIR/current.path" 2>/dev/null || true)
  if [ -n "$_ptr" ] && [ -d "$_ptr" ]; then
    INSTALLED_PY=$(find_bin "$_ptr" python 2>/dev/null || true)
  fi
fi

HASH=''
if [ -n "$INSTALLED_PY" ]; then
  HASH=$(hash_pyproject "$INSTALLED_PY")
  if [ -n "$HASH" ] && find_bin "$DATA_DIR/venv-$HASH" clara-mcp >/dev/null 2>&1; then
    # Installed venv already matches the current pyproject: nothing to build,
    # and no system Python is required to say so.
    PY="$INSTALLED_PY"
  fi
fi

# Python gate: first interpreter >= 3.10 wins (the library's requires-python).
# Only needed when an install/upgrade is actually required.
if [ -z "${PY:-}" ]; then
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
    if [ -n "$INSTALLED_PY" ]; then
      # Installed but stale (pyproject changed) and no system Python to rebuild
      # with: keep serving the existing venv rather than disabling memory.
      log 'no Python >= 3.10 on PATH; continuing with the installed environment (upgrade skipped).'
      _stale_venv=$(dirname "$(dirname "$INSTALLED_PY")")
      if [ ! -e "$DATA_DIR/shim/clara-mcp" ] && [ ! -e "$DATA_DIR/shim/clara-mcp.exe" ]; then
        ensure_shim "$_stale_venv" "$DATA_DIR" || true
      fi
      exit 0
    fi
    # Give a command the user can actually paste, not just a download URL.
    log 'no Python >= 3.10 on PATH (tried python3.13 python3.12 python3.11 python3.10 python3 python).'
    case "$(uname -s 2>/dev/null || echo unknown)" in
      Darwin) log 'install it with:  brew install python@3.12' ;;
      Linux)
        if command -v apt-get >/dev/null 2>&1; then
          log 'install it with:  sudo apt-get install -y python3 python3-venv'
        elif command -v dnf >/dev/null 2>&1; then
          log 'install it with:  sudo dnf install -y python3'
        elif command -v pacman >/dev/null 2>&1; then
          log 'install it with:  sudo pacman -S --noconfirm python'
        else
          log 'install Python 3.10+ with your package manager.'
        fi
        ;;
      *) log 'install Python 3.10+ from https://www.python.org/downloads/' ;;
    esac
    log 'then start a new session — CLARA sets itself up automatically.'
    exit 1
  fi
fi

# Versioned venv: hash of pyproject.toml selects the environment.
if [ -z "$HASH" ]; then
  HASH=$(hash_pyproject "$PY")
fi
if [ -z "$HASH" ]; then
  log "cannot hash $PLUGIN_ROOT/pyproject.toml"
  exit 1
fi
VENV="$DATA_DIR/venv-$HASH"

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
  write_current_path "$VENV" "$DATA_DIR"
  # The venv matches pyproject, but the package inside it may predate the
  # newest clara/*.py — refresh it before declaring the environment ready.
  ensure_current_sources "$VENV" "$DATA_DIR"
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
