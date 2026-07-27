"""
CLARA — single store resolver shared by every entry point.

One rule, three tiers, first match wins:

1. ``CLARA_DB_PATH`` — explicit override, used verbatim (scope ``explicit``).
2. Project store — ``<git toplevel of anchor>/.clara/clara.db`` **iff the file
   exists** (``clara init --project`` creating it is the opt-in signal).
3. Global store — ``$CLARA_HOME/clara.db`` else ``~/.clara/clara.db``.

Stdlib-only on purpose: the fastpath (SessionStart hook) imports this module,
and the fastpath import-purity test forbids anything heavier. The doc ledger
deliberately does NOT use tier 2 — it always lives in the global store, keyed
by repo_id, so worktrees and clones share one ledger.

``resolve_store(create=False)`` never touches the filesystem beyond ``stat``;
writer paths pass ``create=True`` to mkdir parents and tighten permissions.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_GIT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class StoreResolution:
    """Where the store is (or would be) and how the location was chosen."""

    db_path: Path
    scope: str  # "explicit" | "project" | "global"
    exists: bool
    repo_root: str | None


def git_toplevel(cwd: str) -> str | None:
    """``git rev-parse --show-toplevel`` for *cwd*, or ``None`` outside a repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def global_db_path() -> Path:
    """The global store path ($CLARA_DB_PATH / $CLARA_HOME / ~/.clara)."""
    explicit = os.environ.get("CLARA_DB_PATH")
    if explicit:
        return Path(explicit)
    base = Path(os.environ.get("CLARA_HOME") or (Path.home() / ".clara"))
    return base / "clara.db"


def resolve_store(anchor: str | None = None, *, create: bool = False) -> StoreResolution:
    """Resolve the store for *anchor* (a directory; defaults to cwd).

    ``create=True`` makes the parent directory and applies restrictive
    permissions — writer paths only. ``create=False`` never touches disk.
    """
    anchor = anchor or os.getcwd()
    explicit = os.environ.get("CLARA_DB_PATH")
    root = git_toplevel(anchor)

    if explicit:
        resolution = StoreResolution(
            db_path=Path(explicit),
            scope="explicit",
            exists=Path(explicit).is_file(),
            repo_root=root,
        )
    else:
        project = Path(root or anchor) / ".clara" / "clara.db"
        if project.is_file():
            resolution = StoreResolution(
                db_path=project, scope="project", exists=True, repo_root=root
            )
        else:
            candidate = global_db_path()
            resolution = StoreResolution(
                db_path=candidate,
                scope="global",
                exists=candidate.is_file(),
                repo_root=root,
            )

    if create:
        resolution.db_path.parent.mkdir(parents=True, exist_ok=True)
        secure_store_file(resolution.db_path)
    return resolution


def orphaned_project_stores(anchor: str, resolved: Path) -> list[Path]:
    """Project stores between *anchor* and the git toplevel that are NOT the
    resolved store — the residue of a pre-fix ``clara init --project`` run
    from a subdirectory. Surfaced by ``clara doctor``."""
    orphans: list[Path] = []
    top = git_toplevel(anchor)
    stop = Path(top).resolve() if top else Path(anchor).resolve()
    current = Path(anchor).resolve()
    resolved = resolved.resolve()
    while True:
        candidate = current / ".clara" / "clara.db"
        if candidate.is_file() and candidate.resolve() != resolved:
            orphans.append(candidate)
        if current == stop or current.parent == current:
            break
        current = current.parent
    return orphans


def secure_store_file(path: Path) -> None:
    """Tighten permissions on the store and its WAL/SHM side files.

    POSIX: 0700 on the parent, 0600 on the files. Windows: home-directory
    ACLs are already user-scoped, so this is a no-op unless
    ``CLARA_TIGHTEN_ACLS=1`` requests a best-effort icacls pass. Never fatal.
    """
    try:
        if os.name == "posix":
            os.chmod(path.parent, stat.S_IRWXU)
            for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                if candidate.exists():
                    os.chmod(candidate, stat.S_IRUSR | stat.S_IWUSR)
        elif os.environ.get("CLARA_TIGHTEN_ACLS") == "1" and path.exists():
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r",
                 f"{os.environ.get('USERNAME', 'SYSTEM')}:F"],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
            )
    except OSError as exc:  # permissions are best-effort, never fatal
        print(f"clara: could not tighten permissions on {path}: {exc}", file=sys.stderr)
