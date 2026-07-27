"""
CLARA — stable repository identity.

``repo_id`` returns a 16-hex-character identifier for the repository that
contains ``path``, stable across clones, worktrees, and machines wherever
git can supply an identity:

1. sha256 of the root commit hash (``git rev-list --max-parents=0 HEAD``,
   first line) — worktrees of one repo share the object database, so they
   share this id;
2. else sha256 of the normalized ``remote.origin.url`` (repos with no
   commits yet);
3. else sha256 of the normalized real path (not a git repo, or git absent).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

_ID_HEX_LEN = 16
_GIT_TIMEOUT_S = 10.0

# scp-like syntax: [user@]host:path — host must be at least 2 chars and
# contain no backslash, so Windows drive letters (C:\x, C:/x) fall through
# to plain-path handling instead of parsing "C" as a host.
_SCP_RE = re.compile(r"^(?:[^@/:]+@)?([^:/\\]{2,}):(.*)$")


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:_ID_HEX_LEN]


def _git_output(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
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


def _root_commit(cwd: Path) -> str | None:
    out = _git_output(["rev-list", "--max-parents=0", "HEAD"], cwd)
    if out is None:
        return None
    return out.splitlines()[0].strip() or None


def _origin_url(cwd: Path) -> str | None:
    return _git_output(["config", "--get", "remote.origin.url"], cwd)


def normalize_git_url(url: str) -> str:
    """Normalize a git remote URL to a canonical ``host/path`` string.

    Handles URL syntax (``scheme://[user@]host[:port]/org/repo``) and
    scp-like syntax (``git@host:org/repo``); drops user and port, strips
    trailing ``/`` and ``.git``, and lowercases the result so equivalent
    remotes hash to the same id.
    """
    url = url.strip()
    host = ""
    path = url
    if "://" in url:
        parts = urlsplit(url)
        host = parts.hostname or ""
        path = parts.path
    else:
        match = _SCP_RE.match(url)
        if match:
            host, path = match.group(1), match.group(2)
    path = path.strip("/")
    path = path.removesuffix(".git")
    combined = f"{host}/{path}" if host else path
    return combined.strip("/").lower()


def repo_id(path: str | os.PathLike[str] | None = None) -> str:
    """Return the 16-hex repository id for ``path`` (default: cwd)."""
    where = Path(path) if path is not None else Path.cwd()
    root = _root_commit(where)
    if root is not None:
        return _digest(root)
    origin = _origin_url(where)
    if origin is not None:
        return _digest(normalize_git_url(origin))
    return _digest(os.path.normcase(str(where.resolve())))
