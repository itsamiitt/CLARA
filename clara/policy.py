"""
CLARA — repo policy file (``clara.yml``).

Loads the team policy that the Claude Code plugin layers (doc curator,
knowledge graph) consult: document tiers, type patterns, staleness windows,
archive behavior, and terminology aliases. The file lives at the repository
root as ``clara.yml`` — not inside ``.clara/`` — so teams review and commit
it like any other config file. ``clara.yml.example`` documents the shape.

Semantics:

- A missing or empty file yields the built-in defaults (``Policy()``).
- Keys replace at the top level, never deep-merge: a file that sets
  ``tiers`` defines the complete tier map.
- Unknown top-level keys are ignored (forward compatibility; that is what
  ``version`` is for).
- Malformed YAML or wrongly-typed values raise ``ValueError`` naming the
  file. The policy later drives archive/git-mv actions, so silently
  substituting defaults after a typo would be dangerous.

Usage::

    from clara.policy import load_policy

    policy = load_policy()           # <cwd>/clara.yml, defaults if absent
    policy = load_policy(repo_root)  # explicit repo root
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default values (keep in sync with clara.yml.example)
# ---------------------------------------------------------------------------

_POLICY_FILENAME = "clara.yml"

_DEFAULT_VERSION = 1
_DEFAULT_ARCHIVE_DIR = "docs/archive"
_DEFAULT_TIERS: dict[str, list[str]] = {
    "T1": ["docs/adr/**", "CODING_STANDARDS.md", "CLAUDE.md"],
    "T3": ["notes/**", "**/*brainstorm*.md"],
}
_DEFAULT_TYPES: dict[str, list[str]] = {
    "plan": ["docs/plans/**", "**/PLAN*.md"],
}
_DEFAULT_STALE_AFTER_DAYS: dict[str, int] = {"plan": 60, "brainstorm": 14}
_DEFAULT_EXCLUDE: list[str] = ["vendor/**"]
_DEFAULT_ARCHIVE_MODE = "git-mv"
_DEFAULT_ALIASES: dict[str, str] = {"pg": "postgresql"}


@dataclass(frozen=True, slots=True)
class Policy:
    """Team policy from ``clara.yml``; every field falls back to a default.

    Attributes:
        version:
            Policy schema version (currently 1).
        archive_dir:
            Directory documents are archived into.
        tiers:
            Tier name -> glob patterns (e.g. T1 = load-bearing docs).
        types:
            Document type -> glob patterns (e.g. plan documents).
        stale_after_days:
            Document type -> days until it counts as stale.
        exclude:
            Glob patterns never scanned or curated.
        archive_mode:
            How archiving is performed (e.g. ``git-mv``).
        aliases:
            Terminology aliases (short form -> canonical form).
    """

    version: int = _DEFAULT_VERSION
    archive_dir: str = _DEFAULT_ARCHIVE_DIR
    tiers: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in _DEFAULT_TIERS.items()}
    )
    types: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in _DEFAULT_TYPES.items()}
    )
    stale_after_days: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_STALE_AFTER_DAYS)
    )
    exclude: list[str] = field(default_factory=lambda: list(_DEFAULT_EXCLUDE))
    archive_mode: str = _DEFAULT_ARCHIVE_MODE
    aliases: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_ALIASES))


# ---------------------------------------------------------------------------
# Value validators (each raises ValueError naming the file and key)
# ---------------------------------------------------------------------------


def _as_int(value: object, *, path: Path, key: str) -> int:
    # bool is an int subclass; a bare `true` in YAML must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: {key} must be an integer, got {value!r}")
    return value


def _as_str(value: object, *, path: Path, key: str, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be a string, got {value!r}")
    if non_empty and not value.strip():
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return value


def _as_repo_relative_dir(value: str, *, path: Path, key: str) -> str:
    """Reject a directory that would escape the repository.

    ``clara.yml`` is committed to the repo, so cloning an untrusted repository
    imports its policy. This value later feeds a ``mkdir -p`` and a ``git mv``,
    so an absolute path or one containing ``..`` must be refused at load time
    rather than at the (also-guarded) move site.
    """
    cleaned = value.strip().replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part not in ("", ".")]
    if PurePosixPath(cleaned).is_absolute() or Path(cleaned).is_absolute():
        raise ValueError(f"{path}: {key} must be relative to the repository root")
    if ".." in parts:
        raise ValueError(f"{path}: {key} must not contain '..'")
    if not parts:
        raise ValueError(f"{path}: {key} must name a directory")
    return "/".join(parts)


def _as_str_list(value: object, *, path: Path, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {key} must be a list of strings, got {value!r}")
    return list(value)


def _as_pattern_map(value: object, *, path: Path, key: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be a mapping of name -> pattern list")
    return {
        _as_str(name, path=path, key=key): _as_str_list(patterns, path=path, key=f"{key}.{name}")
        for name, patterns in value.items()
    }


def _as_int_map(value: object, *, path: Path, key: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be a mapping of name -> integer")
    return {
        _as_str(name, path=path, key=key): _as_int(days, path=path, key=f"{key}.{name}")
        for name, days in value.items()
    }


def _as_str_map(value: object, *, path: Path, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be a mapping of string -> string")
    return {
        _as_str(name, path=path, key=key): _as_str(alias, path=path, key=f"{key}.{name}")
        for name, alias in value.items()
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_policy(root: str | os.PathLike[str] | None = None) -> Policy:
    """Load ``clara.yml`` from ``root`` (default: cwd); defaults when absent."""
    base = Path(root) if root is not None else Path.cwd()
    path = base / _POLICY_FILENAME
    if not path.exists():
        return Policy()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return Policy()
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

    kwargs: dict[str, Any] = {}
    if "version" in raw:
        kwargs["version"] = _as_int(raw["version"], path=path, key="version")
    if "archive_dir" in raw:
        kwargs["archive_dir"] = _as_repo_relative_dir(
            _as_str(raw["archive_dir"], path=path, key="archive_dir", non_empty=True),
            path=path,
            key="archive_dir",
        )
    if "tiers" in raw:
        kwargs["tiers"] = _as_pattern_map(raw["tiers"], path=path, key="tiers")
    if "types" in raw:
        kwargs["types"] = _as_pattern_map(raw["types"], path=path, key="types")
    if "stale_after_days" in raw:
        kwargs["stale_after_days"] = _as_int_map(
            raw["stale_after_days"], path=path, key="stale_after_days"
        )
    if "exclude" in raw:
        kwargs["exclude"] = _as_str_list(raw["exclude"], path=path, key="exclude")
    if "archive_mode" in raw:
        kwargs["archive_mode"] = _as_str(
            raw["archive_mode"], path=path, key="archive_mode", non_empty=True
        )
    if "aliases" in raw:
        kwargs["aliases"] = _as_str_map(raw["aliases"], path=path, key="aliases")
    return Policy(**kwargs)
