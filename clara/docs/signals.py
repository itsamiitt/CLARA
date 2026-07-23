"""
CLARA docs — deterministic signal collectors.

Every signal is a verifiable fact with a one-sentence explanation (each
collector's docstring IS that sentence). No LLM calls; no repo mutations.

Git access is batched: one ``git log`` parse per scan run feeds age, author,
churn, and merge-evidence for every document, so an incremental single-doc
rescan needs at most one history call plus capped ``git grep`` symbol probes.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clara.policy import Policy

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_S = 30.0
_HISTORY_MAX_COMMITS = 2000
_SYMBOL_PROBE_CAP = 8
_MERGE_ID_CAP = 10
_COMMIT_MARK = "@@CLARA@@"

_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[( |x|X)\]", re.MULTILINE)

# Built-in path conventions (policy globs take precedence).
_BUILTIN_TYPES: list[tuple[str, str, str | None]] = [
    ("docs/adr/**", "adr", "T1"),
    ("**/adr/**", "adr", "T1"),
    ("docs/plans/**", "plan", None),
    ("**/PLAN*.md", "plan", None),
    ("**/*brainstorm*.md", "brainstorm", "T3"),
    ("**/TODO*.md", "todo", None),
    ("README*", "readme", None),
    ("**/README*.md", "readme", None),
    ("docs/spec/**", "spec", None),
    ("**/*audit*.md", "audit", None),
    ("**/*migration*.md", "migration", None),
    ("CODING_STANDARDS.md", "standard", "T1"),
    ("**/CONTRIBUTING*.md", "guide", None),
    ("docs/**", "guide", None),
]


@dataclass
class Commit:
    sha: str
    ts: int
    author: str
    subject: str
    files: set[str] = field(default_factory=set)


@dataclass
class GitHistory:
    """Parsed repo history (newest first), the shared source for git signals."""

    commits: list[Commit] = field(default_factory=list)

    def last_touch(self, rel_path: str) -> Commit | None:
        for commit in self.commits:
            if rel_path in commit.files:
                return commit
        return None

    def commits_touching_since(self, paths: set[str], since_ts: int) -> int:
        return sum(
            1
            for commit in self.commits
            if commit.ts > since_ts and not paths.isdisjoint(commit.files)
        )

    def subjects_blob(self) -> str:
        return "\n".join(commit.subject for commit in self.commits)


def _git(root: str, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def load_history(root: str, max_commits: int = _HISTORY_MAX_COMMITS) -> GitHistory:
    """One batched ``git log --name-only`` parse feeding every git signal."""
    out = _git(
        root,
        [
            "log",
            f"--max-count={max_commits}",
            f"--format={_COMMIT_MARK}%H|%ct|%an|%s",
            "--name-only",
        ],
    )
    history = GitHistory()
    if not out:
        return history
    current: Commit | None = None
    for line in out.splitlines():
        if line.startswith(_COMMIT_MARK):
            parts = line[len(_COMMIT_MARK) :].split("|", 3)
            if len(parts) == 4:
                current = Commit(parts[0], int(parts[1]), parts[2], parts[3])
                history.commits.append(current)
            continue
        if line.strip() and current is not None:
            current.files.add(line.strip().replace("\\", "/"))
    return history


# ---------------------------------------------------------------------------
# Individual signals — the docstring is the one-sentence explanation.
# ---------------------------------------------------------------------------


def age_days(history: GitHistory, rel_path: str, now_ts: float) -> int | None:
    """Days since a commit last touched this document."""
    commit = history.last_touch(rel_path)
    if commit is None:
        return None
    return max(0, int((now_ts - commit.ts) // 86_400))


def last_author(history: GitHistory, rel_path: str) -> str | None:
    """Author of the most recent commit that touched this document."""
    commit = history.last_touch(rel_path)
    return commit.author if commit else None


def churn_divergence(history: GitHistory, rel_path: str, file_targets: set[str]) -> str:
    """Commits touching the doc's referenced files since the doc last changed,
    versus commits touching the doc itself, as ``targets:doc`` (e.g. 47:0)."""
    commit = history.last_touch(rel_path)
    since = commit.ts if commit else 0
    target_commits = history.commits_touching_since(file_targets, since) if file_targets else 0
    doc_commits = history.commits_touching_since({rel_path}, since)
    return f"{target_commits}:{doc_commits}"


def dead_refs(
    root: str, refs: list[tuple[str, str]], *, probe_symbols: bool = True
) -> dict[str, Any]:
    """Referenced repo paths that no longer exist, plus referenced symbols no
    longer found by ``git grep -w``."""
    missing_files: list[str] = []
    for kind, value in refs:
        if kind in ("file", "doc") and not (Path(root) / value).exists():
            missing_files.append(value)
    missing_symbols: list[str] = []
    if probe_symbols:
        symbol_refs = [v for k, v in refs if k == "symbol"][:_SYMBOL_PROBE_CAP]
        for ref in symbol_refs:
            path, _, symbol = ref.partition("::")
            if path and not (Path(root) / path).exists():
                missing_symbols.append(ref)
                continue
            out = _git(root, ["grep", "-lw", "-e", symbol, "--", path or "."])
            if out is not None and not out.strip():
                missing_symbols.append(ref)
    return {
        "files": missing_files[:20],
        "symbols": missing_symbols[:20],
        "count": len(missing_files) + len(missing_symbols),
    }


def checkbox_state(text: str) -> dict[str, Any] | None:
    """Ratio of checked to total markdown task checkboxes in the document."""
    boxes = _CHECKBOX_RE.findall(text)
    if not boxes:
        return None
    checked = sum(1 for box in boxes if box.lower() == "x")
    return {"checked": checked, "total": len(boxes),
            "ratio": round(checked / len(boxes), 3)}


def merge_evidence(history: GitHistory, refs: list[tuple[str, str]]) -> dict[str, Any] | None:
    """Issue/PR ids mentioned in the doc that appear in merged commit subjects."""
    ids = [value for kind, value in refs if kind == "issue"][:_MERGE_ID_CAP]
    if not ids:
        return None
    subjects = history.subjects_blob()
    merged = [issue_id for issue_id in ids if f"#{issue_id}" in subjects]
    return {"ids": ids, "merged": merged}


def _match(pattern: str, rel_path: str) -> bool:
    # fnmatch's `*` already crosses `/`, so `**` collapses to `*`.
    return fnmatch.fnmatch(rel_path, pattern.replace("**", "*"))


def path_convention(rel_path: str, policy: Policy) -> dict[str, Any]:
    """Document type and tier implied by its path, from clara.yml plus
    built-in conventions (docs/adr/** → adr/T1, **/PLAN*.md → plan, ...)."""
    doc_type: str | None = None
    tier: str | None = None
    source = "heuristic"
    for candidate_type, patterns in policy.types.items():
        if any(_match(p, rel_path) for p in patterns):
            doc_type = candidate_type
            source = "policy"
            break
    for candidate_tier, patterns in policy.tiers.items():
        if any(_match(p, rel_path) for p in patterns):
            tier = candidate_tier
            source = "policy"
            break
    if doc_type is None:
        for pattern, builtin_type, builtin_tier in _BUILTIN_TYPES:
            if _match(pattern, rel_path):
                doc_type = builtin_type
                if tier is None:
                    tier = builtin_tier
                break
    return {"doc_type": doc_type or "unknown", "tier": tier or "T2", "source": source}


def frontmatter(text: str) -> dict[str, Any] | None:
    """The document's own ``status:/type:/supersedes:`` frontmatter keys —
    the highest-priority heuristic because the author stated them."""
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    end = None
    for i in range(1, min(len(lines), 60)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return None
    try:
        import yaml

        parsed = yaml.safe_load("\n".join(lines[1:end]))
    except Exception:  # noqa: BLE001 — a broken frontmatter block is just absent
        return None
    if not isinstance(parsed, dict):
        return None
    keep = {
        key: parsed[key]
        for key in ("status", "type", "supersedes", "tier")
        if key in parsed and isinstance(parsed[key], (str, int, float))
    }
    return keep or None


def collect_signals(
    *,
    root: str,
    rel_path: str,
    text: str,
    refs: list[tuple[str, str]],
    history: GitHistory,
    policy: Policy,
    now_ts: float,
    probe_symbols: bool = True,
) -> dict[str, Any]:
    """Assemble all per-document signals into one JSON-serializable dict."""
    file_targets = {value for kind, value in refs if kind in ("file", "doc")}
    signals: dict[str, Any] = {
        "age_days": age_days(history, rel_path, now_ts),
        "last_author": last_author(history, rel_path),
        "churn_divergence": churn_divergence(history, rel_path, file_targets),
        "dead_refs": dead_refs(root, refs, probe_symbols=probe_symbols),
        "path_convention": path_convention(rel_path, policy),
    }
    for name, value in (
        ("checkbox_state", checkbox_state(text)),
        ("merge_evidence", merge_evidence(history, refs)),
        ("frontmatter", frontmatter(text)),
    ):
        if value is not None:
            signals[name] = value
    return signals
