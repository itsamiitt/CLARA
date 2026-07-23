"""
CLARA docs — reference extraction from markdown.

Pulls the outbound references a document makes: markdown links, inline-code
repo paths, ``path::symbol`` mentions, URLs, and issue/PR ids. These land in
``doc_refs`` and feed the churn/dead-ref signals.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_SYMBOL_RE = re.compile(r"([\w./\\-]+\.\w{1,6})::(\w+)")
_ISSUE_RE = re.compile(r"(?:^|[\s(])#(\d{1,6})\b")
_PATHISH_RE = re.compile(r"^[\w.@/\\-]+$")
_CODE_FENCE_RE = re.compile(r"^(```|~~~)", re.MULTILINE)

_PATH_SUFFIX_HINTS = (
    ".py", ".md", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml",
    ".toml", ".sh", ".sql", ".rs", ".go", ".java", ".rb", ".css", ".html",
    ".txt", ".cfg", ".ini", ".proto",
)


def _strip_fences(text: str) -> str:
    """Drop fenced code blocks — their contents are examples, not references."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _norm_path(raw: str) -> str:
    cleaned = raw.strip().strip("'\"").replace("\\", "/")
    cleaned = cleaned.split("#", 1)[0]  # drop anchors
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return str(PurePosixPath(cleaned)) if cleaned else ""


def _looks_like_repo_path(value: str) -> bool:
    if not value or " " in value or value.startswith(("http://", "https://", "mailto:")):
        return False
    if not _PATHISH_RE.match(value):
        return False
    return "/" in value or value.lower().endswith(_PATH_SUFFIX_HINTS)


def extract_refs(text: str) -> list[tuple[str, str]]:
    """Return ``(ref_kind, ref_value)`` pairs found in the markdown source.

    Kinds: ``file`` (repo-relative path), ``doc`` (path ending in .md),
    ``symbol`` (``path::symbol``), ``url``, ``issue``.
    """
    body = _strip_fences(text)
    refs: dict[tuple[str, str], None] = {}

    for match in _MD_LINK_RE.finditer(body):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://")):
            refs[("url", target)] = None
            continue
        path = _norm_path(target)
        if path and _looks_like_repo_path(path):
            kind = "doc" if path.lower().endswith(".md") else "file"
            refs[(kind, path)] = None

    for match in _SYMBOL_RE.finditer(body):
        path = _norm_path(match.group(1))
        refs[("symbol", f"{path}::{match.group(2)}")] = None
        if _looks_like_repo_path(path):
            refs[("file", path)] = None

    for match in _CODE_SPAN_RE.finditer(body):
        candidate = match.group(0).strip("`")
        if "::" in candidate:
            continue  # handled by the symbol pass
        path = _norm_path(candidate)
        if path and _looks_like_repo_path(path):
            kind = "doc" if path.lower().endswith(".md") else "file"
            refs[(kind, path)] = None

    for match in _ISSUE_RE.finditer(body):
        refs[("issue", match.group(1))] = None

    return list(refs.keys())
