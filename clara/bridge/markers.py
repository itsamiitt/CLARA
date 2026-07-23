"""
Marker fence for CLARA-managed sections in native memory files. Stdlib-only.

    <!-- clara:begin v=1 store=global sha=1f2a3b4c5d6e ts=2026-07-23T10:15:00+00:00 -->
    ... generated lines ...
    <!-- clara:end -->

The recorded ``sha`` is the hash of the generated body; when the body on
disk no longer matches it, a human (or Claude) edited inside the fence —
the splice reports a conflict instead of overwriting, and the caller
imports those lines first.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

BEGIN_PREFIX = "<!-- clara:begin"
END_MARKER = "<!-- clara:end -->"
PROVENANCE_RE = re.compile(r"<!--clara:[0-9a-f]{8}-->")

_FENCE_RE = re.compile(
    r"<!-- clara:begin v=(?P<version>\d+) store=(?P<store>[a-z]+) "
    r"sha=(?P<sha>[0-9a-f]{12}) ts=(?P<ts>[^ ]+) -->\n"
    r"(?P<body>.*?)"
    r"<!-- clara:end -->",
    re.DOTALL,
)


def body_sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def render_fence(body: str, *, store: str, ts: str) -> str:
    if body and not body.endswith("\n"):
        body += "\n"
    return (
        f"<!-- clara:begin v=1 store={store} sha={body_sha(body)} ts={ts} -->\n"
        f"{body}"
        f"{END_MARKER}"
    )


@dataclass(frozen=True)
class SpliceResult:
    text: str
    changed: bool
    conflict_body: str | None  # fence body that was hand-edited, else None


def find_fence(text: str) -> re.Match[str] | None:
    return _FENCE_RE.search(text)


def replace_section(text: str, new_body: str, *, store: str, ts: str) -> SpliceResult:
    """Splice the CLARA fence into *text*, touching nothing outside it.

    - No fence yet: append one (with a separating blank line).
    - Fence present, recorded sha matches the on-disk body, new body equal:
      no-op. Different: replace.
    - Fence present but the on-disk body was edited (sha mismatch): DO NOT
      overwrite — return the edited body as ``conflict_body`` so the caller
      imports it, and leave the text unchanged.
    """
    fence = render_fence(new_body, store=store, ts=ts)
    match = find_fence(text)
    if match is None:
        prefix = text
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix.strip():
            prefix += "\n"
        return SpliceResult(text=prefix + fence + "\n", changed=True, conflict_body=None)

    on_disk_body = match.group("body")
    recorded_sha = match.group("sha")
    if body_sha(on_disk_body) != recorded_sha:
        return SpliceResult(text=text, changed=False, conflict_body=on_disk_body)

    if body_sha(new_body if new_body.endswith("\n") or not new_body else new_body + "\n") == recorded_sha:
        return SpliceResult(text=text, changed=False, conflict_body=None)

    new_text = text[: match.start()] + fence + text[match.end():]
    return SpliceResult(text=new_text, changed=True, conflict_body=None)


def strip_fences(text: str) -> str:
    """Text with every CLARA fence removed — the import view of a file."""
    return _FENCE_RE.sub("", text)
