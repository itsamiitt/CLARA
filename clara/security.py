"""
CLARA — secret detection for the memory write path.

A memory store that accepts credentials re-injects them into every future
session's context. Policy (``CLARA_SECRET_POLICY``):

- ``reject`` (default) — refuse the save with a clear error. The calling
  model sees the MCP error, rephrases (store a *reference* to the secret's
  location, not the value), and retries. Chosen over silent redaction
  because a redacted memory is silently corrupt.
- ``redact`` — replace the match with ``[REDACTED:<name>]`` and stamp
  ``metadata.redacted``.
- ``off`` — no scanning.

Patterns require the shape of a real credential (fixed prefixes, or an
assignment ``key: value`` form) so prose like "token bucket algorithm"
passes. Stdlib-only.
"""

from __future__ import annotations

import os
import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("api-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[\w-]{10,}\.eyJ[\w-]{10,}\.[\w-]{10,}\b")),
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*\S{8,}"
        ),
    ),
]

REDACTED_TEMPLATE = "[REDACTED:{name}]"


def secret_policy() -> str:
    """Current policy: reject (default) | redact | off."""
    raw = os.environ.get("CLARA_SECRET_POLICY", "reject").strip().lower()
    return raw if raw in {"reject", "redact", "off"} else "reject"


def find_secret(text: str) -> str | None:
    """Name of the first secret pattern matching *text*, or ``None``."""
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            return name
    return None


def redact(text: str) -> tuple[str, list[str]]:
    """Replace every secret match; return ``(clean_text, matched_names)``."""
    matched: list[str] = []
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            matched.append(name)
            text = pattern.sub(REDACTED_TEMPLATE.format(name=name), text)
    return text, matched


class SecretRejected(ValueError):
    """Raised by the save path when content matches a secret pattern."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"refusing to store: content matches secret pattern '{name}'. "
            "Never store credentials — store a reference to where the secret "
            "lives instead (env var name, vault path). "
            "Set CLARA_SECRET_POLICY=redact or =off to change this policy."
        )
        self.pattern_name = name
