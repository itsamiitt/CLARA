"""
CLARA — text sanitisation helpers.

Memory content is user/LLM-derived and gets interpolated into the
``=== MEMORY CONTEXT ===`` block that is injected into an LLM system prompt.
Without sanitisation, a stored memory whose text contains newlines, fake
section headers (``[BELIEFS]``), or block delimiters (``=== … ===``) could
inject structure or instructions into the prompt. :func:`sanitize_memory_text`
neutralises those vectors while keeping the text readable.
"""

from __future__ import annotations

import re

# Control characters (including newlines/tabs) are collapsed to spaces so a
# single stored value cannot span multiple lines or inject new bullets/sections.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# CLARA's own structural markers: collapse "===" fences and defang [SECTION] headers.
_FENCE_RE = re.compile(r"={2,}")
_SECTION_RE = re.compile(r"\[(/?[A-Z][A-Z0-9 _-]*)\]")
_WS_RE = re.compile(r"\s{2,}")

DEFAULT_MAX_LEN = 500


def sanitize_memory_text(value: object, *, max_len: int = DEFAULT_MAX_LEN) -> str:
    """Make arbitrary stored content safe to interpolate into a prompt block.

    - Collapses newlines/tabs/control chars to single spaces (no structural injection).
    - Collapses ``===`` fences and turns ``[SECTION]`` markers into ``(SECTION)``.
    - Collapses whitespace runs and truncates to *max_len* characters.
    """
    text = _CONTROL_RE.sub(" ", str(value))
    text = _FENCE_RE.sub("=", text)
    text = _SECTION_RE.sub(r"(\1)", text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text
