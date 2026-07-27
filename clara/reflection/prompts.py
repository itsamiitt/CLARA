"""Prompt helpers for reflection and insight synthesis."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clara.reflection.pipeline import PatternCandidate


DEFAULT_REFLECTION_SYSTEM_PROMPT = (
    "You summarize recurring memory patterns into a single grounded insight. "
    "Stay faithful to the evidence. Do not invent facts beyond the supplied pattern."
)


def build_reflection_prompt(
    pattern: PatternCandidate,
    *,
    evidence_lines: Iterable[str],
) -> str:
    evidence_block = "\n".join(f"- {line}" for line in evidence_lines) or "- (none)"
    return (
        "Summarize this memory pattern as one concise sentence.\n\n"
        f"Pattern type: {pattern.pattern_type}\n"
        f"Candidate belief: {pattern.subject} {pattern.relation} {pattern.object}\n"
        f"Occurrences: {pattern.count}\n"
        "Evidence:\n"
        f"{evidence_block}\n"
    )


def fallback_reflection_text(pattern: PatternCandidate) -> str:
    if pattern.pattern_type == "recurring_entity":
        return (
            f"Reflection: {pattern.subject} appears repeatedly in recent memory "
            f"activity ({pattern.count} occurrences)."
        )
    if pattern.pattern_type == "repeated_event_type":
        verb = pattern.relation.removeprefix("repeatedly_")
        return (
            f"Reflection: {pattern.subject} repeatedly {verb} "
            f"{pattern.object} ({pattern.count} occurrences)."
        )
    if pattern.pattern_type == "skill_generalization":
        return (
            f"Reflection: {pattern.subject} consistently demonstrates {pattern.object} "
            f"({pattern.count} observations)."
        )
    return (
        f"Reflection: {pattern.subject} {pattern.relation} {pattern.object} "
        f"({pattern.count} supporting memories)."
    )
