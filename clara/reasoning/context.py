"""CLARA - Context assembly helpers for reasoning and prompt injection."""

from __future__ import annotations

from typing import Any

from clara.core.text import sanitize_memory_text as _s
from clara.retrieval.engine import RetrievalEngine, RetrievalResult, ScoredMemory

# Mirrors clara.fastpath.context.RATIONALE_MAX_LEN — a parity test keeps the
# two renderers in lockstep.
RATIONALE_MAX_LEN = 120


def _rationale(sm: ScoredMemory) -> str:
    """The saved reasoning for a belief, or "" when there is none.

    BeliefMemory stores the caller's description as ``metadata.evidence[0]``,
    so the *why* behind a decision never appears in ``content``.
    """
    meta = sm.memory.metadata_ or {}
    evidence = meta.get("evidence") if isinstance(meta, dict) else None
    if not isinstance(evidence, list) or not evidence:
        return ""
    first = evidence[0]
    if not isinstance(first, dict):
        return ""
    return _s(first.get("text", ""), max_len=RATIONALE_MAX_LEN)


def _format_belief(sm: ScoredMemory) -> str:
    c = sm.memory.content
    domain = c.get("domain")
    core = f"{_s(c.get('subject', '?'))} {_s(c.get('relation', '?'))} {_s(c.get('object', '?'))}"
    if c.get("is_negation"):
        core = f"not ({core})"
    line = f"- {core}"
    line += f" (confidence: {sm.memory.confidence:.2f}"
    if domain:
        line += f", domain: {_s(domain)}"
    line += ")"
    rationale = _rationale(sm)
    # Skip a rationale that merely restates the triple.
    if rationale and rationale.lower() not in core.lower():
        line += f" — {rationale}"
    return line


def _format_event(sm: ScoredMemory) -> str:
    c = sm.memory.content
    ts = sm.memory.created_at.strftime("%Y-%m-%d") if sm.memory.created_at else "?"
    desc = _s(c.get("object", c.get("description", "")))
    subj = _s(c.get("subject", ""))
    rel = _s(c.get("relation", ""))
    return f"- {ts}: {subj} {rel} {desc}"


def _format_skill(sm: ScoredMemory) -> str:
    c = sm.memory.content
    name = _s(c.get("name", c.get("object", "unnamed skill")))
    return f"- {name} (confidence: {sm.memory.confidence:.2f})"


def _format_world_model(sm: ScoredMemory) -> str:
    c = sm.memory.content
    parts = []
    for key in ("name", "subject", "object"):
        if key in c and c[key]:
            parts.append(_s(c[key]))
            break

    props = c.get("properties", {})
    if props and isinstance(props, dict):
        prop_strs = [f"{_s(k)}: {_s(v)}" for k, v in props.items()]
        parts.append(" | ".join(prop_strs))
    elif c.get("relation") and c.get("object"):
        parts.append(f"{_s(c.get('relation', ''))} {_s(c.get('object', ''))}")

    return f"- {' | '.join(parts)}" if parts else "- (world model entry)"


def format_context(
    result: RetrievalResult, foreign_ids: set[str] | None = None
) -> str:
    """Build the standard memory-context block from a retrieval result.

    *foreign_ids* are memories saved while working in a different repository;
    their lines are labeled so the reader cannot mistake another project's
    fact for this one's. Relevance order is untouched — an explicit query's
    ranking should win — the label only supplies provenance.
    """
    marked = foreign_ids or set()

    def _line(sm: Any, formatter: Any) -> str:
        line = str(formatter(sm))
        if str(sm.memory.memory_id) in marked:
            line += "  [from another project]"
        return line

    sections: list[str] = ["=== MEMORY CONTEXT ===", ""]

    sections.append("[BELIEFS]")
    if result.beliefs:
        for sm in result.beliefs:
            sections.append(_line(sm, _format_belief))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[WORLD MODEL]")
    if result.world_model:
        for sm in result.world_model:
            sections.append(_line(sm, _format_world_model))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[RECENT EVENTS]")
    if result.events:
        for sm in result.events:
            sections.append(_line(sm, _format_event))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[RELEVANT SKILLS]")
    if result.skills:
        for sm in result.skills:
            sections.append(_line(sm, _format_skill))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("=== END MEMORY CONTEXT ===")
    return "\n".join(sections)


class ContextAssembler:
    """Thin wrapper around retrieval + prompt-context formatting."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self._retriever = retrieval_engine

    async def retrieve(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        return await self._retriever.search(query, top_k=top_k, user_id=user_id)

    async def build(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int = 8,
    ) -> str:
        result = await self.retrieve(query, user_id=user_id, top_k=top_k)
        return format_context(result)

    async def assemble(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int = 8,
    ) -> tuple[RetrievalResult, str]:
        result = await self.retrieve(query, user_id=user_id, top_k=top_k)
        return result, format_context(result)
