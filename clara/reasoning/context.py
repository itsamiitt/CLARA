"""CLARA - Context assembly helpers for reasoning and prompt injection."""

from __future__ import annotations

from clara.retrieval.engine import RetrievalEngine, RetrievalResult, ScoredMemory


def _format_belief(sm: ScoredMemory) -> str:
    c = sm.memory.content
    domain = c.get("domain")
    core = f"{c.get('subject', '?')} {c.get('relation', '?')} {c.get('object', '?')}"
    if c.get("is_negation"):
        core = f"not ({core})"
    line = f"- {core}"
    line += f" (confidence: {sm.memory.confidence:.2f}"
    if domain:
        line += f", domain: {domain}"
    line += ")"
    return line


def _format_event(sm: ScoredMemory) -> str:
    c = sm.memory.content
    ts = sm.memory.created_at.strftime("%Y-%m-%d") if sm.memory.created_at else "?"
    desc = c.get("object", c.get("description", ""))
    subj = c.get("subject", "")
    rel = c.get("relation", "")
    return f"- {ts}: {subj} {rel} {desc}"


def _format_skill(sm: ScoredMemory) -> str:
    c = sm.memory.content
    name = c.get("name", c.get("object", "unnamed skill"))
    return f"- {name} (confidence: {sm.memory.confidence:.2f})"


def _format_world_model(sm: ScoredMemory) -> str:
    c = sm.memory.content
    parts = []
    for key in ("name", "subject", "object"):
        if key in c and c[key]:
            parts.append(c[key])
            break

    props = c.get("properties", {})
    if props and isinstance(props, dict):
        prop_strs = [f"{k}: {v}" for k, v in props.items()]
        parts.append(" | ".join(prop_strs))
    elif c.get("relation") and c.get("object"):
        parts.append(f"{c.get('relation', '')} {c.get('object', '')}")

    return f"- {' | '.join(parts)}" if parts else "- (world model entry)"


def format_context(result: RetrievalResult) -> str:
    """Build the standard memory-context block from a retrieval result."""
    sections: list[str] = ["=== MEMORY CONTEXT ===", ""]

    sections.append("[BELIEFS]")
    if result.beliefs:
        for sm in result.beliefs:
            sections.append(_format_belief(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[WORLD MODEL]")
    if result.world_model:
        for sm in result.world_model:
            sections.append(_format_world_model(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[RECENT EVENTS]")
    if result.events:
        for sm in result.events:
            sections.append(_format_event(sm))
    else:
        sections.append("- (none)")
    sections.append("")

    sections.append("[RELEVANT SKILLS]")
    if result.skills:
        for sm in result.skills:
            sections.append(_format_skill(sm))
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
