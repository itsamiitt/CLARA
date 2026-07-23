"""
CLARA graph — render traversal results as a ``[GRAPH]`` context section.

Triple-shaped lines grouped by seed entity. Invalidated edges get a ``✗``
prefix plus their validity range (shown only when history is relevant —
callers pass them in only for ``include_history``/``as_of`` queries).
Dates from reconstructed edges are prefixed with ``~``. Hard token budget:
whole lines are dropped (lowest score last-in-group first) until the block
fits ``budget_tokens``.
"""

from __future__ import annotations

from typing import Any

GRAPH_TOKEN_BUDGET = 400


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _date(value: Any, *, reconstructed: bool) -> str:
    raw = str(value or "")[:10]
    if not raw:
        return "?"
    return f"~{raw}" if reconstructed else raw


def _display(node: dict[str, Any] | None, node_id: str) -> str:
    if node is None:
        return node_id[:8]
    return str(node.get("display_name") or node.get("canonical_name") or node_id[:8])


def format_edge_line(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    """One triple-shaped line for an edge."""
    src = _display(nodes.get(edge["src_id"]), edge["src_id"])
    dst = _display(nodes.get(edge["dst_id"]), edge["dst_id"])
    reconstructed = edge.get("temporal_precision") == "reconstructed"
    if edge.get("invalid_at"):
        span = (
            f"{_date(edge.get('valid_from'), reconstructed=reconstructed)}"
            f" → {_date(edge.get('invalid_at'), reconstructed=False)}"
        )
        return f"- ✗ {src} {edge['relation']} {dst} ({span})"
    line = f"- {src} {edge['relation']} {dst} ({float(edge.get('confidence') or 0):.2f}"
    if reconstructed:
        line += f", since {_date(edge.get('valid_from'), reconstructed=True)}"
    line += ")"
    return line


def render_graph_section(
    groups: list[tuple[str, list[dict[str, Any]]]],
    nodes: dict[str, dict[str, Any]],
    *,
    budget_tokens: int = GRAPH_TOKEN_BUDGET,
) -> str:
    """Build the ``[GRAPH]`` block from (seed_display, edges) groups.

    Edges within a group should arrive score-sorted; trimming removes the
    tail line of the largest group until the block fits the budget.
    """
    sections: list[tuple[str, list[str]]] = []
    for seed_display, edges in groups:
        lines = [format_edge_line(edge, nodes) for edge in edges]
        if lines:
            sections.append((seed_display, lines))
    if not sections:
        return ""

    def _assemble() -> str:
        parts = ["[GRAPH]"]
        for seed_display, lines in sections:
            parts.append(f"{seed_display}:")
            parts.extend(lines)
        return "\n".join(parts)

    block = _assemble()
    while _approx_tokens(block) > budget_tokens:
        largest = max(sections, key=lambda item: len(item[1]))
        if len(largest[1]) == 0:
            break
        largest[1].pop()
        sections = [(name, lines) for name, lines in sections if lines]
        if not sections:
            return ""
        block = _assemble()
    return block
