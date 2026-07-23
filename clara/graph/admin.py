"""
CLARA graph — administrative operations: node merge and graph export.

Merge is transactional and reversible-auditable: the loser keeps
``status='merged'`` + ``merged_into``, its name becomes an alias of the
winner, edges are re-pointed, and a ``merge_audit`` record inside the
loser's properties lists every re-pointed edge id so the pre-merge state
can be reconstructed. Nothing is deleted.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from clara.graph.resolve import add_alias, resolve_node


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


async def merge_nodes(
    session: AsyncSession, dup_name: str, canonical_name: str
) -> dict[str, Any]:
    """Merge node *dup_name* into *canonical_name*; returns the audit record."""
    loser = await resolve_node(session, dup_name, create=False, bump_mention=False)
    winner = await resolve_node(session, canonical_name, create=False, bump_mention=False)
    if loser is None or winner is None:
        missing = [n for n, r in ((dup_name, loser), (canonical_name, winner)) if r is None]
        return {"merged": False, "missing": missing}
    if loser["node_id"] == winner["node_id"]:
        return {"merged": False, "reason": "both names resolve to the same node"}

    src_edges = [
        str(row[0]) for row in (
            await session.execute(
                sa_text("SELECT edge_id FROM graph_edges WHERE src_id = :nid"),
                {"nid": loser["node_id"]},
            )
        ).all()
    ]
    dst_edges = [
        str(row[0]) for row in (
            await session.execute(
                sa_text("SELECT edge_id FROM graph_edges WHERE dst_id = :nid"),
                {"nid": loser["node_id"]},
            )
        ).all()
    ]
    await session.execute(
        sa_text("UPDATE graph_edges SET src_id = :win WHERE src_id = :lose"),
        {"win": winner["node_id"], "lose": loser["node_id"]},
    )
    await session.execute(
        sa_text("UPDATE graph_edges SET dst_id = :win WHERE dst_id = :lose"),
        {"win": winner["node_id"], "lose": loser["node_id"]},
    )

    audit = {
        "merge_id": uuid.uuid4().hex,
        "merged_at": _now(),
        "winner": winner["node_id"],
        "loser": loser["node_id"],
        "loser_canonical": loser["canonical_name"],
        "repointed_src_edges": src_edges,
        "repointed_dst_edges": dst_edges,
    }
    loser_props = {}
    with contextlib.suppress(ValueError):
        loser_props = json.loads(loser.get("properties") or "{}")
    loser_props["merge_audit"] = audit
    await session.execute(
        sa_text(
            "UPDATE graph_nodes SET status = 'merged', merged_into = :win, "
            "properties = :props, updated_at = :now WHERE node_id = :lose"
        ),
        {"win": winner["node_id"], "props": json.dumps(loser_props),
         "now": _now(), "lose": loser["node_id"]},
    )
    await add_alias(
        session, winner["node_id"], loser["canonical_name"], source="merge"
    )
    # Merging resolves the duplicate — drop the loser from the winner's list.
    winner_props = {}
    with contextlib.suppress(ValueError):
        winner_props = json.loads(winner.get("properties") or "{}")
    dupes = winner_props.get("possible_duplicates")
    if isinstance(dupes, list) and loser["node_id"] in dupes:
        winner_props["possible_duplicates"] = [
            d for d in dupes if d != loser["node_id"]
        ]
        await session.execute(
            sa_text("UPDATE graph_nodes SET properties = :props WHERE node_id = :nid"),
            {"props": json.dumps(winner_props), "nid": winner["node_id"]},
        )
    return {"merged": True, "winner": winner["node_id"], "loser": loser["node_id"],
            "audit": audit}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _sanitize_label(raw: str) -> str:
    """Escape characters that break mermaid/dot syntax: quotes, brackets, newlines."""
    text = str(raw).replace("\n", " ").replace("\r", " ")
    text = text.replace('"', "'").replace("[", "(").replace("]", ")")
    return text


async def export_graph(
    session: AsyncSession, *, format: str = "mermaid", limit: int = 500
) -> str:
    """Render valid edges as mermaid, dot, or json (labels sanitized)."""
    edges = (
        await session.execute(
            sa_text(
                "SELECT e.edge_id, e.relation, e.confidence, "
                "s.display_name AS src, d.display_name AS dst, "
                "s.node_id AS src_id, d.node_id AS dst_id "
                "FROM graph_edges e "
                "JOIN graph_nodes s ON s.node_id = e.src_id "
                "JOIN graph_nodes d ON d.node_id = e.dst_id "
                "WHERE e.invalid_at IS NULL "
                "ORDER BY e.weight * e.confidence DESC LIMIT :lim"
            ),
            {"lim": limit},
        )
    ).mappings().all()

    if format == "json":
        return json.dumps(
            [
                {"edge_id": e["edge_id"], "src": e["src"], "relation": e["relation"],
                 "dst": e["dst"], "confidence": e["confidence"]}
                for e in edges
            ],
            indent=2,
        )

    if format == "dot":
        lines = ["digraph clara {"]
        for e in edges:
            lines.append(
                f'  "{_sanitize_label(e["src"])}" -> "{_sanitize_label(e["dst"])}" '
                f'[label="{_sanitize_label(e["relation"])}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    # mermaid (default)
    lines = ["graph LR"]
    for e in edges:
        src_id = e["src_id"][:8]
        dst_id = e["dst_id"][:8]
        lines.append(
            f'  {src_id}["{_sanitize_label(e["src"])}"] '
            f'-- {_sanitize_label(e["relation"])} --> '
            f'{dst_id}["{_sanitize_label(e["dst"])}"]'
        )
    return "\n".join(lines)
