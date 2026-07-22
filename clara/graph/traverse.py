"""
CLARA graph — bounded traversal via a recursive CTE.

Fan-out is capped IN SQL: valid edges are pre-ranked per endpoint with
``ROW_NUMBER() OVER (PARTITION BY src|dst ORDER BY weight*confidence DESC)``
and only rows with rank <= ``fanout`` join the walk, so a 5,000-edge hub
node contributes exactly ``fanout`` edges instead of exploding the frontier.
Expansion continues only from nodes with ``expandable = 1`` (seeds always
expand); cycles are cut by tracking visited edge ids in a path string.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from clara.graph.normalize import normalize_relation

DEFAULT_DEPTH = 2
DEFAULT_FANOUT = 6
DEFAULT_HOP_DECAY = 0.6

_TRAVERSE_SQL = """
WITH RECURSIVE ranked AS (
    SELECT e.edge_id, e.src_id, e.dst_id, e.relation, e.confidence, e.weight,
           e.valid_from, e.invalid_at, e.temporal_precision, e.belief_id,
           ROW_NUMBER() OVER (
               PARTITION BY e.src_id ORDER BY e.weight * e.confidence DESC, e.edge_id
           ) AS rank_out,
           ROW_NUMBER() OVER (
               PARTITION BY e.dst_id ORDER BY e.weight * e.confidence DESC, e.edge_id
           ) AS rank_in
    FROM graph_edges e
    WHERE (
            (:as_of IS NULL AND e.invalid_at IS NULL)
         OR (:as_of IS NOT NULL
             AND strftime('%s', e.valid_from) <= strftime('%s', :as_of)
             AND (e.invalid_at IS NULL
                  OR strftime('%s', e.invalid_at) > strftime('%s', :as_of)))
        )
      AND (:relation IS NULL OR e.relation = :relation)
      AND (:uid IS NULL OR coalesce(e.user_id, '') = :uid)
),
walk (node_id, edge_id, depth, decay_pow, score, path) AS (
    SELECT je.value, NULL, 0, 1.0, 1.0, ''
    FROM json_each(:seeds) AS je
    UNION ALL
    SELECT CASE WHEN r.src_id = w.node_id THEN r.dst_id ELSE r.src_id END,
           r.edge_id,
           w.depth + 1,
           w.decay_pow * :hop_decay,
           w.decay_pow * :hop_decay * r.confidence,
           w.path || r.edge_id || '>'
    FROM walk w
    JOIN graph_nodes n ON n.node_id = w.node_id
    JOIN ranked r
      ON ((r.src_id = w.node_id AND r.rank_out <= :fanout)
       OR (r.dst_id = w.node_id AND r.rank_in <= :fanout))
    WHERE w.depth < :depth
      AND (w.depth = 0 OR n.expandable = 1)
      AND instr(w.path, r.edge_id || '>') = 0
)
SELECT * FROM (
    SELECT r.edge_id, r.src_id, r.dst_id, r.relation, r.confidence, r.weight,
           r.valid_from, r.invalid_at, r.temporal_precision, r.belief_id,
           w.node_id AS reached_id, w.depth, w.score, w.path,
           ROW_NUMBER() OVER (PARTITION BY w.edge_id ORDER BY w.score DESC) AS pick
    FROM walk w
    JOIN ranked r ON r.edge_id = w.edge_id
    WHERE w.edge_id IS NOT NULL
)
WHERE pick = 1
ORDER BY score DESC, depth ASC
LIMIT :lim
"""


async def traverse(
    session: AsyncSession,
    seed_ids: list[str],
    *,
    depth: int = DEFAULT_DEPTH,
    fanout: int = DEFAULT_FANOUT,
    hop_decay: float = DEFAULT_HOP_DECAY,
    as_of: str | None = None,
    relation: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Edges reachable from *seed_ids*, scored ``decay^depth × confidence``."""
    if not seed_ids:
        return []
    rows = (
        await session.execute(
            sa_text(_TRAVERSE_SQL),
            {
                "seeds": json.dumps(list(seed_ids)),
                "depth": max(0, depth),
                "fanout": max(1, fanout),
                "hop_decay": hop_decay,
                "as_of": as_of,
                "relation": normalize_relation(relation) if relation else None,
                "uid": user_id if user_id is not None else None,
                "lim": limit,
            },
        )
    ).mappings().all()
    return [dict(row) for row in rows]


async def bump_traversed(session: AsyncSession, edges: list[dict[str, Any]]) -> None:
    """Traversed-and-kept edges gain weight; their endpoints gain mentions."""
    if not edges:
        return
    edge_ids = sorted({e["edge_id"] for e in edges})
    node_ids = sorted({e["src_id"] for e in edges} | {e["dst_id"] for e in edges})
    edge_ph = ", ".join(f":e{i}" for i in range(len(edge_ids)))
    node_ph = ", ".join(f":n{i}" for i in range(len(node_ids)))
    await session.execute(
        sa_text(f"UPDATE graph_edges SET weight = weight + 0.05 WHERE edge_id IN ({edge_ph})"),
        {f"e{i}": eid for i, eid in enumerate(edge_ids)},
    )
    await session.execute(
        sa_text(
            f"UPDATE graph_nodes SET mention_count = mention_count + 1 "
            f"WHERE node_id IN ({node_ph})"
        ),
        {f"n{i}": nid for i, nid in enumerate(node_ids)},
    )


async def fetch_nodes(
    session: AsyncSession, node_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """node_id → node row for rendering."""
    if not node_ids:
        return {}
    unique = sorted(set(node_ids))
    placeholders = ", ".join(f":n{i}" for i in range(len(unique)))
    rows = (
        await session.execute(
            sa_text(f"SELECT * FROM graph_nodes WHERE node_id IN ({placeholders})"),
            {f"n{i}": nid for i, nid in enumerate(unique)},
        )
    ).mappings().all()
    return {row["node_id"]: dict(row) for row in rows}


async def find_path(
    session: AsyncSession,
    src_id: str,
    dst_id: str,
    *,
    max_hops: int = 4,
    as_of: str | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]] | None:
    """Best path (list of edge rows, in hop order) from src to dst, or None."""
    if src_id == dst_id:
        return []
    rows = await traverse(
        session,
        [src_id],
        depth=max_hops,
        as_of=as_of,
        user_id=user_id,
        limit=500,
    )
    arrivals = [r for r in rows if r["reached_id"] == dst_id]
    if not arrivals:
        return None
    best = min(arrivals, key=lambda r: (r["depth"], -r["score"]))
    edge_ids = [eid for eid in best["path"].split(">") if eid]
    by_id = {r["edge_id"]: r for r in rows}
    path_edges: list[dict[str, Any]] = []
    for eid in edge_ids:
        edge = by_id.get(eid)
        if edge is None:
            row = (
                await session.execute(
                    sa_text("SELECT * FROM graph_edges WHERE edge_id = :eid"),
                    {"eid": eid},
                )
            ).mappings().first()
            edge = dict(row) if row else None
        if edge is None:
            return None
        path_edges.append(dict(edge))
    return path_edges
