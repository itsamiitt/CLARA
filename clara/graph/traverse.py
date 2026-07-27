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

# Never write a bind name with its colon inside a comment in the template
# below: SQLAlchemy's text() does not parse SQL, so it binds the name and
# substitutes the "?" where SQLite cannot see it, and every call then fails
# with "Incorrect number of bindings supplied".
#
# The validity predicate is substituted at import rather than branched on at
# runtime. It has to be a *static* `invalid_at IS NULL` for the as-of-now case,
# because ix_graph_edges_src_valid / _dst_valid are PARTIAL indexes with
# exactly that WHERE clause: with the check buried inside an OR-ed as-of
# expression SQLite cannot prove it holds and scans the whole edge table.

_VALIDITY_NOW = "e.invalid_at IS NULL"

_VALIDITY_AS_OF = """(
            strftime('%s', e.valid_from) <= strftime('%s', :as_of)
        AND (e.invalid_at IS NULL
             OR strftime('%s', e.invalid_at) > strftime('%s', :as_of))
    )"""

_EDGE_COLUMNS = """e.edge_id, e.src_id, e.dst_id, e.relation, e.confidence,
           e.weight, e.valid_from, e.invalid_at, e.temporal_precision,
           e.belief_id"""

_TRAVERSE_TEMPLATE = """
WITH RECURSIVE candidates (node_id, depth) AS (
    -- Nodes within `depth` hops of the seeds; reachability only, no window
    -- functions and no fanout. Its sole job is to bound the set that `ranked`
    -- below has to sort. Without it every traversal window-ranked the WHOLE
    -- edge table twice, so a five-neighbour seed cost 1.2 s once the graph
    -- reached 100k edges -- time spent ordering edges the walk can never
    -- reach.
    --
    -- Safe to over-approximate: dropping the fanout cap and the `expandable`
    -- check here can only make this a superset of what the walk visits, and
    -- ranking a superset cannot change which edges the walk selects.
    --
    -- The two directions are separate recursive terms rather than one
    -- `ON (src = node OR dst = node)` join: a disjunction across two columns
    -- is not index-seekable, so the single-term form rescanned every edge at
    -- every level. SQLite permits multiple recursive terms in the compound.
    SELECT je.value, 0 FROM json_each(:seeds) AS je
    UNION  -- not UNION ALL: dedup keeps a hub from re-expanding
    SELECT e.dst_id, c.depth + 1
    FROM candidates c
    CROSS JOIN graph_edges e ON e.src_id = c.node_id
    WHERE c.depth < :depth AND {validity}
      AND (:relation IS NULL OR e.relation = :relation)
      AND (:uid IS NULL OR coalesce(e.user_id, '') = :uid)
    UNION
    SELECT e.src_id, c.depth + 1
    FROM candidates c
    CROSS JOIN graph_edges e ON e.dst_id = c.node_id
    WHERE c.depth < :depth AND {validity}
      AND (:relation IS NULL OR e.relation = :relation)
      AND (:uid IS NULL OR coalesce(e.user_id, '') = :uid)
),
relevant AS (
    -- Every edge incident to a candidate node, by the same two index-seekable
    -- branches. A candidate keeps ALL of its edges, so its rank_out/rank_in
    -- partition below stays complete and the fanout cut is unchanged; a
    -- non-candidate endpoint may have a partial partition, but the walk only
    -- ever ranks partitions belonging to nodes it actually visits, and every
    -- visited node is a candidate.
    --
    -- CROSS JOIN is load-bearing, not style: it is SQLite's documented way to
    -- pin the outer loop. A materialized CTE carries no row estimate, so with
    -- a plain JOIN the planner drove from graph_edges and probed `candidates`
    -- with a bloom filter -- scanning all 100k edges and taking 763 ms, worse
    -- than the version this replaced. Driving from the handful of candidates
    -- instead turns it into an index seek.
    SELECT {edge_columns}
    FROM candidates c
    CROSS JOIN graph_edges e ON e.src_id = c.node_id
    WHERE {validity}
      AND (:relation IS NULL OR e.relation = :relation)
      AND (:uid IS NULL OR coalesce(e.user_id, '') = :uid)
    UNION  -- an edge between two candidates matches both branches
    SELECT {edge_columns}
    FROM candidates c
    CROSS JOIN graph_edges e ON e.dst_id = c.node_id
    WHERE {validity}
      AND (:relation IS NULL OR e.relation = :relation)
      AND (:uid IS NULL OR coalesce(e.user_id, '') = :uid)
),
ranked AS (
    -- Aliased `rel`, not `e`: `e` means the graph_edges table everywhere else
    -- in this query, and test_traversal_is_index_driven asserts on the plan
    -- that no line scans `e`. Reusing the alias here would make a real full
    -- table scan indistinguishable from this (correct, tiny) scan of a CTE.
    SELECT rel.edge_id, rel.src_id, rel.dst_id, rel.relation, rel.confidence,
           rel.weight, rel.valid_from, rel.invalid_at, rel.temporal_precision,
           rel.belief_id,
           ROW_NUMBER() OVER (
               PARTITION BY rel.src_id
               ORDER BY rel.weight * rel.confidence DESC, rel.edge_id
           ) AS rank_out,
           ROW_NUMBER() OVER (
               PARTITION BY rel.dst_id
               ORDER BY rel.weight * rel.confidence DESC, rel.edge_id
           ) AS rank_in
    FROM relevant rel
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

# Built once at import; the two differ only in the substituted predicate, so
# they cannot drift apart the way two hand-maintained queries would.
_TRAVERSE_SQL_NOW = _TRAVERSE_TEMPLATE.format(
    validity=_VALIDITY_NOW, edge_columns=_EDGE_COLUMNS
)
_TRAVERSE_SQL_AS_OF = _TRAVERSE_TEMPLATE.format(
    validity=_VALIDITY_AS_OF, edge_columns=_EDGE_COLUMNS
)


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
    params: dict[str, Any] = {
        "seeds": json.dumps(list(seed_ids)),
        "depth": max(0, depth),
        "fanout": max(1, fanout),
        "hop_decay": hop_decay,
        "relation": normalize_relation(relation) if relation else None,
        "uid": user_id if user_id is not None else None,
        "lim": limit,
    }
    # The as-of-now variant has no :as_of bind at all, so it must not be passed.
    if as_of is None:
        sql = _TRAVERSE_SQL_NOW
    else:
        sql = _TRAVERSE_SQL_AS_OF
        params["as_of"] = as_of
    rows = (await session.execute(sa_text(sql), params)).mappings().all()
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
