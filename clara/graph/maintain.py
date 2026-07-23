"""
CLARA graph — daily maintenance (rides the opportunistic maintenance pass).

- recompute degree-based ``expandable`` flags (hubs with active degree >
  ``hub_degree`` stop expanding; singletons never re-open),
- mirror belief decay onto edge confidence,
- archive (invalidate) edges whose belief fell below the archival threshold
  or is no longer active,
- prune (status change, never delete) degree-0 nodes with
  ``mention_count <= 1`` older than the stale window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

HUB_DEGREE = 50


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)

# The belief_id ⇄ memories.memory_id join normalizes dashes on both sides so
# it holds regardless of how the Uuid column serializes on this build.
_BELIEF_JOIN = (
    "replace(CAST(m.memory_id AS TEXT), '-', '') = replace(graph_edges.belief_id, '-', '')"
)


async def run_graph_maintenance(
    session: AsyncSession,
    *,
    hub_degree: int = HUB_DEGREE,
    archival_threshold: float = 0.15,
    stale_days: int = 90,
) -> dict[str, int]:
    """Run all graph maintenance steps; returns per-step row counts."""
    from clara.flags import graph_enabled

    if not graph_enabled():
        return {}
    ready = (
        await session.execute(
            sa_text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_nodes'")
        )
    ).first()
    if ready is None:
        return {}

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(sep=" ")
    counts: dict[str, int] = {}

    hubs = await session.execute(
        sa_text(
            """
            UPDATE graph_nodes SET expandable = CASE WHEN coalesce((
                SELECT c FROM (
                    SELECT node_id, COUNT(*) AS c FROM (
                        SELECT src_id AS node_id FROM graph_edges WHERE invalid_at IS NULL
                        UNION ALL
                        SELECT dst_id FROM graph_edges WHERE invalid_at IS NULL
                    ) GROUP BY node_id
                ) deg WHERE deg.node_id = graph_nodes.node_id
            ), 0) > :hub THEN 0 ELSE 1 END,
            updated_at = updated_at
            WHERE status = 'active' AND entity_type NOT IN ('user', 'skill')
            """
        ),
        {"hub": hub_degree},
    )
    counts["expandable_recomputed"] = _rowcount(hubs)

    mirrored = await session.execute(
        sa_text(
            f"""
            UPDATE graph_edges SET confidence = (
                SELECT m.confidence FROM memories m
                WHERE {_BELIEF_JOIN} AND m.status = 'active'
            )
            WHERE invalid_at IS NULL AND belief_id IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM memories m
                WHERE {_BELIEF_JOIN} AND m.status = 'active'
                  AND m.confidence <> graph_edges.confidence
              )
            """
        )
    )
    counts["confidence_mirrored"] = _rowcount(mirrored)

    orphaned = await session.execute(
        sa_text(
            f"""
            UPDATE graph_edges SET invalid_at = :now,
                metadata = json_set(coalesce(metadata, '{{}}'),
                                    '$.invalidated_by', 'belief_inactive')
            WHERE invalid_at IS NULL AND belief_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM memories m WHERE {_BELIEF_JOIN} AND m.status = 'active'
              )
            """
        ),
        {"now": now_iso},
    )
    counts["belief_inactive_archived"] = _rowcount(orphaned)

    archived = await session.execute(
        sa_text(
            """
            UPDATE graph_edges SET invalid_at = :now,
                metadata = json_set(coalesce(metadata, '{}'),
                                    '$.invalidated_by', 'decay_archived')
            WHERE invalid_at IS NULL AND confidence < :threshold
            """
        ),
        {"now": now_iso, "threshold": archival_threshold},
    )
    counts["decay_archived"] = _rowcount(archived)

    cutoff = (now - timedelta(days=stale_days)).isoformat(sep=" ")
    pruned = await session.execute(
        sa_text(
            """
            UPDATE graph_nodes SET status = 'pruned', updated_at = :now
            WHERE status = 'active' AND mention_count <= 1
              AND entity_type NOT IN ('user', 'skill')
              AND strftime('%s', coalesce(updated_at, created_at)) < strftime('%s', :cutoff)
              AND node_id NOT IN (
                SELECT src_id FROM graph_edges WHERE invalid_at IS NULL
                UNION
                SELECT dst_id FROM graph_edges WHERE invalid_at IS NULL
              )
            """
        ),
        {"now": now_iso, "cutoff": cutoff},
    )
    counts["nodes_pruned"] = _rowcount(pruned)
    return counts


def maintenance_summary(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "graph: no-op"
