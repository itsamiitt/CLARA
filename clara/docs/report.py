"""
CLARA docs — status lookups and the rot report.

The report is PROPOSALS ONLY: it names stale documents, dead-ref documents,
duplicate clusters, and archive candidates, and never mutates anything.
Tier T0/T1 documents are never proposed for archiving.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from clara.policy import Policy

_PROTECTED_TIERS = {"T0", "T1"}


def _signals(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(row["signals"] or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_status(
    conn: sqlite3.Connection, repo: str, path_or_query: str
) -> dict[str, Any] | None:
    """Standing of one document: tier, lifecycle, named signals, chain."""
    needle = path_or_query.replace("\\", "/").strip()
    row = conn.execute(
        "SELECT * FROM doc_registry WHERE repo_id = ? AND rel_path = ? "
        "AND invalid_at IS NULL",
        (repo, needle),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM doc_registry WHERE repo_id = ? AND invalid_at IS NULL "
            "AND rel_path LIKE ? ORDER BY length(rel_path) LIMIT 1",
            (repo, f"%{needle}%"),
        ).fetchone()
    if row is None:
        return None

    sig = _signals(row)
    named: list[str] = []
    if sig.get("age_days") is not None:
        named.append(f"age {sig['age_days']}d")
    churn = sig.get("churn_divergence")
    if churn and not churn.startswith("0:"):
        named.append(f"churn {churn}")
    dead = sig.get("dead_refs") or {}
    if dead.get("count"):
        named.append(f"{dead['count']} dead refs")
    boxes = sig.get("checkbox_state")
    if boxes:
        named.append(f"checkboxes {boxes['checked']}/{boxes['total']}")
    if sig.get("duplicate_cluster"):
        named.append(f"near-duplicate of {', '.join(sig['duplicate_cluster'])}")

    chain: list[str] = []
    cursor = row
    seen = set()
    while cursor is not None and cursor["superseded_by"] and cursor["doc_id"] not in seen:
        seen.add(cursor["doc_id"])
        cursor = conn.execute(
            "SELECT * FROM doc_registry WHERE doc_id = ?", (cursor["superseded_by"],)
        ).fetchone()
        if cursor is not None:
            chain.append(cursor["rel_path"])

    standing = (
        f"{row['rel_path']} is {row['tier']}/{row['lifecycle']} "
        f"({row['doc_type']}, via {row['type_source']})"
    )
    if named:
        standing += ": " + ", ".join(named)
    if chain:
        standing += f"; superseded by {chain[0]}"

    return {
        "found": True,
        "rel_path": row["rel_path"],
        "doc_id": row["doc_id"],
        "tier": row["tier"],
        "lifecycle": row["lifecycle"],
        "doc_type": row["doc_type"],
        "type_source": row["type_source"],
        "standing": standing,
        "signals": sig,
        "supersession_chain": chain,
        "last_verified": row["last_verified"],
    }


def build_report(
    conn: sqlite3.Connection, repo: str, policy: Policy
) -> dict[str, Any]:
    """Rot report over the ledger: stale, dead-ref, duplicate, archivable —
    proposals only, and never an archive proposal for T0/T1."""
    rows = conn.execute(
        "SELECT * FROM doc_registry WHERE repo_id = ? AND invalid_at IS NULL",
        (repo,),
    ).fetchall()

    stale: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    clusters_seen: set[frozenset[str]] = set()

    for row in rows:
        sig = _signals(row)
        age = sig.get("age_days")
        window = policy.stale_after_days.get(row["doc_type"])
        if (
            window is not None
            and age is not None
            and age > window
            and row["lifecycle"] in ("active", "draft")
        ):
            stale.append({
                "rel_path": row["rel_path"], "doc_type": row["doc_type"],
                "age_days": age, "window_days": window, "tier": row["tier"],
            })
        dead_sig = sig.get("dead_refs") or {}
        if dead_sig.get("count"):
            dead.append({
                "rel_path": row["rel_path"], "count": dead_sig["count"],
                "files": dead_sig.get("files", []),
                "symbols": dead_sig.get("symbols", []),
            })
        partners = sig.get("duplicate_cluster")
        if partners:
            clusters_seen.add(frozenset([row["rel_path"], *partners]))

        if row["tier"] in _PROTECTED_TIERS:
            continue  # T0/T1 are never archive candidates
        boxes = sig.get("checkbox_state") or {}
        reasons: list[str] = []
        if row["lifecycle"] in ("fulfilled", "superseded"):
            reasons.append(f"lifecycle {row['lifecycle']}")
        if (
            window is not None and age is not None and age > window
            and (row["tier"] == "T3" or boxes.get("ratio") == 1.0)
        ):
            reasons.append(f"stale {row['doc_type']} ({age}d > {window}d)")
        if reasons:
            archive.append({
                "rel_path": row["rel_path"], "tier": row["tier"],
                "reasons": reasons,
            })

    return {
        "total_docs": len(rows),
        "stale": sorted(stale, key=lambda item: -item["age_days"]),
        "dead_refs": sorted(dead, key=lambda item: -item["count"]),
        "duplicate_clusters": [sorted(group) for group in
                               sorted(clusters_seen, key=lambda g: sorted(g)[0])],
        "archive_candidates": sorted(archive, key=lambda item: item["rel_path"]),
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"documents tracked: {report['total_docs']}"]
    if report["stale"]:
        lines.append(f"\nstale ({len(report['stale'])}):")
        for item in report["stale"]:
            lines.append(
                f"  {item['rel_path']}  [{item['doc_type']}] "
                f"{item['age_days']}d old (window {item['window_days']}d)"
            )
    if report["dead_refs"]:
        lines.append(f"\ndead references ({len(report['dead_refs'])}):")
        for item in report["dead_refs"]:
            sample = ", ".join((item["files"] + item["symbols"])[:3])
            lines.append(f"  {item['rel_path']}  {item['count']} dead ({sample})")
    if report["duplicate_clusters"]:
        lines.append(f"\nduplicate clusters ({len(report['duplicate_clusters'])}):")
        for group in report["duplicate_clusters"]:
            lines.append("  " + "  ~  ".join(group))
    if report["archive_candidates"]:
        count = len(report["archive_candidates"])
        lines.append(f"\narchive candidates ({count}) — proposals only:")
        for item in report["archive_candidates"]:
            lines.append(f"  {item['rel_path']}  ({'; '.join(item['reasons'])})")
    if len(lines) == 1:
        lines.append("no rot detected")
    return "\n".join(lines)
