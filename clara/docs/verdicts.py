"""
CLARA docs — the judgment layer: attested verdicts about documents.

Every operation here writes a ``doc_attestations`` row (actor, verdict,
rationale, evidence) — judgment enters the ledger ONLY through these tools;
the scan engine never self-classifies beyond heuristic priors.

``fulfill`` is atomic: distilled memories, the lifecycle transition, the
attestation, and (when the graph layer is present) provenance edges all
commit in one transaction or not at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from clara.integrations.local_memory import LocalMemory

_TIERS = {"T0", "T1", "T2", "T3", "TX"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


async def _attest(
    session: AsyncSession,
    doc_id: str,
    *,
    actor: str,
    verdict: str,
    rationale: str,
    evidence: dict[str, Any] | None = None,
) -> str:
    att_id = uuid.uuid4().hex
    await session.execute(
        sa_text(
            "INSERT INTO doc_attestations (att_id, doc_id, actor, verdict, rationale, "
            "evidence, created_at) VALUES (:aid, :did, :actor, :verdict, :rationale, "
            ":evidence, :now)"
        ),
        {
            "aid": att_id, "did": doc_id, "actor": actor, "verdict": verdict,
            "rationale": rationale, "evidence": json.dumps(evidence or {}),
            "now": _now(),
        },
    )
    return att_id


async def _doc_by_path(
    session: AsyncSession, repo: str, rel_path: str
) -> dict[str, Any] | None:
    row = (
        await session.execute(
            sa_text(
                "SELECT * FROM doc_registry WHERE repo_id = :repo "
                "AND rel_path = :path AND invalid_at IS NULL"
            ),
            {"repo": repo, "path": rel_path.replace("\\", "/")},
        )
    ).mappings().first()
    return dict(row) if row else None


async def _graph_doc_node(session: AsyncSession, rel_path: str) -> str | None:
    """node_id of the document's graph node, or None (graph absent / no node)."""
    ready = (
        await session.execute(
            sa_text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_nodes'")
        )
    ).first()
    if ready is None:
        return None
    row = (
        await session.execute(
            sa_text(
                "SELECT node_id FROM graph_nodes WHERE status = 'active' "
                "AND entity_type = 'document' AND canonical_name = :name"
            ),
            {"name": rel_path},
        )
    ).first()
    if row is not None:
        return str(row[0])
    node_id = uuid.uuid4().hex
    now = _now()
    await session.execute(
        sa_text(
            "INSERT INTO graph_nodes (node_id, canonical_name, display_name, "
            "entity_type, properties, mention_count, expandable, status, "
            "created_at, updated_at) "
            "VALUES (:nid, :name, :name, 'document', '{}', 1, 1, 'active', :now, :now)"
        ),
        {"nid": node_id, "name": rel_path, "now": now},
    )
    return node_id


async def _graph_edge(
    session: AsyncSession,
    *,
    src_id: str,
    dst_id: str,
    relation: str,
    belief_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    edge_id = uuid.uuid4().hex
    await session.execute(
        sa_text(
            "INSERT INTO graph_edges (edge_id, src_id, dst_id, relation, belief_id, "
            "confidence, weight, valid_from, metadata) "
            "VALUES (:eid, :src, :dst, :rel, :bid, 0.9, 1.0, :now, :meta)"
        ),
        {
            "eid": edge_id, "src": src_id, "dst": dst_id, "rel": relation,
            "bid": belief_id, "now": _now(), "meta": json.dumps(metadata or {}),
        },
    )
    return edge_id


def _refresh_side_files(db_path: str, repo: str) -> None:
    """Regenerate the quarantine manifest (sync sqlite3; post-commit)."""
    import sqlite3

    from clara.docs.scan import _write_quarantine_manifest

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _write_quarantine_manifest(conn, db_path, repo)
    finally:
        conn.close()


def _drop_from_proposals(db_path: str, repo: str, rel_path: str) -> None:
    from pathlib import Path

    proposals = Path(db_path).resolve().parent / "proposals" / f"{repo}.txt"
    if not proposals.exists():
        return
    lines = [
        line for line in proposals.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split("\t")[0] != rel_path
    ]
    proposals.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
    )


# ---------------------------------------------------------------------------
# Verdict operations
# ---------------------------------------------------------------------------


async def classify(
    memory: LocalMemory,
    repo: str,
    *,
    path: str,
    doc_type: str,
    tier: str | None,
    rationale: str,
) -> dict[str, Any]:
    """Set a document's type (and optionally tier) with ``type_source='claude'``.

    Idempotent per (doc, verdict, content_hash): re-classifying unchanged
    content with the same verdict is a no-op.
    """
    if tier is not None and tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_TIERS)}")
    verdict = f"classify:{doc_type}" + (f":{tier}" if tier else "")
    async with memory._session_factory() as session, session.begin():
        doc = await _doc_by_path(session, repo, path)
        if doc is None:
            return {"found": False, "path": path,
                    "hint": "not in the ledger — run `clara docs scan`"}
        existing = (
            await session.execute(
                sa_text(
                    "SELECT att_id FROM doc_attestations WHERE doc_id = :did "
                    "AND verdict = :verdict AND evidence LIKE :hash_like"
                ),
                {"did": doc["doc_id"], "verdict": verdict,
                 "hash_like": f'%{doc["content_hash"]}%'},
            )
        ).first()
        if existing is not None:
            return {"found": True, "doc_id": doc["doc_id"], "action": "unchanged",
                    "doc_type": doc_type, "tier": tier or doc["tier"]}
        await session.execute(
            sa_text(
                "UPDATE doc_registry SET doc_type = :dtype, tier = :tier, "
                "type_source = 'claude', updated_at = :now WHERE doc_id = :did"
            ),
            {"dtype": doc_type, "tier": tier or doc["tier"], "now": _now(),
             "did": doc["doc_id"]},
        )
        await _attest(
            session, doc["doc_id"], actor="claude", verdict=verdict,
            rationale=rationale,
            evidence={"content_hash": doc["content_hash"], "doc_type": doc_type,
                      "tier": tier},
        )
    await asyncio.to_thread(_refresh_side_files, memory._db_path, repo)
    return {"found": True, "doc_id": doc["doc_id"], "action": "classified",
            "doc_type": doc_type, "tier": tier or doc["tier"]}


async def supersede(
    memory: LocalMemory,
    repo: str,
    *,
    old_path: str,
    new_path: str,
    rationale: str,
) -> dict[str, Any]:
    """Mark ``old_path`` superseded by ``new_path`` (+ graph ``supersedes`` edge)."""
    async with memory._session_factory() as session, session.begin():
        old_doc = await _doc_by_path(session, repo, old_path)
        new_doc = await _doc_by_path(session, repo, new_path)
        if old_doc is None or new_doc is None:
            missing = [p for p, d in ((old_path, old_doc), (new_path, new_doc))
                       if d is None]
            return {"found": False, "missing": missing,
                    "hint": "not in the ledger — run `clara docs scan`"}
        await session.execute(
            sa_text(
                "UPDATE doc_registry SET lifecycle = 'superseded', "
                "superseded_by = :new_id, updated_at = :now WHERE doc_id = :old_id"
            ),
            {"new_id": new_doc["doc_id"], "now": _now(), "old_id": old_doc["doc_id"]},
        )
        att_id = await _attest(
            session, old_doc["doc_id"], actor="claude", verdict="superseded",
            rationale=rationale, evidence={"superseded_by": new_doc["rel_path"]},
        )
        edge_id: str | None = None
        new_node = await _graph_doc_node(session, new_doc["rel_path"])
        if new_node is not None:
            old_node = await _graph_doc_node(session, old_doc["rel_path"])
            if old_node is not None:
                edge_id = await _graph_edge(
                    session, src_id=new_node, dst_id=old_node, relation="supersedes",
                    metadata={"attestation": att_id},
                )
        old_doc_id = old_doc["doc_id"]
        new_doc_id = new_doc["doc_id"]
    await asyncio.to_thread(_refresh_side_files, memory._db_path, repo)
    return {"found": True, "old_doc_id": old_doc_id,
            "new_doc_id": new_doc_id, "edge_id": edge_id,
            "action": "superseded"}


async def fulfill(
    memory: LocalMemory,
    repo: str,
    *,
    path: str,
    distilled: list[dict[str, Any]],
    evidence: str | None,
    rationale: str = "plan completed",
) -> dict[str, Any]:
    """Fulfill a document: distill facts into memory, one transaction.

    Each entry in *distilled* is memory_save-shaped (mem_type + fields).
    Saved memories carry provenance ``{source: 'docs_fulfill', doc_id, tier}``
    plus ``doc_tier`` (feeds retrieval weighting). With the graph layer
    present, each belief gets a ``derived_from`` edge to the document node,
    and the document a ``fulfilled_by`` edge to the evidence ref.
    """
    from clara.integrations.local_memory import VALID_TYPES
    from clara.integrations.local_memory import LocalMemory as _LM

    if not distilled:
        raise ValueError("distilled must contain at least one fact")
    for fact in distilled:
        mem_type = fact.get("mem_type", "belief")
        if mem_type not in VALID_TYPES:
            raise ValueError(f"unknown mem_type {mem_type!r} in distilled fact")

    memory_ids: list[str] = []
    edge_ids: list[str] = []
    async with memory._session_factory() as session, session.begin():
        doc = await _doc_by_path(session, repo, path)
        if doc is None:
            return {"found": False, "path": path,
                    "hint": "not in the ledger — run `clara docs scan`"}
        doc_node = await _graph_doc_node(session, doc["rel_path"])

        for fact in distilled:
            fact = dict(fact)
            mem_type = fact.pop("mem_type", "belief")
            confidence = fact.pop("confidence", None)
            tags = fact.pop("tags", None)
            record = await _LM._route_save(
                session,
                mem_type=mem_type,
                subject=fact.get("subject"),
                relation=fact.get("relation"),
                object=fact.get("object"),
                is_negation=bool(fact.get("is_negation", False)),
                event_type=fact.get("event_type"),
                name=fact.get("name"),
                trigger_conditions=fact.get("trigger_conditions"),
                steps=fact.get("steps"),
                entity_type=fact.get("entity_type"),
                properties=fact.get("properties"),
                description=fact.get("description"),
                domain=fact.get("domain"),
                confidence=confidence,
                source="agent_reflection",
                user_id=None,
            )
            if confidence is not None:
                record.confidence = max(0.0, min(1.0, float(confidence)))
            meta = dict(record.metadata_) if record.metadata_ else {}
            if tags:
                meta["tags"] = list(tags)
            meta["provenance"] = {
                "source": "docs_fulfill", "doc_id": doc["doc_id"], "tier": doc["tier"],
            }
            meta["doc_tier"] = doc["tier"]
            record.metadata_ = meta
            await session.flush()
            memory_id = str(record.memory_id)
            memory_ids.append(memory_id)
            if doc_node is not None and mem_type == "belief":
                from clara.graph.resolve import resolve_node

                subject_node = await resolve_node(
                    session, fact.get("subject", ""), create=False, bump_mention=False
                )
                if subject_node is not None:
                    edge_ids.append(await _graph_edge(
                        session, src_id=subject_node["node_id"], dst_id=doc_node,
                        relation="derived_from", belief_id=memory_id,
                        metadata={"doc_id": doc["doc_id"]},
                    ))

        await session.execute(
            sa_text(
                "UPDATE doc_registry SET lifecycle = 'fulfilled', fulfilled_by = :fb, "
                "updated_at = :now WHERE doc_id = :did"
            ),
            {"fb": json.dumps({"evidence": evidence, "memory_ids": memory_ids}),
             "now": _now(), "did": doc["doc_id"]},
        )
        await _attest(
            session, doc["doc_id"], actor="claude", verdict="fulfilled",
            rationale=rationale,
            evidence={"evidence": evidence, "memory_ids": memory_ids},
        )
        if doc_node is not None and evidence:
            from clara.graph.resolve import resolve_node

            ref_node = await resolve_node(
                session, evidence, entity_type="ref", create=True, bump_mention=False
            )
            if ref_node is not None:
                edge_ids.append(await _graph_edge(
                    session, src_id=doc_node, dst_id=ref_node["node_id"],
                    relation="fulfilled_by", metadata={"doc_id": doc["doc_id"]},
                ))

    await asyncio.to_thread(_refresh_side_files, memory._db_path, repo)
    await asyncio.to_thread(
        _drop_from_proposals, memory._db_path, repo, doc["rel_path"]
    )
    return {"found": True, "doc_id": doc["doc_id"], "memory_ids": memory_ids,
            "edge_ids": edge_ids, "action": "fulfilled"}


# ---------------------------------------------------------------------------
# Archive plumbing (user-invoked only; Prompt 06 wires governance around it)
# ---------------------------------------------------------------------------


def _git_mv(root: str, src: str, dst: str) -> bool:
    import subprocess
    from pathlib import Path

    (Path(root) / dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "mv", src, dst], cwd=root, capture_output=True, text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


async def archive(
    memory: LocalMemory,
    repo: str,
    *,
    repo_root: str,
    path: str,
    rationale: str = "archived by user",
) -> dict[str, Any]:
    """Archive a document: ledger transition + git mv per policy. Never automatic."""
    from clara.policy import load_policy

    policy = load_policy(repo_root)
    async with memory._session_factory() as session, session.begin():
        doc = await _doc_by_path(session, repo, path)
        if doc is None:
            return {"found": False, "path": path}
        if doc["tier"] in ("T0", "T1"):
            return {"found": True, "action": "refused",
                    "reason": f"{doc['tier']} documents are never archived"}
        new_rel = doc["rel_path"]
        moved = False
        if policy.archive_mode == "git-mv":
            target = f"{policy.archive_dir}/{doc['rel_path']}"
            moved = await asyncio.to_thread(
                _git_mv, repo_root, doc["rel_path"], target
            )
            if moved:
                new_rel = target
        await session.execute(
            sa_text(
                "UPDATE doc_registry SET lifecycle = 'archived', rel_path = :rel, "
                "updated_at = :now WHERE doc_id = :did"
            ),
            {"rel": new_rel, "now": _now(), "did": doc["doc_id"]},
        )
        await _attest(
            session, doc["doc_id"], actor="user", verdict="archived",
            rationale=rationale,
            evidence={"from": doc["rel_path"], "to": new_rel, "moved": moved},
        )
    await asyncio.to_thread(_refresh_side_files, memory._db_path, repo)
    return {"found": True, "action": "archived", "from": doc["rel_path"],
            "to": new_rel, "moved": moved}


async def restore(
    memory: LocalMemory,
    repo: str,
    *,
    repo_root: str,
    path: str,
    rationale: str = "restored by user",
) -> dict[str, Any]:
    """Reverse an archive: move the file back and reactivate the ledger row."""
    async with memory._session_factory() as session, session.begin():
        doc = await _doc_by_path(session, repo, path)
        if doc is None or doc["lifecycle"] != "archived":
            return {"found": False, "path": path,
                    "hint": "no archived ledger entry at this path"}
        original = (
            await session.execute(
                sa_text(
                    "SELECT evidence FROM doc_attestations WHERE doc_id = :did "
                    "AND verdict = 'archived' ORDER BY created_at DESC LIMIT 1"
                ),
                {"did": doc["doc_id"]},
            )
        ).first()
        evidence = json.loads(original[0]) if original else {}
        back_rel = evidence.get("from") or doc["rel_path"]
        moved = False
        if evidence.get("moved"):
            moved = await asyncio.to_thread(
                _git_mv, repo_root, doc["rel_path"], back_rel
            )
        new_rel = back_rel if (moved or not evidence.get("moved")) else doc["rel_path"]
        await session.execute(
            sa_text(
                "UPDATE doc_registry SET lifecycle = 'active', rel_path = :rel, "
                "updated_at = :now WHERE doc_id = :did"
            ),
            {"rel": new_rel, "now": _now(), "did": doc["doc_id"]},
        )
        await _attest(
            session, doc["doc_id"], actor="user", verdict="restored",
            rationale=rationale,
            evidence={"from": doc["rel_path"], "to": new_rel, "moved": moved},
        )
    await asyncio.to_thread(_refresh_side_files, memory._db_path, repo)
    return {"found": True, "action": "restored", "to": new_rel, "moved": moved}
