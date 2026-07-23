"""
CLARA docs — the ledger scan engine.

Idempotent by design: ``clara docs scan`` is the universal repair path. Scans
are hash-gated (unchanged documents skip signal collection), renames are
detected by content hash (the doc keeps its ``doc_id``), vanished documents
are invalidated (never deleted), and the quarantine manifest is regenerated
on every scan. Nothing here mutates repository files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clara.docs import refs as refs_mod
from clara.docs import signals as signals_mod
from clara.docs.simhash import cluster, simhash
from clara.policy import Policy, load_policy
from clara.repoid import repo_id as compute_repo_id

logger = logging.getLogger(__name__)

_LIFECYCLES = {"draft", "active", "fulfilled", "superseded", "archived"}
_TIERS = {"T0", "T1", "T2", "T3", "TX"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


def find_repo_root(cwd: str) -> str:
    """Git toplevel of *cwd*, else *cwd* itself."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return cwd


def clara_yml_ignored(root: str) -> bool:
    """True when clara.yml exists but is git-ignored (policy silently unused)."""
    import subprocess

    if not (Path(root) / "clara.yml").exists():
        return False
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "clara.yml"],
            cwd=root, capture_output=True, timeout=10,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def quarantine_dir(db_path: str) -> Path:
    """$CLARA_HOME/quarantine — resolved relative to the global store file so
    tests (which point CLARA_DB_PATH at a temp file) stay isolated."""
    return Path(db_path).resolve().parent / "quarantine"


def dirty_sidecar(db_path: str, repo: str) -> Path:
    return Path(db_path).resolve().parent / "dirty" / f"{repo}.list"


def _connect(db_path: str) -> sqlite3.Connection:
    from clara.db.migrations import ensure_schema

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _discover(root: Path, policy: Policy) -> list[str]:
    found: list[str] = []
    for path in root.rglob("*.md"):
        parts = path.relative_to(root).parts
        # Hidden/tool directories (.git, .pytest_cache, .venv, ...) are never
        # documentation; .github is the one dot-dir teams document in.
        if any(part.startswith(".") and part != ".github" for part in parts[:-1]):
            continue
        rel = path.relative_to(root).as_posix()
        if any(signals_mod._match(pattern, rel) for pattern in policy.exclude):
            continue
        found.append(rel)
    return sorted(found)


def _hash_file(path: Path) -> str:
    # Hash the DECODED text (newline-translated), matching the per-doc scan
    # hash — raw bytes would break rename detection on CRLF checkouts.
    text = path.read_text(encoding="utf-8", errors="replace")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _attest(
    conn: sqlite3.Connection, doc_id: str, verdict: str, rationale: str,
    evidence: dict[str, Any] | None = None, actor: str = "signal",
) -> None:
    conn.execute(
        "INSERT INTO doc_attestations (att_id, doc_id, actor, verdict, rationale, "
        "evidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, doc_id, actor, verdict, rationale,
         json.dumps(evidence or {}), _now_iso()),
    )


def _apply_frontmatter(
    conn: sqlite3.Connection, repo: str, doc_id: str, row_update: dict[str, Any],
    fm: dict[str, Any],
) -> None:
    """Author-stated frontmatter wins over path heuristics."""
    fm_type = str(fm.get("type", "")).strip().lower()
    if fm_type:
        row_update["doc_type"] = fm_type
        row_update["type_source"] = "frontmatter"
    fm_tier = str(fm.get("tier", "")).strip().upper()
    if fm_tier in _TIERS:
        row_update["tier"] = fm_tier
    fm_status = str(fm.get("status", "")).strip().lower()
    if fm_status in _LIFECYCLES:
        row_update["lifecycle"] = fm_status
    supersedes = str(fm.get("supersedes", "")).strip().replace("\\", "/")
    if supersedes:
        target = conn.execute(
            "SELECT doc_id, lifecycle FROM doc_registry WHERE repo_id = ? "
            "AND rel_path = ? AND invalid_at IS NULL",
            (repo, supersedes),
        ).fetchone()
        if target is not None and target["lifecycle"] != "superseded":
            conn.execute(
                "UPDATE doc_registry SET lifecycle = 'superseded', superseded_by = ?, "
                "updated_at = ? WHERE doc_id = ?",
                (doc_id, _now_iso(), target["doc_id"]),
            )
            _attest(conn, target["doc_id"], "superseded",
                    f"frontmatter of {row_update.get('rel_path', doc_id)} names it",
                    {"supersedes": supersedes})


def _write_quarantine_manifest(conn: sqlite3.Connection, db_path: str, repo: str) -> Path:
    rows = conn.execute(
        "SELECT rel_path, lifecycle, tier, superseded_by FROM doc_registry "
        "WHERE repo_id = ? AND invalid_at IS NULL "
        "AND (lifecycle IN ('superseded', 'archived') OR tier = 'TX') "
        "ORDER BY rel_path",
        (repo,),
    ).fetchall()
    lines = []
    for row in rows:
        status = "TX" if row["tier"] == "TX" else row["lifecycle"]
        note = ""
        if row["superseded_by"]:
            by = conn.execute(
                "SELECT rel_path FROM doc_registry WHERE doc_id = ?",
                (row["superseded_by"],),
            ).fetchone()
            note = f"superseded by {by['rel_path']}" if by else "superseded"
        lines.append(f"{row['rel_path']}\t{status}\t{note}")
    target = quarantine_dir(db_path) / f"{repo}.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
    )
    return target


def _project_doc_to_graph(
    conn: sqlite3.Connection, repo: str, doc_id: str, rel_path: str,
    file_refs: list[str],
) -> None:
    """Feature-detected: document nodes + describes→file edges when the graph
    layer's tables exist; silently skipped otherwise."""
    ready = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'graph_nodes'"
    ).fetchone()
    if ready is None:
        return
    now = _now_iso()

    def _node(canonical: str, entity_type: str) -> str:
        row = conn.execute(
            "SELECT node_id FROM graph_nodes WHERE status = 'active' "
            "AND canonical_name = ? AND entity_type = ? AND coalesce(user_id, '') = ''",
            (canonical, entity_type),
        ).fetchone()
        if row is not None:
            return str(row["node_id"])
        node_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO graph_nodes (node_id, canonical_name, display_name, "
            "entity_type, properties, mention_count, expandable, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, '{}', 1, 1, 'active', ?, ?)",
            (node_id, canonical, canonical, entity_type, now, now),
        )
        return node_id

    doc_node = _node(rel_path, "document")
    current_targets = set(file_refs)
    existing = conn.execute(
        "SELECT e.edge_id, n.canonical_name AS target FROM graph_edges e "
        "JOIN graph_nodes n ON n.node_id = e.dst_id "
        "WHERE e.src_id = ? AND e.relation = 'describes' AND e.invalid_at IS NULL",
        (doc_node,),
    ).fetchall()
    for row in existing:
        if row["target"] not in current_targets:
            conn.execute(
                "UPDATE graph_edges SET invalid_at = ?, metadata = json_set("
                "coalesce(metadata, '{}'), '$.invalidated_by', 'doc_ref_removed') "
                "WHERE edge_id = ?",
                (now, row["edge_id"]),
            )
    known = {row["target"] for row in existing}
    for target in sorted(current_targets - known)[:50]:
        target_node = _node(target, "file")
        conn.execute(
            "INSERT INTO graph_edges (edge_id, src_id, dst_id, relation, belief_id, "
            "confidence, weight, valid_from, metadata) "
            "VALUES (?, ?, ?, 'describes', NULL, 0.9, 1.0, ?, ?)",
            (uuid.uuid4().hex, doc_node, target_node, now,
             json.dumps({"source": "doc_ref", "doc_id": doc_id})),
        )


def scan_repo(
    db_path: str,
    repo_root: str,
    *,
    changed: list[str] | None = None,
    probe_symbols: bool = True,
) -> dict[str, Any]:
    """Scan repo documents into the ledger; returns summary counts.

    ``changed=None`` scans everything; a list limits signal collection to
    those paths (other rows keep their stored signals). Unchanged content
    hashes skip collection entirely.
    """
    root = Path(repo_root).resolve()
    repo = compute_repo_id(str(root))
    policy = load_policy(root)
    now_ts = time.time()

    conn = _connect(db_path)
    try:
        all_docs = _discover(root, policy)
        targets = set(all_docs) if changed is None else (
            {c.replace("\\", "/") for c in changed} & set(all_docs)
        )

        rows = {
            row["rel_path"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM doc_registry WHERE repo_id = ? AND invalid_at IS NULL",
                (repo,),
            ).fetchall()
        }

        history: signals_mod.GitHistory | None = None
        summary: dict[str, Any] = {
            "scanned": 0, "unchanged": 0, "new": 0, "moved": 0, "vanished": 0,
        }

        # --- rename / vanish pass (full scans only see the whole picture) ---
        if changed is None:
            present = set(all_docs)
            hash_by_path = {}
            for rel in present - set(rows):
                hash_by_path[rel] = _hash_file(root / rel)
            for rel, row in list(rows.items()):
                if rel in present:
                    continue
                new_home = next(
                    (p for p, h in hash_by_path.items()
                     if h == row["content_hash"] and p not in rows),
                    None,
                )
                if new_home is not None:
                    conn.execute(
                        "UPDATE doc_registry SET rel_path = ?, updated_at = ? "
                        "WHERE doc_id = ?",
                        (new_home, _now_iso(), row["doc_id"]),
                    )
                    _attest(conn, row["doc_id"], "moved", f"{rel} -> {new_home}")
                    rows[new_home] = {**row, "rel_path": new_home}
                    del rows[rel]
                    summary["moved"] += 1
                else:
                    conn.execute(
                        "UPDATE doc_registry SET invalid_at = ?, updated_at = ? "
                        "WHERE doc_id = ?",
                        (_now_iso(), _now_iso(), row["doc_id"]),
                    )
                    _attest(conn, row["doc_id"], "vanished", f"{rel} no longer exists")
                    del rows[rel]
                    summary["vanished"] += 1

        # --- per-document signal collection (hash-gated) ---
        for rel in sorted(targets):
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            existing = rows.get(rel)
            if existing is not None and existing["content_hash"] == content_hash:
                conn.execute(
                    "UPDATE doc_registry SET last_verified = ? WHERE doc_id = ?",
                    (_now_iso(), existing["doc_id"]),
                )
                summary["unchanged"] += 1
                continue

            if history is None:
                history = signals_mod.load_history(str(root))
            doc_refs = refs_mod.extract_refs(text)
            sig = signals_mod.collect_signals(
                root=str(root), rel_path=rel, text=text, refs=doc_refs,
                history=history, policy=policy, now_ts=now_ts,
                probe_symbols=probe_symbols,
            )
            sig["simhash"] = f"{simhash(text):016x}"
            convention = sig["path_convention"]

            row_update: dict[str, Any] = {
                "rel_path": rel,
                "content_hash": content_hash,
                "doc_type": convention["doc_type"],
                "tier": convention["tier"],
                "type_source": convention["source"] if convention["doc_type"] != "unknown"
                else "heuristic",
            }
            fm = sig.get("frontmatter")

            if existing is None:
                doc_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO doc_registry (doc_id, repo_id, rel_path, content_hash, "
                    "doc_type, tier, lifecycle, type_source, signals, valid_from, "
                    "last_verified, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
                    (doc_id, repo, rel, content_hash, row_update["doc_type"],
                     row_update["tier"], row_update["type_source"], json.dumps(sig),
                     _now_iso(), _now_iso(), _now_iso()),
                )
                summary["new"] += 1
            else:
                doc_id = existing["doc_id"]
                # Preserve judgment-set fields: only heuristics may be overwritten.
                if existing["type_source"] in ("claude", "user"):
                    row_update["doc_type"] = existing["doc_type"]
                    row_update["type_source"] = existing["type_source"]
                    row_update["tier"] = existing["tier"]
                conn.execute(
                    "UPDATE doc_registry SET content_hash = ?, doc_type = ?, tier = ?, "
                    "type_source = ?, signals = ?, last_verified = ?, updated_at = ? "
                    "WHERE doc_id = ?",
                    (content_hash, row_update["doc_type"], row_update["tier"],
                     row_update["type_source"], json.dumps(sig), _now_iso(),
                     _now_iso(), doc_id),
                )
            if fm:
                _apply_frontmatter(conn, repo, doc_id, row_update, fm)
                if "tier" in row_update or "doc_type" in row_update or "lifecycle" in row_update:
                    conn.execute(
                        "UPDATE doc_registry SET doc_type = coalesce(?, doc_type), "
                        "tier = coalesce(?, tier), lifecycle = coalesce(?, lifecycle), "
                        "type_source = coalesce(?, type_source) WHERE doc_id = ?",
                        (row_update.get("doc_type"), row_update.get("tier"),
                         row_update.get("lifecycle"),
                         "frontmatter" if fm.get("type") or fm.get("status") else None,
                         doc_id),
                    )

            conn.execute("DELETE FROM doc_refs WHERE doc_id = ?", (doc_id,))
            for kind, value in doc_refs:
                resolved = 1
                if kind in ("file", "doc"):
                    resolved = 1 if (root / value).exists() else 0
                conn.execute(
                    "INSERT OR IGNORE INTO doc_refs (doc_id, ref_kind, ref_value, "
                    "resolved) VALUES (?, ?, ?, ?)",
                    (doc_id, kind, value, resolved),
                )
            _project_doc_to_graph(
                conn, repo, doc_id, rel,
                [v for k, v in doc_refs if k in ("file", "doc")],
            )
            summary["scanned"] += 1

        _recompute_repo_wide(conn, repo)
        manifest = _write_quarantine_manifest(conn, db_path, repo)
        sidecar = dirty_sidecar(db_path, repo)
        if changed is not None and sidecar.exists():
            sidecar.unlink()
        conn.commit()
        summary["manifest"] = str(manifest)
        summary["total_active"] = int(conn.execute(
            "SELECT COUNT(*) FROM doc_registry WHERE repo_id = ? AND invalid_at IS NULL",
            (repo,),
        ).fetchone()[0])
        return summary
    finally:
        conn.close()


def _recompute_repo_wide(conn: sqlite3.Connection, repo: str) -> None:
    """Signals that need the whole repo view: link_indegree, duplicate_cluster."""
    rows = conn.execute(
        "SELECT doc_id, rel_path, signals FROM doc_registry "
        "WHERE repo_id = ? AND invalid_at IS NULL",
        (repo,),
    ).fetchall()
    by_path = {row["rel_path"]: row for row in rows}

    indegree: dict[str, int] = {}
    ref_rows = conn.execute(
        "SELECT r.ref_value, COUNT(*) AS n FROM doc_refs r "
        "JOIN doc_registry d ON d.doc_id = r.doc_id "
        "WHERE d.repo_id = ? AND d.invalid_at IS NULL AND r.ref_kind = 'doc' "
        "GROUP BY r.ref_value",
        (repo,),
    ).fetchall()
    for row in ref_rows:
        indegree[row["ref_value"]] = int(row["n"])

    fingerprints: dict[str, int] = {}
    for row in rows:
        try:
            sig = json.loads(row["signals"] or "{}")
        except ValueError:
            sig = {}
        fp = sig.get("simhash")
        if isinstance(fp, str):
            with contextlib.suppress(ValueError):
                fingerprints[row["rel_path"]] = int(fp, 16)
    clusters = cluster(fingerprints)
    cluster_of: dict[str, list[str]] = {}
    for group in clusters:
        for member in group:
            cluster_of[member] = sorted(group - {member})

    for rel_path, row in by_path.items():
        try:
            sig = json.loads(row["signals"] or "{}")
        except ValueError:
            sig = {}
        new_indegree = indegree.get(rel_path, 0)
        new_cluster = cluster_of.get(rel_path)
        if sig.get("link_indegree") == new_indegree and sig.get("duplicate_cluster") == new_cluster:
            continue
        sig["link_indegree"] = new_indegree
        if new_cluster:
            sig["duplicate_cluster"] = new_cluster
        else:
            sig.pop("duplicate_cluster", None)
        conn.execute(
            "UPDATE doc_registry SET signals = ? WHERE doc_id = ?",
            (json.dumps(sig), row["doc_id"]),
        )
