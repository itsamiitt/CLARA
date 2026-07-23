"""Tests for the judgment layer: verdicts, fulfillment, hooks (Prompt 05)."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from clara.docs.scan import scan_repo
from clara.fastpath import docs_map
from clara.integrations.local_memory import LocalMemory
from clara.repoid import repo_id
from tests.test_docs import _commit_all, _make_repo, requires_git
from tests.test_plugin_layout import _working_bash

_ROOT = Path(__file__).parents[1]
_PLAN = (
    "# Plan: switch primary store\n\n"
    "- [x] evaluate options\n- [x] migrate data\n- [x] cut over\n\n"
    "Decision: use postgresql over mysql for the primary store. Refs #42.\n"
)


async def _fixture(tmp_path, monkeypatch):
    """Git repo with a completed plan + a v1/v2 doc pair, scanned into a
    global store that LocalMemory shares."""
    root = tmp_path / "repo"
    _make_repo(root)
    (root / "docs" / "plans").mkdir(parents=True)
    (root / "docs" / "plans" / "store-plan.md").write_text(_PLAN, encoding="utf-8")
    (root / "docs" / "plans" / "old-design.md").write_text(
        "# Old design\n\nfirst attempt\n", encoding="utf-8")
    (root / "docs" / "plans" / "new-design.md").write_text(
        "# New design\n\nsecond attempt\n", encoding="utf-8")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    _commit_all(root, "docs")

    db = tmp_path / "global" / "clara.db"
    db.parent.mkdir()
    monkeypatch.setenv("CLARA_DB_PATH", str(db))
    monkeypatch.setenv("CLARA_HOME", str(db.parent))
    scan_repo(str(db), str(root), probe_symbols=False)
    memory = await LocalMemory.create(str(db))
    return root, db, memory


_DISTILLED = [
    {"mem_type": "belief", "subject": "project", "relation": "uses",
     "object": "postgresql over mysql", "confidence": 0.95,
     "description": "primary store decision"},
    {"mem_type": "belief", "subject": "project", "relation": "requires",
     "object": "zero-downtime migrations", "confidence": 0.9},
    {"mem_type": "world_model", "entity_type": "service", "name": "primary-store",
     "properties": {"engine": "postgresql"}},
]


@requires_git
class TestFulfill:
    async def test_end_to_end(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        result = await memory.docs_fulfill(
            str(root), path="docs/plans/store-plan.md",
            distilled=_DISTILLED, evidence="PR #42",
        )
        assert result["found"] and result["action"] == "fulfilled"
        assert len(result["memory_ids"]) == 3
        assert result["edge_ids"], "expected derived_from/fulfilled_by edges"

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            doc = conn.execute(
                "SELECT lifecycle, fulfilled_by FROM doc_registry "
                "WHERE rel_path = 'docs/plans/store-plan.md'"
            ).fetchone()
            assert doc["lifecycle"] == "fulfilled"
            assert "PR #42" in doc["fulfilled_by"]
            atts = conn.execute(
                "SELECT verdict FROM doc_attestations WHERE verdict = 'fulfilled'"
            ).fetchall()
            assert len(atts) == 1
            meta_rows = conn.execute("SELECT metadata FROM memories").fetchall()
            provenanced = [
                json.loads(row["metadata"]) for row in meta_rows
                if "docs_fulfill" in (row["metadata"] or "")
            ]
            assert len(provenanced) == 3
            assert all(m["provenance"]["source"] == "docs_fulfill" for m in provenanced)
            assert all(m["doc_tier"] == "T2" for m in provenanced)
            derived = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE relation = 'derived_from'"
            ).fetchone()[0]
            fulfilled_by = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE relation = 'fulfilled_by'"
            ).fetchone()[0]
            assert derived >= 2  # the two beliefs
            assert fulfilled_by == 1
        finally:
            conn.close()
        await memory.close()

        # A NEW session answers the plan's key decision from [MEMORY CONTEXT]
        # without the document in context.
        env = os.environ.copy()
        env["CLARA_DB_PATH"] = str(db)
        env["CLARA_HOME"] = str(db.parent)
        proc = subprocess.run(
            [sys.executable, "-m", "clara.fastpath.context", "--cwd", str(root)],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert "postgresql over mysql" in proc.stdout
        assert "cut over" not in proc.stdout  # no doc content leaked

    async def test_fulfill_is_atomic(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        bad = [dict(_DISTILLED[0]), {"mem_type": "belief", "subject": "x"}]  # missing fields
        with pytest.raises(ValueError):
            await memory.docs_fulfill(
                str(root), path="docs/plans/store-plan.md", distilled=bad,
            )
        conn = sqlite3.connect(db)
        try:
            (lifecycle,) = conn.execute(
                "SELECT lifecycle FROM doc_registry "
                "WHERE rel_path = 'docs/plans/store-plan.md'"
            ).fetchone()
            (mem_count,) = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        finally:
            conn.close()
        await memory.close()
        assert lifecycle == "active"
        assert mem_count == 0  # nothing committed from the failed transaction

    async def test_empty_distilled_rejected(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="at least one"):
            await memory.docs_fulfill(
                str(root), path="docs/plans/store-plan.md", distilled=[],
            )
        await memory.close()


@requires_git
class TestClassifyAndSupersede:
    async def test_classify_idempotent(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        first = await memory.docs_classify(
            str(root), path="README.md", doc_type="guide", tier="T1",
            rationale="canonical project entry point",
        )
        second = await memory.docs_classify(
            str(root), path="README.md", doc_type="guide", tier="T1",
            rationale="same again",
        )
        await memory.close()
        assert first["action"] == "classified"
        assert second["action"] == "unchanged"
        conn = sqlite3.connect(db)
        try:
            (atts,) = conn.execute(
                "SELECT COUNT(*) FROM doc_attestations WHERE verdict LIKE 'classify%'"
            ).fetchone()
            row = conn.execute(
                "SELECT doc_type, tier, type_source FROM doc_registry "
                "WHERE rel_path = 'README.md'"
            ).fetchone()
        finally:
            conn.close()
        assert atts == 1
        assert row == ("guide", "T1", "claude")

    async def test_supersede_quarantines_and_links(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        result = await memory.docs_supersede(
            str(root), old_path="docs/plans/old-design.md",
            new_path="docs/plans/new-design.md", rationale="v2 replaces v1",
        )
        await memory.close()
        assert result["found"] and result["edge_id"]
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT lifecycle, superseded_by FROM doc_registry "
                "WHERE rel_path = 'docs/plans/old-design.md'"
            ).fetchone()
            (edge_rel,) = conn.execute(
                "SELECT relation FROM graph_edges WHERE edge_id = ?",
                (result["edge_id"],),
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "superseded"
        assert row[1] == result["new_doc_id"]
        assert edge_rel == "supersedes"
        manifest = db.parent / "quarantine" / f"{repo_id(str(root))}.tsv"
        assert "docs/plans/old-design.md\tsuperseded" in manifest.read_text("utf-8")


@requires_git
class TestArchiveRestore:
    async def test_round_trip_with_git_mv(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        archived = await memory.docs_archive(str(root), path="docs/plans/old-design.md")
        assert archived["action"] == "archived" and archived["moved"]
        assert archived["to"] == "docs/archive/docs/plans/old-design.md"
        assert not (root / "docs" / "plans" / "old-design.md").exists()
        assert (root / archived["to"]).exists()

        restored = await memory.docs_restore(str(root), path=archived["to"])
        await memory.close()
        assert restored["action"] == "restored" and restored["moved"]
        assert (root / "docs" / "plans" / "old-design.md").exists()
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT lifecycle FROM doc_registry "
                "WHERE rel_path = 'docs/plans/old-design.md' AND invalid_at IS NULL"
            ).fetchone()
            verdicts = [v for (v,) in conn.execute(
                "SELECT verdict FROM doc_attestations ORDER BY created_at"
            ).fetchall()]
        finally:
            conn.close()
        assert row[0] == "active"
        assert "archived" in verdicts and "restored" in verdicts

    async def test_t1_archive_refused(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        await memory.docs_classify(
            str(root), path="README.md", doc_type="guide", tier="T1",
            rationale="entry point",
        )
        refused = await memory.docs_archive(str(root), path="README.md")
        await memory.close()
        assert refused["action"] == "refused"
        assert (root / "README.md").exists()


class TestGraphMergeExport:
    async def test_merge_reversible_auditable(self, tmp_path):
        memory = await LocalMemory.create(str(tmp_path / "g.db"))
        await memory.save(mem_type="belief", subject="frontend", relation="uses",
                          object="reactquery")
        await memory.save(mem_type="belief", subject="dashboard", relation="uses",
                          object="tanstack query")
        result = await memory.graph_merge("reactquery", "tanstack query")
        assert result["merged"], result

        async with memory._session_factory() as session:
            loser = (await session.execute(
                sa_text("SELECT * FROM graph_nodes WHERE node_id = :nid"),
                {"nid": result["loser"]},
            )).mappings().first()
            winner_edges = (await session.execute(
                sa_text("SELECT COUNT(*) FROM graph_edges WHERE dst_id = :nid "
                        "AND invalid_at IS NULL"),
                {"nid": result["winner"]},
            )).scalar_one()
            alias = (await session.execute(
                sa_text("SELECT node_id FROM graph_aliases WHERE alias_norm = 'reactquery'"),
            )).first()
        assert loser["status"] == "merged"
        assert loser["merged_into"] == result["winner"]
        audit = json.loads(loser["properties"])["merge_audit"]
        # Audit + merged_into reconstruct the pre-merge state: every re-pointed
        # edge is listed, and the loser's identity is intact.
        assert set(audit["repointed_dst_edges"]) | set(audit["repointed_src_edges"])
        assert audit["loser_canonical"] == "reactquery"
        assert winner_edges == 2
        assert alias is not None and alias[0] == result["winner"]

        card = await memory.graph_entity("reactquery")
        await memory.close()
        assert card["found"]
        assert card["canonical_name"] == "tanstack query"  # alias resolves to winner

    async def test_export_sanitizes_labels(self, tmp_path):
        memory = await LocalMemory.create(str(tmp_path / "g.db"))
        await memory.save(mem_type="belief", subject='evil "quoted [name]',
                          relation="uses", object="clean target")
        mermaid = await memory.graph_export(format="mermaid")
        dot = await memory.graph_export(format="dot")
        exported = await memory.graph_export(format="json")
        await memory.close()
        assert mermaid.startswith("graph LR")
        assert '"quoted' not in mermaid.split("graph LR", 1)[1].replace(
            '["', "").replace('"]', "")
        assert "evil 'quoted (name)" in mermaid
        assert "digraph clara" in dot
        assert '\\"' not in dot
        assert json.loads(exported)[0]["relation"] == "uses"


@requires_git
class TestProposalsAndStopHook:
    async def _prepared(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        await memory.close()
        block = docs_map.build_map(str(root))
        assert block is not None
        return root, db

    async def test_proposals_file_lists_complete_plan(self, tmp_path, monkeypatch):
        root, db = await self._prepared(tmp_path, monkeypatch)
        proposals = db.parent / "proposals" / f"{repo_id(str(root))}.txt"
        assert proposals.exists()
        assert "docs/plans/store-plan.md" in proposals.read_text("utf-8")
        index = db.parent / "proposals" / "index.tsv"
        assert f"\t{repo_id(str(root))}" in index.read_text("utf-8")

    async def test_stop_hook_nudges_exactly_once(self, tmp_path, monkeypatch):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._prepared(tmp_path, monkeypatch)
        env = {**os.environ, "CLARA_HOME": str(db.parent),
               "CLAUDE_SESSION_ID": "sess-once"}
        script = str(_ROOT / "scripts" / "session-stop.sh")

        outputs, timings = [], []
        for _ in range(3):
            start = time.perf_counter()
            proc = subprocess.run(
                [bash, script], cwd=str(root), env=env,
                capture_output=True, text=True, timeout=30,
            )
            timings.append(time.perf_counter() - start)
            assert proc.returncode == 0
            outputs.append(proc.stdout)
        nudges = [out for out in outputs if "looks complete" in out]
        assert len(nudges) == 1, outputs
        assert "docs/plans/store-plan.md" in nudges[0]
        assert "/clara:done" in nudges[0]
        print(f"stop-hook timings: {[f'{t * 1000:.0f}ms' for t in timings]}")

    async def test_stop_hook_silent_without_proposals(self, tmp_path, monkeypatch):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._prepared(tmp_path, monkeypatch)
        proposals = db.parent / "proposals" / f"{repo_id(str(root))}.txt"
        proposals.write_text("", encoding="utf-8")
        env = {**os.environ, "CLARA_HOME": str(db.parent),
               "CLAUDE_SESSION_ID": "sess-none"}
        proc = subprocess.run(
            [_working_bash(), str(_ROOT / "scripts" / "session-stop.sh")],
            cwd=str(root), env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0
        assert proc.stdout == ""


@requires_git
class TestReadAnnotateHook:
    async def _quarantined(self, tmp_path, monkeypatch):
        root, db, memory = await _fixture(tmp_path, monkeypatch)
        await memory.docs_supersede(
            str(root), old_path="docs/plans/old-design.md",
            new_path="docs/plans/new-design.md", rationale="v2",
        )
        await memory.close()
        docs_map.build_map(str(root))  # writes the repo index for the hook
        return root, db

    def _run_hook(self, bash, root, db, file_path, session="sess-read"):
        # realpath expands Windows 8.3 short names — the long-form native
        # path is what Claude's Read tool passes in real sessions.
        payload = json.dumps({
            "tool_name": "Read",
            "cwd": str(root),
            "tool_input": {"file_path": os.path.realpath(file_path)},
        })
        env = {**os.environ, "CLARA_HOME": str(db.parent),
               "CLAUDE_SESSION_ID": session}
        start = time.perf_counter()
        proc = subprocess.run(
            [bash, str(_ROOT / "scripts" / "read-annotate.sh")],
            input=payload, cwd=str(root), env=env,
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.perf_counter() - start
        assert proc.returncode == 0
        return proc.stdout, elapsed

    async def test_quarantined_read_annotated_once(self, tmp_path, monkeypatch):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._quarantined(tmp_path, monkeypatch)
        target = root / "docs" / "plans" / "old-design.md"

        out1, t1 = self._run_hook(bash, root, db, target)
        parsed = json.loads(out1)
        context = parsed["hookSpecificOutput"]["additionalContext"]
        assert parsed["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        assert context.startswith("CLARA: docs/plans/old-design.md is superseded")
        assert "historical record" in context

        out2, t2 = self._run_hook(bash, root, db, target)
        assert out2 == ""  # per-file once per session

        out3, _ = self._run_hook(bash, root, db, target, session="sess-other")
        assert "additionalContext" in out3  # fresh session annotates again
        print(f"read-annotate timings: {t1 * 1000:.0f}ms / {t2 * 1000:.0f}ms")

    async def test_normal_read_silent(self, tmp_path, monkeypatch):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._quarantined(tmp_path, monkeypatch)
        out, _ = self._run_hook(bash, root, db, root / "README.md")
        assert out == ""

    async def test_garbage_stdin_fails_open(self, tmp_path, monkeypatch):
        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._quarantined(tmp_path, monkeypatch)
        env = {**os.environ, "CLARA_HOME": str(db.parent)}
        proc = subprocess.run(
            [bash, str(_ROOT / "scripts" / "read-annotate.sh")],
            input="not json at all {{{", cwd=str(root), env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0
        assert proc.stdout == ""
