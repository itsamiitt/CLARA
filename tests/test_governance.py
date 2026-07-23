"""Governance milestone tests: scripted docs-review reversibility + kill switches."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess

import pytest

from clara import cli
from clara.docs.scan import scan_repo
from clara.fastpath import docs_map
from clara.fastpath.context import rank
from clara.integrations.local_memory import LocalMemory
from clara.repoid import repo_id
from tests.test_docs import _build_rotting_repo, _commit_all, _git, requires_git


def _run_cli(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code or 0)


def _ledger_rows(db) -> dict[str, dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return {
            row["doc_id"]: {
                "rel_path": row["rel_path"],
                "lifecycle": row["lifecycle"],
                "tier": row["tier"],
                "doc_type": row["doc_type"],
            }
            for row in conn.execute(
                "SELECT * FROM doc_registry WHERE invalid_at IS NULL"
            ).fetchall()
        }
    finally:
        conn.close()


def _valid_edge_triples(db) -> set[tuple[str, str, str]]:
    conn = sqlite3.connect(db)
    try:
        return {
            (src, rel, dst)
            for src, rel, dst in conn.execute(
                "SELECT s.canonical_name, e.relation, d.canonical_name "
                "FROM graph_edges e "
                "JOIN graph_nodes s ON s.node_id = e.src_id "
                "JOIN graph_nodes d ON d.node_id = e.dst_id "
                "WHERE e.invalid_at IS NULL"
            ).fetchall()
        }
    finally:
        conn.close()


@requires_git
class TestReversibility:
    """The milestone centerpiece.

    Intended semantics, asserted below: a docs-review commit is fully
    reversible for REPO-DERIVED state — `git revert` restores the files and
    `clara docs scan` restores every location-coupled ledger field (paths,
    archived lifecycles via the recorded prior_lifecycle). KNOWLEDGE is
    intentionally NOT reverted: promoted memories, `fulfilled` and
    `superseded` verdicts (and their graph edges / manifest lines) persist,
    because they are attested observations about document content that were
    true when made — reverting a file move does not make a distilled
    decision un-decided. The test asserts both halves exactly.
    """

    async def test_review_apply_revert_rescan(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        _build_rotting_repo(root)
        # One genuinely complete plan for the promotion pass.
        done_plan = root / "docs" / "plans" / "done-plan.md"
        done_plan.write_text(
            "# Done plan\n\n- [x] ship it\n- [x] verify\n\n"
            "Decision: reviews land as single commits.\n",
            encoding="utf-8",
        )
        _commit_all(root, "add completed plan")

        db = tmp_path / "global" / "clara.db"
        db.parent.mkdir()
        monkeypatch.setenv("CLARA_DB_PATH", str(db))
        monkeypatch.setenv("CLARA_HOME", str(db.parent))
        scan_repo(str(db), str(root), probe_symbols=False)

        memory = await LocalMemory.create(str(db))
        await memory.graph_rebuild()

        pre_rows = _ledger_rows(db)
        pre_edges = _valid_edge_triples(db)
        pre_files = {p.relative_to(root).as_posix() for p in root.rglob("*.md")}

        # --- scripted review: approve everything the report proposes ---
        report = await memory.docs_report(str(root))

        promoted = await memory.docs_fulfill(
            str(root), path="docs/plans/done-plan.md",
            distilled=[
                {"mem_type": "belief", "subject": "team", "relation": "requires",
                 "object": "single-commit doc reviews", "confidence": 0.95},
                {"mem_type": "belief", "subject": "docs-review", "relation": "uses",
                 "object": "git revert for rollback", "confidence": 0.9},
            ],
            evidence="commit review-1",
        )
        assert promoted["found"]
        fulfilled_ids = {promoted["doc_id"]}

        superseded_ids = set()
        for cluster in report["duplicate_clusters"]:
            keeper, *losers = sorted(cluster)
            for loser in losers:
                result = await memory.docs_supersede(
                    str(root), old_path=loser, new_path=keeper,
                    rationale="duplicate cluster resolution",
                )
                assert result["found"]
                superseded_ids.add(result["old_doc_id"])

        archived_paths: list[str] = []
        refreshed_report = await memory.docs_report(str(root))
        for item in refreshed_report["archive_candidates"]:
            result = await memory.docs_archive(str(root), path=item["rel_path"])
            assert result["action"] == "archived", result
            archived_paths.append(item["rel_path"])
        assert archived_paths, "fixture must yield at least one archive candidate"
        await memory.close()

        # ONE reviewable commit for the whole session.
        _git(root, "add", "-A")
        _git(root, "commit", "-m", "docs-review: archives + promotions")

        post_rows = _ledger_rows(db)
        for path in archived_paths:
            assert not (root / path).exists()
            row = next(r for r in post_rows.values()
                       if r["rel_path"].endswith(path.rsplit("/", 1)[-1])
                       and r["lifecycle"] == "archived")
            assert row["rel_path"].startswith("docs/archive/")

        # --- revert + repair ---
        _git(root, "revert", "--no-edit", "HEAD")
        scan_repo(str(db), str(root), probe_symbols=False)
        memory = await LocalMemory.create(str(db))
        await memory.graph_rebuild()
        await memory.close()

        # Repo files exactly restored.
        assert {p.relative_to(root).as_posix() for p in root.rglob("*.md")} == pre_files

        # Ledger: identical doc set; every field equals the pre-review
        # snapshot EXCEPT the intended judgment residue.
        final_rows = _ledger_rows(db)
        assert set(final_rows) == set(pre_rows)
        for doc_id, pre in pre_rows.items():
            final = final_rows[doc_id]
            assert final["rel_path"] == pre["rel_path"], doc_id
            assert final["tier"] == pre["tier"], doc_id
            assert final["doc_type"] == pre["doc_type"], doc_id
            if doc_id in fulfilled_ids:
                assert final["lifecycle"] == "fulfilled"
            elif doc_id in superseded_ids:
                assert final["lifecycle"] == "superseded"
            else:
                assert final["lifecycle"] == pre["lifecycle"], (doc_id, pre, final)
        assert not any(r["lifecycle"] == "archived" for r in final_rows.values())

        # Quarantine manifest: exactly the review's superseded lines — no
        # archived residue.
        manifest = (db.parent / "quarantine" / f"{repo_id(str(root))}.tsv").read_text(
            "utf-8"
        )
        manifest_paths = {line.split("\t")[0] for line in manifest.splitlines() if line}
        expected_superseded = {
            pre_rows[doc_id]["rel_path"] for doc_id in superseded_ids
        }
        assert manifest_paths == expected_superseded
        assert "archived" not in manifest

        # Graph: every pre-review edge still valid, and the promoted
        # memories are re-linked (derived_from edges to the plan document).
        final_edges = _valid_edge_triples(db)
        assert pre_edges <= final_edges
        assert ("team", "requires", "single commit doc reviews") in {
            (s, r, d) for s, r, d in final_edges
        }
        derived = {t for t in final_edges if t[1] == "derived_from"}
        assert any(d == "docs/plans/done-plan.md" for _, _, d in derived)

        # Promoted memories persist with provenance.
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT metadata FROM memories WHERE status = 'active'"
            ).fetchall()
        finally:
            conn.close()
        provenanced = [r for (r,) in rows if r and "docs_fulfill" in r]
        assert len(provenanced) == 2


class TestGraphFlag:
    async def test_mcp_surface_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        memory = await LocalMemory.create(str(tmp_path / "m.db"))
        saved = await memory.save(mem_type="belief", subject="user",
                                  relation="uses", object="pytest")
        assert saved["action"] == "saved"  # memory core unaffected
        entity = await memory.graph_entity("user")
        link = await memory.memory_link("a", "uses", "b")
        merge = await memory.graph_merge("a", "b")
        search = await memory.search("pytest", graph_depth=1)
        conn = sqlite3.connect(tmp_path / "m.db")
        try:
            (edges,) = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()
        finally:
            conn.close()
        await memory.close()
        assert entity["disabled"] and "CLARA_GRAPH_ENABLED" in entity["error"]
        assert link["disabled"] and merge["disabled"]
        assert edges == 0  # projection no-oped
        assert "[GRAPH]" not in search["context"]

    async def test_export_raises_with_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        memory = await LocalMemory.create(str(tmp_path / "m.db"))
        with pytest.raises(RuntimeError, match="CLARA_GRAPH_ENABLED"):
            await memory.graph_export()
        with pytest.raises(RuntimeError, match="CLARA_GRAPH_ENABLED"):
            await memory.graph_rebuild()
        await memory.close()

    def test_cli_prints_reenable_hint(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        assert _run_cli(["graph", "stats"]) == 2
        assert "CLARA_GRAPH_ENABLED=1" in capsys.readouterr().err

    async def test_reenabled_after_flag_flip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        memory = await LocalMemory.create(str(tmp_path / "m.db"))
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="uv")
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "1")
        rebuilt = await memory.graph_rebuild()
        card = await memory.graph_entity("user")
        await memory.close()
        assert rebuilt["edges"] == 1  # rebuild recovers the missed projection
        assert card["found"]


@requires_git
class TestDocsFlag:
    async def _scanned(self, tmp_path, monkeypatch):
        from tests.test_docs import _make_repo

        root = tmp_path / "repo"
        _make_repo(root)
        (root / "docs").mkdir()
        (root / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
        _commit_all(root, "docs")
        db = tmp_path / "global" / "clara.db"
        db.parent.mkdir()
        monkeypatch.setenv("CLARA_DB_PATH", str(db))
        monkeypatch.setenv("CLARA_HOME", str(db.parent))
        scan_repo(str(db), str(root), probe_symbols=False)
        docs_map.build_map(str(root))
        return root, db

    async def test_mcp_surface_disabled(self, tmp_path, monkeypatch):
        root, db = await self._scanned(tmp_path, monkeypatch)
        monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
        memory = await LocalMemory.create(str(db))
        for result in (
            await memory.docs_report(str(root)),
            await memory.docs_classify(str(root), path="docs/a.md",
                                       doc_type="guide", rationale="x"),
            await memory.docs_fulfill(str(root), path="docs/a.md",
                                      distilled=[{"mem_type": "belief",
                                                  "subject": "a", "relation": "is",
                                                  "object": "b"}]),
        ):
            assert result["disabled"] and "CLARA_DOCS_ENABLED" in result["error"]
        await memory.close()

    async def test_scan_and_fastpath_disabled(self, tmp_path, monkeypatch):
        root, db = await self._scanned(tmp_path, monkeypatch)
        monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
        with pytest.raises(RuntimeError, match="CLARA_DOCS_ENABLED"):
            scan_repo(str(db), str(root))
        assert docs_map.build_map(str(root)) is None  # map omitted

    def test_cli_prints_reenable_hint(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
        monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
        monkeypatch.chdir(tmp_path)
        assert _run_cli(["docs", "report"]) == 2
        assert "CLARA_DOCS_ENABLED=1" in capsys.readouterr().err

    async def test_hooks_short_circuit(self, tmp_path, monkeypatch):
        from tests.test_plugin_layout import _working_bash

        bash = _working_bash()
        if bash is None:
            pytest.skip("no working bash")
        root, db = await self._scanned(tmp_path, monkeypatch)
        script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "CLARA_HOME": str(db.parent),
               "CLARA_DOCS_ENABLED": "0", "CLAUDE_SESSION_ID": "flagged"}
        for script, payload in (
            ("session-stop.sh", None),
            ("read-annotate.sh",
             json.dumps({"tool_input": {"file_path": str(root / "docs" / "a.md")}})),
        ):
            proc = subprocess.run(
                [bash, os.path.join(script_root, "scripts", script)],
                input=payload, cwd=str(root), env=env,
                capture_output=True, text=True, timeout=30,
            )
            assert proc.returncode == 0
            assert proc.stdout == ""


class TestCoreSurvivesBothFlagsOff:
    async def test_memory_core_intact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
        memory = await LocalMemory.create(str(tmp_path / "m.db"))
        await memory.save(mem_type="belief", subject="user", relation="prefers",
                          object="ripgrep")
        result = await memory.search("ripgrep")
        stats = await memory.stats()
        await memory.close()
        assert result["total"] == 1
        assert stats["active_by_type"]["belief"] == 1

    def test_fastpath_context_intact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        monkeypatch.setenv("CLARA_DOCS_ENABLED", "0")
        memories = [{
            "type": "belief", "confidence": 0.9, "updated_epoch": 1_700_000_000,
            "created_at": "2026-01-01",
            "content": {"subject": "user", "relation": "uses", "object": "sqlite"},
            "metadata": {},
        }]
        assert rank(memories, now_epoch=1_700_000_000)


class TestDoctorPluginHealth:
    def test_doctor_reports_health_block(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CLARA_DB_PATH", str(tmp_path / "clara.db"))
        monkeypatch.setenv("CLARA_GRAPH_ENABLED", "0")
        assert _run_cli(["init", "--agent", "generic"]) == 0
        capsys.readouterr()
        code = _run_cli(["doctor"])
        out = capsys.readouterr().out
        from clara.db.migrations import SCHEMA_VERSION

        assert code in (0, 1)
        assert "plugin:" in out
        assert f"schema: v{SCHEMA_VERSION}" in out
        assert "flags: graph=OFF docs=on" in out
        assert "quarantine manifests:" in out
        assert "venv:" in out
