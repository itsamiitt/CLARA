"""Tests for the native-memory bridge (clara/bridge/)."""

from __future__ import annotations

import time

import pytest

from clara.bridge import markers, paths
from clara.bridge.exporter import build_section_body, export_native
from clara.bridge.importer import candidate_lines, import_native
from clara.integrations.local_memory import LocalMemory


class TestMarkers:
    def test_fresh_file_gets_fence(self):
        result = markers.replace_section("", "body line\n", store="global", ts="T")
        assert result.changed
        assert result.text.startswith("<!-- clara:begin")
        assert "body line" in result.text
        assert result.text.rstrip().endswith(markers.END_MARKER)

    def test_existing_content_preserved(self):
        original = "# My notes\n\n- user note\n"
        result = markers.replace_section(original, "clara\n", store="global", ts="T")
        assert result.text.startswith("# My notes")
        assert "- user note" in result.text

    def test_replace_updates_only_fence(self):
        first = markers.replace_section("# top\n", "v1\n", store="global", ts="T1")
        second = markers.replace_section(first.text, "v2\n", store="global", ts="T2")
        assert second.changed
        assert "v2" in second.text
        assert "v1" not in second.text
        assert second.text.startswith("# top")

    def test_unchanged_body_is_noop(self):
        first = markers.replace_section("", "same\n", store="global", ts="T1")
        second = markers.replace_section(first.text, "same\n", store="global", ts="T2")
        assert not second.changed
        assert second.conflict_body is None

    def test_hand_edited_fence_reports_conflict(self):
        first = markers.replace_section("", "original\n", store="global", ts="T1")
        tampered = first.text.replace("original", "edited by hand")
        result = markers.replace_section(tampered, "new\n", store="global", ts="T2")
        assert not result.changed
        assert result.conflict_body is not None
        assert "edited by hand" in result.conflict_body
        assert result.text == tampered  # never overwrites an edited fence

    def test_strip_fences(self):
        first = markers.replace_section("keep me\n", "fenced\n", store="global", ts="T")
        stripped = markers.strip_fences(first.text)
        assert "keep me" in stripped
        assert "fenced" not in stripped


class TestPaths:
    def test_encoding_matches_live_convention(self):
        assert (
            paths.encode_project_dir(r"C:\Users\Administrator\Downloads\CLARA\CLARA")
            == "C--Users-Administrator-Downloads-CLARA-CLARA"
        )
        assert paths.encode_project_dir("/home/user/my proj (2)") == "-home-user-my-proj--2-"

    def test_disable_env_var(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")
        assert paths.auto_memory_enabled() is False
        assert paths.auto_memory_dir(".") is None


class TestImportCandidates:
    def test_bullets_and_paragraphs(self):
        text = (
            "# Heading skipped\n\n"
            "- user prefers pnpm over npm\n"
            "1. numbered bullet line here\n"
            "plain paragraph line that is long enough\n"
            "```\ncode line ignored entirely\n```\n"
            "short\n"
        )
        lines = candidate_lines(text)
        assert "user prefers pnpm over npm" in lines
        assert "numbered bullet line here" in lines
        assert "plain paragraph line that is long enough" in lines
        assert all("code line" not in line for line in lines)
        assert all(line != "short" for line in lines)

    def test_fenced_and_tagged_lines_never_import(self):
        fenced = markers.replace_section(
            "- outside line stays importable\n",
            "- inside fence never imports\n",
            store="global",
            ts="T",
        )
        text = fenced.text + "\n- tagged line <!--clara:12ab34cd--> here\n"
        lines = candidate_lines(text)
        assert any("outside line" in line for line in lines)
        assert all("inside fence" not in line for line in lines)
        assert all("tagged line" not in line for line in lines)


@pytest.fixture
def native_home(tmp_path, monkeypatch):
    """Fake $HOME with a .claude tree + repo anchor, plus isolated store."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", raising=False)
    monkeypatch.delenv("CLARA_DB_PATH", raising=False)
    monkeypatch.setenv("CLARA_HOME", str(home / ".clara"))
    repo = tmp_path / "repo"
    repo.mkdir()
    return home, repo


class TestExportNative:
    async def test_export_creates_fence_and_topic(self, native_home, tmp_path):
        home, repo = native_home
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="user", relation="prefers", object="uv"
            )
        finally:
            await memory.close()

        summary = export_native(db, str(repo))
        assert "MEMORY.md updated" in summary

        memory_md = paths.memory_md_path(str(repo))
        assert memory_md is not None and memory_md.is_file()
        content = memory_md.read_text(encoding="utf-8")
        assert "clara:begin" in content
        assert "user prefers uv" in content
        assert "<!--clara:" in content  # provenance tag

        topic = paths.topic_file_path(str(repo))
        assert topic is not None and topic.is_file()
        assert "user prefers uv" in topic.read_text(encoding="utf-8")

    async def test_export_idempotent(self, native_home, tmp_path):
        home, repo = native_home
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="a", relation="b", object="c"
            )
        finally:
            await memory.close()
        export_native(db, str(repo))
        memory_md = paths.memory_md_path(str(repo))
        before = memory_md.read_text(encoding="utf-8")
        summary = export_native(db, str(repo))
        assert "unchanged" in summary
        assert memory_md.read_text(encoding="utf-8") == before

    async def test_export_preserves_claude_notes(self, native_home, tmp_path):
        home, repo = native_home
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="a", relation="b", object="c"
            )
        finally:
            await memory.close()
        memory_md = paths.memory_md_path(str(repo))
        memory_md.parent.mkdir(parents=True, exist_ok=True)
        memory_md.write_text("# Claude's own notes\n\n- native note kept\n",
                             encoding="utf-8")
        export_native(db, str(repo))
        content = memory_md.read_text(encoding="utf-8")
        assert content.startswith("# Claude's own notes")
        assert "- native note kept" in content
        assert "clara:begin" in content

    def test_export_disabled(self, native_home, tmp_path, monkeypatch):
        home, repo = native_home
        monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")
        assert export_native(str(tmp_path / "none.db"), str(repo)) == "auto-memory disabled"

    def test_section_budget(self):
        now = time.time()
        memories = [
            {
                "type": "belief",
                "content": {"subject": f"svc{i}", "relation": "uses", "object": "x" * 80},
                "confidence": 0.9,
                "metadata": {},
                "created_at": "2026-01-01",
                "updated_epoch": int(now),
                "memory_id": f"{i:032x}",
            }
            for i in range(500)
        ]
        body = build_section_body(memories, now)
        assert len(body.splitlines()) <= 60
        assert len(body.encode("utf-8")) <= 6 * 1024


class TestImportNative:
    async def test_import_claude_md(self, native_home, tmp_path):
        home, repo = native_home
        (repo / "CLAUDE.md").write_text(
            "# Project\n\n- We use PostgreSQL for the backend\n", encoding="utf-8"
        )
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            results = await import_native(memory, str(repo))
            imported = sum(r["imported"] for r in results.values())
            assert imported >= 1
            found = await memory.search("postgresql", top_k=5)
            assert found["total"] >= 1
            # Re-import dedups on origin_hash.
            again = await import_native(memory, str(repo))
            assert sum(r["imported"] for r in again.values()) == 0
        finally:
            await memory.close()

    async def test_round_trip_stability(self, native_home, tmp_path):
        """export -> import -> export must not change the fence (no echo)."""
        home, repo = native_home
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="team", relation="uses", object="pytest"
            )
            export_native(db, str(repo))
            memory_md = paths.memory_md_path(str(repo))
            first = memory_md.read_text(encoding="utf-8")
            await import_native(memory, str(repo))
            summary = export_native(db, str(repo))
            assert "unchanged" in summary
            assert memory_md.read_text(encoding="utf-8") == first
        finally:
            await memory.close()

    async def test_verbatim_flag(self, native_home, tmp_path):
        home, repo = native_home
        (repo / "CLAUDE.md").write_text(
            "- zqxjk flumble wandersnatch quorble grimble\n", encoding="utf-8"
        )
        db = str(tmp_path / "store.db")
        memory = await LocalMemory.create(db)
        try:
            strict = await import_native(memory, str(repo))
            assert sum(r["imported"] for r in strict.values()) == 0
            loose = await import_native(memory, str(repo), verbatim=True)
            assert sum(r["imported"] for r in loose.values()) == 1
        finally:
            await memory.close()
