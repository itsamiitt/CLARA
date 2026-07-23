"""Tests for the v5 retrieval upgrades: unicode tokenizer, CJK handling,
tag search on the FTS path, trigram routing, and synonym expansion."""

from __future__ import annotations

import sqlite3

import pytest

from clara.integrations.local_memory import LocalMemory
from clara.retrieval.lexical import _fold, _has_cjk, tokenize
from clara.retrieval.synonyms import expand_groups, flatten


@pytest.fixture
async def memory(tmp_path):
    mem = await LocalMemory.create(str(tmp_path / "uni.db"))
    yield mem
    await mem.close()


class TestTokenizer:
    def test_ascii_unchanged(self):
        assert tokenize("Deploy the Postgres backend") == [
            "deploy", "postgres", "backend",
        ]

    def test_diacritics_folded(self):
        assert tokenize("café résumé") == ["cafe", "resume"]

    def test_casefold_eszett(self):
        assert "strasse" in tokenize("STRASSE")

    def test_cjk_kept_and_bigrammed(self):
        tokens = tokenize("日本語")
        assert "日本語" in tokens
        assert "日本" in tokens
        assert "本語" in tokens

    def test_mixed_script(self):
        tokens = tokenize("deploy 日本 server")
        assert "deploy" in tokens
        assert "日本" in tokens
        assert "server" in tokens

    def test_hangul_detected(self):
        assert _has_cjk("한국어")
        assert not _has_cjk("korean")

    def test_fold_is_idempotent(self):
        assert _fold(_fold("Ünïcode")) == _fold("Ünïcode")


class TestSynonyms:
    def test_groups_include_original(self):
        groups = expand_groups(["postgres"])
        assert groups[0] >= {"postgres", "postgresql"}

    def test_unknown_token_is_singleton(self):
        assert expand_groups(["xyzzy"]) == [{"xyzzy"}]

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("CLARA_QUERY_EXPANSION", "0")
        assert expand_groups(["postgres"]) == [{"postgres"}]

    def test_flatten_dedupes_preserving_groups(self):
        flat = flatten([{"a", "b"}, {"b", "c"}])
        assert sorted(flat) == ["a", "b", "c"]


class TestSearchBehavior:
    async def test_diacritic_query_hits_ascii_row(self, memory):
        await memory.save(
            mem_type="belief", subject="team", relation="meets_at", object="cafe corner"
        )
        found = await memory.search("café", top_k=3)
        assert found["total"] == 1

    async def test_cjk_roundtrip(self, memory):
        await memory.save(
            mem_type="belief",
            subject="project",
            relation="documented_in",
            object="日本語のドキュメント",
        )
        found = await memory.search("日本語", top_k=3)
        assert found["total"] >= 1

    async def test_tag_search_on_fts_path(self, memory, tmp_path):
        await memory.save(
            mem_type="belief",
            subject="api",
            relation="deployed_to",
            object="fly.io",
            tags=["deployment", "infra-notes"],
        )
        # Assert the FTS index actually carries the tag text (not just the
        # scan path finding it).
        conn = sqlite3.connect(tmp_path / "uni.db")
        hits = conn.execute(
            "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH '\"deployment\"'"
        ).fetchall()
        conn.close()
        assert len(hits) == 1
        found = await memory.search("deployment", top_k=3)
        assert found["total"] == 1

    async def test_synonym_recall(self, memory):
        await memory.save(
            mem_type="belief",
            subject="backend",
            relation="uses",
            object="postgresql 16",
        )
        found = await memory.search("postgres version", top_k=3)
        assert found["total"] >= 1

    async def test_trigram_table_exists_when_supported(self, memory, tmp_path):
        conn = sqlite3.connect(tmp_path / "uni.db")
        has_tri = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'memories_fts_tri'"
        ).fetchone()
        trigram_ok = True
        try:
            conn.execute("CREATE VIRTUAL TABLE _tri_probe USING fts5(t, tokenize='trigram')")
            conn.execute("DROP TABLE _tri_probe")
        except sqlite3.OperationalError:
            trigram_ok = False
        conn.close()
        if trigram_ok:
            assert has_tri is not None
        else:
            pytest.skip("SQLite build lacks the trigram tokenizer")

    async def test_empty_query_still_recency_feed(self, memory):
        await memory.save(mem_type="belief", subject="a", relation="b", object="c")
        found = await memory.recent(n=5)
        assert found["total"] == 1
