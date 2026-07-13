"""Tests for the rule-based (zero-LLM) extractor."""

from __future__ import annotations

import pytest

from clara.extraction.heuristic import HeuristicExtractor
from clara.update.engine import classify_memory_type
from clara.db.models import MemoryType


@pytest.fixture
def extractor() -> HeuristicExtractor:
    return HeuristicExtractor()


def _triples(result):
    return {(f.subject, f.relation, f.object, f.is_negation) for f in result}


class TestPatterns:
    def test_usage(self, extractor):
        result = extractor.extract_sync("I use PostgreSQL for the backend.")
        assert ("user", "uses", "PostgreSQL", False) in _triples(result)
        assert result[0].domain == "the backend"

    def test_preference(self, extractor):
        result = extractor.extract_sync("We prefer tabs over spaces.")
        assert ("user", "prefers", "tabs over spaces", False) in _triples(result)

    def test_switch_produces_negation_pair(self, extractor):
        result = extractor.extract_sync("I switched from npm to pnpm.")
        triples = _triples(result)
        assert ("user", "uses", "npm", True) in triples
        assert ("user", "uses", "pnpm", False) in triples

    def test_stopped_using(self, extractor):
        result = extractor.extract_sync("We stopped using Redux last year")
        assert ("user", "uses", "Redux last year", True) in _triples(result) or (
            "user", "uses", "Redux", True) in _triples(result)

    def test_skill(self, extractor):
        result = extractor.extract_sync("I learned Kubernetes.")
        [fact] = list(result)
        assert fact.relation == "learned"
        assert classify_memory_type(fact) == MemoryType.skill

    def test_event(self, extractor):
        result = extractor.extract_sync("We deployed the payments service.")
        [fact] = list(result)
        assert fact.relation == "deployed"
        assert classify_memory_type(fact) == MemoryType.event

    def test_world_model_runs_on(self, extractor):
        result = extractor.extract_sync("The api service runs on Fly.io")
        [fact] = list(result)
        assert fact.relation == "runs_on"
        assert classify_memory_type(fact) == MemoryType.world_model

    def test_is_a_identity(self, extractor):
        result = extractor.extract_sync("CLARA is a memory system")
        [fact] = list(result)
        assert fact.relation == "is_a"
        assert fact.subject == "CLARA"


class TestFilters:
    def test_hedged_statements_skipped(self, extractor):
        assert extractor.extract_sync("I think I might use Rust maybe.") == []
        assert extractor.extract_sync("We should probably use Postgres.") == []

    def test_empty_input(self, extractor):
        result = extractor.extract_sync("   ")
        assert result == []
        assert result.status == "empty"

    def test_no_match_is_ok_status(self, extractor):
        result = extractor.extract_sync("The weather is nice today?")
        assert result.status == "ok"

    def test_dedup_within_text(self, extractor):
        result = extractor.extract_sync("I use Rust. I use Rust.")
        assert len(result) == 1

    def test_never_raises(self, extractor):
        for text in ("", "\x00\x01", "🎉" * 100, "I use " + "x" * 500, "is a is a"):
            extractor.extract_sync(text)  # must not raise

    def test_multi_sentence(self, extractor):
        result = extractor.extract_sync(
            "I use Neovim. We deployed staging yesterday. I learned Terraform."
        )
        assert len(result) == 3


class TestApiCompat:
    async def test_async_extract(self, extractor):
        result = await extractor.extract("I use SQLite")
        assert len(result) == 1
        assert result.status == "ok"

    def test_provider_marker(self, extractor):
        assert extractor._provider == "none"
        assert extractor._model is None
