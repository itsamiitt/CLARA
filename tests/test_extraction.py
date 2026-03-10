"""
Tests for clara.extraction.extractor — FactExtractor with mocked LLM responses.

No real LLM API calls are made. Tests validate:
* Correct parsing of well-formed JSON responses
* Confidence filtering (< 0.4 discarded)
* Negation flagging
* Hedged statement avoidance (via prompt, verified in system prompt content)
* Graceful handling of malformed JSON, empty responses, and LLM errors
* Provider selection (OpenAI / Anthropic)
* Edge cases: markdown code fences, wrapped objects, empty input
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from clara.extraction.extractor import (
    CONFIDENCE_FLOOR,
    ENV_ANTHROPIC_KEY,
    ENV_LLM_PROVIDER,
    ENV_OPENAI_KEY,
    SYSTEM_PROMPT,
    ExtractedFact,
    FactExtractor,
    _parse_llm_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_FACTS_JSON = json.dumps([
    {
        "subject": "user",
        "relation": "uses",
        "object": "Rust",
        "domain": "systems",
        "source_type": "user_direct",
        "confidence": 0.9,
        "is_negation": False,
    },
    {
        "subject": "user",
        "relation": "uses",
        "object": "Python",
        "domain": "systems",
        "source_type": "user_direct",
        "confidence": 0.85,
        "is_negation": True,
    },
])

SAMPLE_TEXT = "I switched from Python to Rust for my systems work."


# ---------------------------------------------------------------------------
# ExtractedFact dataclass
# ---------------------------------------------------------------------------

class TestExtractedFact:
    def test_fields(self):
        fact = ExtractedFact(
            subject="user",
            relation="uses",
            object="Rust",
            domain="systems",
            source_type="user_direct",
            confidence=0.9,
            is_negation=False,
            raw_text="I use Rust.",
        )
        assert fact.subject == "user"
        assert fact.relation == "uses"
        assert fact.object == "Rust"
        assert fact.domain == "systems"
        assert fact.confidence == 0.9
        assert fact.is_negation is False
        assert fact.raw_text == "I use Rust."

    def test_immutable(self):
        fact = ExtractedFact(
            subject="user", relation="uses", object="Rust",
            domain=None, source_type="user_direct",
            confidence=0.9, is_negation=False, raw_text="test",
        )
        with pytest.raises(AttributeError):
            fact.subject = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _parse_llm_response — JSON parsing
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    """Tests for the internal JSON parser."""

    def test_valid_json_array(self):
        facts = _parse_llm_response(SAMPLE_FACTS_JSON, SAMPLE_TEXT)
        assert len(facts) == 2
        assert facts[0].subject == "user"
        assert facts[0].relation == "uses"
        assert facts[0].object == "Rust"
        assert facts[1].is_negation is True

    def test_raw_text_propagated(self):
        facts = _parse_llm_response(SAMPLE_FACTS_JSON, SAMPLE_TEXT)
        assert all(f.raw_text == SAMPLE_TEXT for f in facts)

    def test_empty_array(self):
        facts = _parse_llm_response("[]", "hello")
        assert facts == []

    def test_malformed_json_returns_empty(self):
        facts = _parse_llm_response("this is not json {{{", "hello")
        assert facts == []

    def test_json_object_with_facts_key(self):
        wrapped = json.dumps({"facts": json.loads(SAMPLE_FACTS_JSON)})
        facts = _parse_llm_response(wrapped, SAMPLE_TEXT)
        assert len(facts) == 2

    def test_json_object_with_results_key(self):
        wrapped = json.dumps({"results": json.loads(SAMPLE_FACTS_JSON)})
        facts = _parse_llm_response(wrapped, SAMPLE_TEXT)
        assert len(facts) == 2

    def test_json_object_unknown_key_returns_empty(self):
        wrapped = json.dumps({"unknown_key": [1, 2, 3]})
        facts = _parse_llm_response(wrapped, SAMPLE_TEXT)
        assert facts == []

    def test_markdown_fenced_json(self):
        fenced = f"```json\n{SAMPLE_FACTS_JSON}\n```"
        facts = _parse_llm_response(fenced, SAMPLE_TEXT)
        assert len(facts) == 2

    def test_confidence_below_threshold_discarded(self):
        low_conf = json.dumps([
            {
                "subject": "user", "relation": "might_use", "object": "Go",
                "domain": None, "source_type": "agent_inference",
                "confidence": 0.3, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(low_conf, "maybe Go")
        assert facts == []

    def test_confidence_at_threshold_kept(self):
        at_threshold = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "Go",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.4, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(at_threshold, "I use Go")
        assert len(facts) == 1

    def test_empty_subject_discarded(self):
        no_subject = json.dumps([
            {
                "subject": "", "relation": "uses", "object": "Rust",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(no_subject, "test")
        assert facts == []

    def test_empty_relation_discarded(self):
        no_relation = json.dumps([
            {
                "subject": "user", "relation": "", "object": "Rust",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(no_relation, "test")
        assert facts == []

    def test_empty_object_discarded(self):
        no_object = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(no_object, "test")
        assert facts == []

    def test_domain_null_allowed(self):
        no_domain = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "Rust",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
        ])
        facts = _parse_llm_response(no_domain, "test")
        assert len(facts) == 1
        assert facts[0].domain is None

    def test_non_list_json_returns_empty(self):
        facts = _parse_llm_response('"just a string"', "test")
        assert facts == []

    def test_malformed_item_skipped(self):
        """Items that can't be converted to ExtractedFact should be skipped."""
        mixed = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "Rust",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
            {
                "subject": "user", "relation": "uses", "object": "Go",
                "domain": None, "source_type": "user_direct",
                "confidence": "not_a_number",  # will cause ValueError
                "is_negation": False,
            },
        ])
        facts = _parse_llm_response(mixed, "test")
        # First item valid, second skipped due to bad confidence
        assert len(facts) == 1
        assert facts[0].object == "Rust"


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Verify the system prompt enforces CONTEXT.md rules."""

    def test_instructs_json_only(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT
        assert "no markdown" in SYSTEM_PROMPT.lower()

    def test_instructs_ignore_hedged(self):
        for word in ("maybe", "might", "I think", "perhaps"):
            assert word in SYSTEM_PROMPT

    def test_instructs_flag_negations(self):
        assert "negation" in SYSTEM_PROMPT.lower()
        assert "is_negation" in SYSTEM_PROMPT

    def test_instructs_domain_context(self):
        assert "domain" in SYSTEM_PROMPT.lower()

    def test_instructs_confidence(self):
        assert "confidence" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# FactExtractor — OpenAI provider
# ---------------------------------------------------------------------------

class TestFactExtractorOpenAI:
    """Tests for FactExtractor with mocked OpenAI calls."""

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_extract_returns_facts(self, mock_openai):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = SAMPLE_FACTS_JSON
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract(SAMPLE_TEXT)

        assert len(facts) == 2
        assert facts[0].object == "Rust"
        assert facts[1].is_negation is True
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_extract_empty_text_returns_empty(self, mock_openai):
        extractor = FactExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_extract_llm_error_returns_empty(self, mock_openai):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract("Some text about Rust")

        assert facts == []

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_extract_unparseable_response_returns_empty(self, mock_openai):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "I cannot extract facts from this."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract("Some text")

        assert facts == []

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_extract_filters_low_confidence(self, mock_openai):
        low_and_high = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "Rust",
                "domain": None, "source_type": "user_direct",
                "confidence": 0.9, "is_negation": False,
            },
            {
                "subject": "user", "relation": "might_use", "object": "Zig",
                "domain": None, "source_type": "agent_inference",
                "confidence": 0.2, "is_negation": False,
            },
        ])
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = low_and_high
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract("I use Rust, might try Zig")

        assert len(facts) == 1
        assert facts[0].object == "Rust"

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_system_prompt_included_in_call(self, mock_openai):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "[]"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        extractor.extract("test input")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "JSON" in system_msgs[0]["content"]

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test", ENV_LLM_PROVIDER: "openai"})
    @patch("clara.extraction.extractor._openai")
    def test_negation_facts_preserved(self, mock_openai):
        negation_json = json.dumps([
            {
                "subject": "user", "relation": "uses", "object": "Python",
                "domain": "systems", "source_type": "user_direct",
                "confidence": 0.85, "is_negation": True,
            },
        ])
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = negation_json
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai.OpenAI.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract("I no longer use Python for systems work")

        assert len(facts) == 1
        assert facts[0].is_negation is True
        assert facts[0].object == "Python"


# ---------------------------------------------------------------------------
# FactExtractor — Anthropic provider
# ---------------------------------------------------------------------------

class TestFactExtractorAnthropic:
    """Tests for FactExtractor with mocked Anthropic calls."""

    @patch.dict(os.environ, {
        ENV_ANTHROPIC_KEY: "sk-ant-test",
        ENV_LLM_PROVIDER: "anthropic",
    })
    @patch("clara.extraction.extractor._anthropic")
    def test_extract_returns_facts(self, mock_anthropic):
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = SAMPLE_FACTS_JSON
        mock_client.messages.create.return_value = MagicMock(
            content=[mock_block]
        )
        mock_anthropic.Anthropic.return_value = mock_client

        extractor = FactExtractor()
        facts = extractor.extract(SAMPLE_TEXT)

        assert len(facts) == 2
        mock_client.messages.create.assert_called_once()

    @patch.dict(os.environ, {
        ENV_ANTHROPIC_KEY: "sk-ant-test",
        ENV_LLM_PROVIDER: "anthropic",
    })
    @patch("clara.extraction.extractor._anthropic")
    def test_anthropic_system_prompt_passed(self, mock_anthropic):
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "[]"
        mock_client.messages.create.return_value = MagicMock(
            content=[mock_block]
        )
        mock_anthropic.Anthropic.return_value = mock_client

        extractor = FactExtractor()
        extractor.extract("hello")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("system") == SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            FactExtractor(provider="gpt5")

    @patch.dict(os.environ, {ENV_LLM_PROVIDER: "openai", ENV_OPENAI_KEY: "sk-test"})
    def test_default_provider_is_openai(self):
        with patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test"}, clear=True):
            extractor = FactExtractor()
            assert extractor._provider == "openai"

    @patch.dict(os.environ, {ENV_LLM_PROVIDER: "anthropic"})
    def test_anthropic_provider_selected(self):
        extractor = FactExtractor()
        assert extractor._provider == "anthropic"

    def test_explicit_provider_override(self):
        extractor = FactExtractor(provider="anthropic")
        assert extractor._provider == "anthropic"

    def test_explicit_model_override(self):
        extractor = FactExtractor(provider="openai", model="gpt-4o")
        assert extractor._model == "gpt-4o"

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test"})
    @patch("clara.extraction.extractor._openai", None)
    def test_missing_openai_package_raises_in_call(self):
        """Calling the internal _call_openai when the package is None raises."""
        from clara.extraction.extractor import _call_openai
        with pytest.raises(ImportError, match="openai"):
            _call_openai("test", "gpt-4o-mini")

    @patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-test"})
    @patch("clara.extraction.extractor._openai", None)
    def test_missing_openai_package_extract_returns_empty(self):
        """extract() should catch ImportError and return [] gracefully."""
        extractor = FactExtractor(provider="openai")
        assert extractor.extract("test text") == []

    @patch.dict(os.environ, {ENV_ANTHROPIC_KEY: "sk-ant-test"})
    @patch("clara.extraction.extractor._anthropic", None)
    def test_missing_anthropic_package_raises_in_call(self):
        from clara.extraction.extractor import _call_anthropic
        with pytest.raises(ImportError, match="anthropic"):
            _call_anthropic("test", "claude-3-5-haiku-20241022")

    @patch.dict(os.environ, {ENV_ANTHROPIC_KEY: "sk-ant-test"})
    @patch("clara.extraction.extractor._anthropic", None)
    def test_missing_anthropic_package_extract_returns_empty(self):
        extractor = FactExtractor(provider="anthropic")
        assert extractor.extract("test text") == []

    @patch.dict(os.environ, {}, clear=True)
    @patch("clara.extraction.extractor._openai", MagicMock())
    def test_missing_openai_key_raises_in_call(self):
        from clara.extraction.extractor import _call_openai
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            _call_openai("test", "gpt-4o-mini")

    @patch.dict(os.environ, {}, clear=True)
    @patch("clara.extraction.extractor._openai", MagicMock())
    def test_missing_openai_key_extract_returns_empty(self):
        extractor = FactExtractor(provider="openai")
        assert extractor.extract("test text") == []

    @patch.dict(os.environ, {}, clear=True)
    @patch("clara.extraction.extractor._anthropic", MagicMock())
    def test_missing_anthropic_key_raises_in_call(self):
        from clara.extraction.extractor import _call_anthropic
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            _call_anthropic("test", "claude-3-5-haiku-20241022")

    @patch.dict(os.environ, {}, clear=True)
    @patch("clara.extraction.extractor._anthropic", MagicMock())
    def test_missing_anthropic_key_extract_returns_empty(self):
        extractor = FactExtractor(provider="anthropic")
        assert extractor.extract("test text") == []

