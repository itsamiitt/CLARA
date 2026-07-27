"""Tests for the shared provider plumbing (clara/core/llm.py).

Audit finding A1: extraction, reasoning and reflection each carried their own
copy of "call the provider and return the text", and the copies had drifted.
The drift that mattered was timeouts -- reflection passed none, and *every*
Ollama client passed none, which for ollama 0.6.2 means no timeout at all
rather than a library default. These tests pin the client construction
arguments, because that is exactly what silently regressed before.

The fakes record their constructor kwargs rather than mocking the call, so an
omitted timeout fails here instead of in production at 03:00 UTC.
"""

from __future__ import annotations

import asyncio

import pytest

from clara.core import llm


class FakeOpenAI:
    """Stands in for the `openai` module."""

    def __init__(self, content: str = "hello"):
        self.content = content
        self.client_kwargs: dict = {}
        self.call_kwargs: dict = {}

        module = self

        class _Message:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.message = _Message(content)

        class _Response:
            def __init__(self, content):
                self.choices = [_Choice(content)]

        class _Completions:
            async def create(self, **kwargs):
                module.call_kwargs = kwargs
                return _Response(module.content)

        class _Chat:
            completions = _Completions()

        class _AsyncOpenAI:
            def __init__(self, **kwargs):
                module.client_kwargs = kwargs
                self.chat = _Chat()

        self.AsyncOpenAI = _AsyncOpenAI


class FakeAnthropic:
    def __init__(self, text: str = "hello"):
        self.text = text
        self.client_kwargs: dict = {}
        self.call_kwargs: dict = {}

        module = self

        class _Block:
            def __init__(self, text):
                self.text = text

        class _Response:
            def __init__(self, text):
                self.content = [_Block(text)]

        class _Messages:
            async def create(self, **kwargs):
                module.call_kwargs = kwargs
                return _Response(module.text)

        class _AsyncAnthropic:
            def __init__(self, **kwargs):
                module.client_kwargs = kwargs
                self.messages = _Messages()

        self.AsyncAnthropic = _AsyncAnthropic


class FakeOllama:
    def __init__(self, response=None):
        self.response = response
        self.client_kwargs: dict = {}
        self.call_kwargs: dict = {}

        module = self

        class _Client:
            def __init__(self, **kwargs):
                module.client_kwargs = kwargs

            def chat(self, **kwargs):
                module.call_kwargs = kwargs
                return module.response

        self.Client = _Client


class TestGuards:
    def test_require_sdk_names_package_and_purpose(self):
        with pytest.raises(ImportError, match="openai.*OpenAI reasoning provider"):
            llm.require_sdk(None, "openai", "OpenAI reasoning provider")

    def test_require_sdk_passes_through(self):
        sentinel = object()
        assert llm.require_sdk(sentinel, "openai", "x") is sentinel

    def test_require_api_key_names_the_variable(self):
        with pytest.raises(OSError, match="OPENAI_API_KEY"):
            llm.require_api_key("OPENAI_API_KEY", lambda _n: None)

    def test_require_api_key_rejects_empty_string(self):
        """An empty value is a misconfiguration, not a key."""
        with pytest.raises(OSError):
            llm.require_api_key("OPENAI_API_KEY", lambda _n: "")


class TestOpenAI:
    def test_timeout_and_retries_are_always_passed(self):
        sdk = FakeOpenAI()
        asyncio.run(
            llm.openai_chat(
                sdk, system="s", user="u", model="m", api_key="k", temperature=0.2
            )
        )
        assert sdk.client_kwargs["timeout"] == llm.LLM_TIMEOUT_SECONDS
        assert sdk.client_kwargs["max_retries"] == llm.LLM_MAX_RETRIES
        assert sdk.client_kwargs["api_key"] == "k"

    def test_messages_and_temperature(self):
        sdk = FakeOpenAI()
        asyncio.run(
            llm.openai_chat(
                sdk, system="sys", user="usr", model="m", api_key="k", temperature=0.7
            )
        )
        assert sdk.call_kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        assert sdk.call_kwargs["temperature"] == 0.7
        assert sdk.call_kwargs["model"] == "m"

    def test_json_mode_is_opt_in(self):
        sdk = FakeOpenAI()
        asyncio.run(
            llm.openai_chat(
                sdk, system="s", user="u", model="m", api_key="k", temperature=0.0
            )
        )
        assert "response_format" not in sdk.call_kwargs

        sdk = FakeOpenAI()
        asyncio.run(
            llm.openai_chat(
                sdk, system="s", user="u", model="m", api_key="k",
                temperature=0.0, json_mode=True,
            )
        )
        assert sdk.call_kwargs["response_format"] == {"type": "json_object"}

    def test_empty_content_is_returned_not_coerced(self):
        """Coercing "" to a placeholder hid failed calls as empty successes."""
        sdk = FakeOpenAI(content=None)
        out = asyncio.run(
            llm.openai_chat(
                sdk, system="s", user="u", model="m", api_key="k", temperature=0.0
            )
        )
        assert out == ""


class TestAnthropic:
    def test_timeout_retries_and_max_tokens(self):
        sdk = FakeAnthropic()
        asyncio.run(
            llm.anthropic_chat(
                sdk, system="s", user="u", model="m", api_key="k",
                temperature=0.2, max_tokens=512,
            )
        )
        assert sdk.client_kwargs["timeout"] == llm.LLM_TIMEOUT_SECONDS
        assert sdk.client_kwargs["max_retries"] == llm.LLM_MAX_RETRIES
        assert sdk.call_kwargs["max_tokens"] == 512
        assert sdk.call_kwargs["system"] == "s"
        assert sdk.call_kwargs["messages"] == [{"role": "user", "content": "u"}]

    def test_returns_first_block_text(self):
        sdk = FakeAnthropic(text="answer")
        out = asyncio.run(
            llm.anthropic_chat(
                sdk, system="s", user="u", model="m", api_key="k",
                temperature=0.0, max_tokens=16,
            )
        )
        assert out == "answer"


class TestOllama:
    def test_client_gets_an_explicit_timeout(self):
        """Verified against ollama 0.6.2: the default is Timeout(timeout=None),
        so omitting this does not fall back to a library default -- it hangs."""
        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="http://h",
            temperature=0.1, num_predict=64,
        )
        assert sdk.client_kwargs["timeout"] == llm.LLM_TIMEOUT_SECONDS
        assert sdk.client_kwargs["host"] == "http://h"

    def test_system_message_can_be_suppressed(self):
        """Reflection sends the prompt as a lone user turn."""
        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="h",
            temperature=0.3, num_predict=64, send_system=False,
        )
        assert sdk.call_kwargs["messages"] == [{"role": "user", "content": "u"}]

        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="h",
            temperature=0.3, num_predict=64,
        )
        assert sdk.call_kwargs["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]

    def test_json_format_is_opt_in(self):
        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="h",
            temperature=0.1, num_predict=64,
        )
        assert "format" not in sdk.call_kwargs

        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="h",
            temperature=0.1, num_predict=64, json_format=True,
        )
        assert sdk.call_kwargs["format"] == "json"

    def test_options_carry_temperature_and_num_predict(self):
        sdk = FakeOllama(response={"message": {"content": "x"}})
        llm.ollama_chat(
            sdk, system="s", user="u", model="m", base_url="h",
            temperature=0.3, num_predict=1024,
        )
        assert sdk.call_kwargs["options"] == {
            "temperature": 0.3,
            "num_predict": 1024,
        }


class TestOllamaText:
    def test_object_shape(self):
        class Message:
            content = "from object"

        class Response:
            message = Message()

        assert llm.ollama_text(Response()) == "from object"

    def test_dict_shape(self):
        assert llm.ollama_text({"message": {"content": "from dict"}}) == "from dict"

    def test_empty_content_is_empty_string_not_none(self):
        """"" and None mean different things: empty answer vs unusable shape."""
        assert llm.ollama_text({"message": {"content": ""}}) == ""

    def test_dict_without_message_is_empty_string(self):
        """Matches the pre-refactor behaviour exactly: the original treated any
        dict as recognised and read a missing "message" as an empty answer. It
        is arguably too lenient, but changing it would alter what extraction
        reports, so it is preserved and pinned here rather than quietly fixed."""
        assert llm.ollama_text({"nope": 1}) == ""

    def test_unrecognised_shape_is_none(self):
        assert llm.ollama_text(["unexpected"]) is None
        assert llm.ollama_text(None) is None
