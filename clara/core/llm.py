"""
Shared chat-completion plumbing for CLARA's three LLM callers.

Extraction, reasoning and reflection each need "send a system + user message to
the configured provider and give me back the text". Each grew its own copy, and
the copies drifted in ways that were invisible until measured:

  * reflection built its OpenAI and Anthropic clients with no ``timeout`` and no
    ``max_retries`` while the other two passed 30 s / 2 retries. Reflection runs
    from the daily scheduler, and nothing in CLARA wraps an LLM call in
    ``asyncio.wait_for``, so a stalled endpoint hung that job indefinitely.
  * every Ollama client was built as ``Client(host=...)``. Verified against
    ollama 0.6.2: the underlying ``httpx.Client`` then carries
    ``Timeout(timeout=None)`` -- no timeout whatsoever, on any of the four call
    sites. Passing ``timeout=`` is accepted and propagates.

The helpers here take the SDK module as their first argument rather than
importing it themselves. That is deliberate: each caller keeps its own guarded
module-level ``_openai`` / ``_anthropic`` / ``_ollama_lib`` handle, so existing
patch points such as ``clara.extraction.extractor._openai`` keep working and a
missing optional dependency is still reported against the subsystem that needed
it.

What is intentionally NOT unified is failure policy. Extraction and reasoning
raise; reflection degrades to template text because it is a background job whose
output is stored. These helpers raise, and each caller keeps its own handler.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Applied to every provider client. Ollama in particular defaults to no timeout
# at all, so omitting this is not "use the library default" -- it is "hang".
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2


def require_sdk(
    sdk: Any, package: str, purpose: str, *, install: str | None = None
) -> Any:
    """Return *sdk*, or raise ImportError naming the package and what needed it.

    *install* overrides the pip target when the useful thing to install is an
    extra rather than the package itself (``clara-memory[ollama]``).
    """
    if sdk is None:
        raise ImportError(
            f"The {package!r} package is required for the {purpose}. "
            f"Install it with: pip install {install or package!r}"
        )
    return sdk


def require_api_key(env_var: str, get_env: Any) -> str:
    """Return the key in *env_var*, or raise OSError naming the variable."""
    api_key = get_env(env_var)
    if not api_key:
        raise OSError(f"Environment variable {env_var!r} is not set.")
    return str(api_key)


async def openai_chat(
    sdk: Any,
    *,
    system: str,
    user: str,
    model: str,
    api_key: str,
    temperature: float,
    json_mode: bool = False,
    timeout: float = LLM_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
) -> str:
    """One OpenAI chat completion; returns the raw content, possibly empty.

    Empty is returned as-is rather than coerced to a placeholder: a caller that
    treats "" as malformed can distinguish a failed call from a genuinely empty
    result, which coercion would hide.
    """
    client = sdk.AsyncOpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def anthropic_chat(
    sdk: Any,
    *,
    system: str,
    user: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float = LLM_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
) -> str:
    """One Anthropic message; returns the first content block's text."""
    client = sdk.AsyncAnthropic(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return response.content[0].text or ""


def ollama_text(response: Any) -> str | None:
    """Text out of an Ollama chat response, or None if the shape is unrecognised.

    None rather than "" so callers can keep their differing policies: extraction
    logs and extracts nothing, reasoning returns empty, reflection falls back to
    template text. Collapsing the two cases would erase that distinction.
    """
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    if isinstance(response, dict):
        payload = response.get("message", {})
        if isinstance(payload, dict):
            return str(payload.get("content", "") or "")
    return None


def ollama_chat(
    sdk: Any,
    *,
    system: str,
    user: str,
    model: str,
    base_url: str,
    temperature: float,
    num_predict: int,
    json_format: bool = False,
    send_system: bool = True,
    timeout: float = LLM_TIMEOUT_SECONDS,
) -> Any:
    """One Ollama chat call. Synchronous -- callers run it in an executor.

    Returns the raw response; pass it through :func:`ollama_text`. Kept separate
    so a caller can inspect an unrecognised response before deciding what to do.
    """
    messages: list[dict[str, str]] = []
    if send_system and system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    client = sdk.Client(host=base_url, timeout=timeout)
    kwargs: dict[str, Any] = {}
    if json_format:
        kwargs["format"] = "json"
    return client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_predict": num_predict},
        **kwargs,
    )
