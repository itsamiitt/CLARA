"""
CLARA — Fact Extraction Module

Converts raw natural-language text into structured ``ExtractedFact`` objects
by calling an LLM (OpenAI or Anthropic, selected via ``CLARA_LLM_PROVIDER``).

The LLM is prompted to return **only** valid JSON — no markdown, no
explanation.  The module enforces the CONTEXT.md extraction rules:

* Entities, relations, events, negations, and domain context are extracted.
* Hedged statements (``maybe``, ``might``, ``I think``) are ignored.
* Negations are flagged explicitly via ``is_negation=True``.
* Candidates with ``confidence < 0.4`` are discarded.
* JSON parse failures are handled gracefully (empty list + logged error).

All LLM calls are async to avoid blocking the event loop.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from clara.core import llm as _llm
from clara.core.ollama import ensure_model as _ensure_ollama_model_shared

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

ENV_LLM_PROVIDER = "CLARA_LLM_PROVIDER"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
ENV_OLLAMA_BASE_URL = "CLARA_OLLAMA_BASE_URL"

ENV_OPENAI_MODEL = "CLARA_OPENAI_MODEL"
ENV_ANTHROPIC_MODEL = "CLARA_ANTHROPIC_MODEL"
ENV_OLLAMA_MODEL = "CLARA_OLLAMA_MODEL"

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

CONFIDENCE_FLOOR = 0.4

# Network resilience for hosted LLM providers. The OpenAI/Anthropic SDKs apply
# their own exponential backoff up to max_retries and abort after timeout, so a
# transient blip doesn't silently lose the extraction.
# Re-exported from clara.core.llm, which owns them now: reasoning imports these
# names from here, and the shared client builders apply them.
LLM_TIMEOUT_SECONDS = _llm.LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES = _llm.LLM_MAX_RETRIES

# ---------------------------------------------------------------------------
# Optional dependency imports (guarded)
# ---------------------------------------------------------------------------

try:
    import openai as _openai  # type: ignore[import-untyped]
except ImportError:
    _openai = None  # type: ignore[assignment]

try:
    import anthropic as _anthropic  # type: ignore[import-untyped]
except ImportError:
    _anthropic = None  # type: ignore[assignment]

try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """A single structured fact extracted from raw text.

    Attributes:
        subject:     The entity the fact is about (e.g. ``"user"``).
        relation:    The relation or verb (e.g. ``"uses"``, ``"started"``).
        object:      The target of the relation (e.g. ``"Rust"``).
        domain:      Optional domain context (e.g. ``"systems"``).
        source_type: Evidence classification (``"user_direct"``, etc.).
        confidence:  Extraction confidence in [0.0, 1.0].
        is_negation: ``True`` if this fact negates or invalidates a prior fact.
        raw_text:    The original text from which this fact was extracted.
    """

    subject: str
    relation: str
    object: str
    domain: str | None
    source_type: str
    confidence: float
    is_negation: bool
    raw_text: str


class ExtractionResult(list):
    """A list of :class:`ExtractedFact` carrying an extraction status.

    Subclasses ``list`` so every existing caller and test that treats the
    return value of :meth:`FactExtractor.extract` as a plain list keeps
    working, while callers that care can distinguish "the text contained
    no facts" from "the extraction pipeline failed":

    * ``ok``                 — extraction ran; facts (possibly zero) parsed.
    * ``empty``              — input text was empty/whitespace.
    * ``llm_unavailable``    — provider missing key/package/connection.
    * ``llm_error``          — provider call failed at runtime.
    * ``malformed_response`` — LLM output could not be parsed as facts.
    """

    __slots__ = ("status", "detail")

    def __init__(
        self,
        facts: list[ExtractedFact] | None = None,
        *,
        status: str = "ok",
        detail: str | None = None,
    ) -> None:
        super().__init__(facts or [])
        self.status = status
        self.detail = detail

    @property
    def failed(self) -> bool:
        """True when the pipeline failed (as opposed to finding no facts)."""
        return self.status in {"llm_unavailable", "llm_error", "malformed_response"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a fact extraction engine for a memory system. Your task is to extract \
structured facts from the user's text.

RULES:
1. Extract entities (people, projects, technologies), relations (uses, builds, \
prefers, owns, manages), events (started, completed, failed, updated), \
preferences, and constraints.
2. IGNORE hedged or uncertain statements containing words like "maybe", \
"might", "I think", "perhaps", "probably", "not sure". Do NOT extract these.
3. FLAG negations explicitly. If the text says "no longer", "switched from", \
"stopped", "quit", "don't use", set is_negation to true for the OLD fact \
being negated.
4. When both a negation and a new fact exist (e.g. "switched from X to Y"), \
produce TWO facts: one negation for X and one positive for Y.
5. Tag domain context when available (e.g. "for web work" → domain: "web", \
"at my day job" → domain: "work").
6. Assign a confidence score between 0.0 and 1.0 based on how explicit and \
unambiguous the statement is. Direct, clear statements get 0.8-1.0. \
Implicit or contextual inferences get 0.5-0.7.
7. Set source_type to one of: "user_direct", "user_indirect", "tool_api", \
"system", "agent_inference".

OUTPUT FORMAT:
Return ONLY a valid JSON object with a top-level "facts" array.
No markdown, no code fences, no explanation.
Each element inside "facts" must have these exact keys:
- "subject" (string)
- "relation" (string)
- "object" (string)
- "domain" (string or null)
- "source_type" (string)
- "confidence" (float)
- "is_negation" (boolean)

Example output:
{"facts": [{"subject": "user", "relation": "uses", "object": "Rust", \
"domain": "systems", "source_type": "user_direct", "confidence": 0.9, \
"is_negation": false}]}

If no facts can be extracted, return: {"facts": []}
"""


# ---------------------------------------------------------------------------
# LLM client wrappers
# ---------------------------------------------------------------------------

async def _call_openai(text: str, model: str) -> str:
    """Call OpenAI chat completion asynchronously and return the raw response content."""
    _llm.require_sdk(_openai, "openai", "OpenAI LLM provider")
    api_key = _llm.require_api_key(ENV_OPENAI_KEY, os.environ.get)
    # Returns the raw content (even if empty). Coercing an empty completion to
    # "[]" used to report status="ok" with zero facts, hiding a failed call as
    # a successful no-fact extraction — let the parser flag it malformed.
    return await _llm.openai_chat(
        _openai,
        system=SYSTEM_PROMPT,
        user=text,
        model=model,
        api_key=api_key,
        temperature=0.0,
        json_mode=True,
    )


async def _call_anthropic(text: str, model: str) -> str:
    """Call Anthropic messages API asynchronously and return the raw response content."""
    _llm.require_sdk(_anthropic, "anthropic", "Anthropic LLM provider")
    api_key = _llm.require_api_key(ENV_ANTHROPIC_KEY, os.environ.get)
    # Anthropic returns content as a list of content blocks. Returned raw (even
    # if empty) so an empty completion is flagged malformed rather than masked
    # as a successful zero-fact extraction.
    return await _llm.anthropic_chat(
        _anthropic,
        system=SYSTEM_PROMPT,
        user=text,
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )


def _ensure_ollama_model(base_url: str, model: str) -> None:
    """Ensure an Ollama model is available — delegates to shared module."""
    _ensure_ollama_model_shared(base_url, model)


def _call_ollama(text: str, model: str, base_url: str) -> str:
    _llm.require_sdk(
        _ollama_lib, "ollama", "Ollama extraction provider",
        install="'clara-memory[ollama]'",
    )
    _ensure_ollama_model(base_url, model)
    response = _llm.ollama_chat(
        _ollama_lib,
        system=SYSTEM_PROMPT,
        user=f"Input:\n{text}\n\nRespond with valid JSON only.",
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_predict=2048,
        json_format=True,
    )
    content = _llm.ollama_text(response)
    if content is None:
        logger.warning(
            "Unexpected Ollama chat response shape (%s); extracting no facts.",
            type(response).__name__,
        )
        return ""
    return content


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, original_text: str) -> ExtractionResult:
    """Parse the LLM's JSON response into ExtractedFact objects.

    Handles:
    * Bare JSON arrays ``[...]``
    * JSON objects wrapping an array ``{"facts": [...]}``
    * Markdown-fenced JSON (strips ` ```json ... ``` `)

    Returns an empty ``ExtractionResult`` with ``status="malformed_response"``
    (and logs a warning) on any parse failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if the LLM disobeyed instructions
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first line (```json) and last line (```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Failed to parse LLM response as JSON: %s — raw=%r",
            exc, raw[:200],
        )
        return ExtractionResult(status="malformed_response", detail=str(exc))

    # Handle {"facts": [...]} wrapper
    if isinstance(parsed, dict):
        for key in ("facts", "results", "extracted", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            logger.warning(
                "LLM returned a JSON object without a recognized list key: %r",
                list(parsed.keys()),
            )
            return ExtractionResult(
                status="malformed_response",
                detail=f"no list key in {sorted(parsed.keys())!r}",
            )

    if not isinstance(parsed, list):
        logger.warning("LLM returned non-list JSON: %s", type(parsed).__name__)
        return ExtractionResult(
            status="malformed_response",
            detail=f"non-list JSON: {type(parsed).__name__}",
        )

    facts: list[ExtractedFact] = []
    for item in parsed:
        if not isinstance(item, dict):
            # e.g. {"facts": ["user uses Rust"]} — used to raise
            # AttributeError on item.get and crash remember()/interact().
            logger.warning("Skipping non-object fact item: %r", item)
            continue
        domain = item.get("domain")
        if domain is not None and not isinstance(domain, str):
            # Non-string domains used to leak into conflict resolution and
            # crash _domains_differ with an AttributeError.
            domain = str(domain)
        try:
            fact = ExtractedFact(
                subject=str(item.get("subject", "")),
                relation=str(item.get("relation", "")),
                object=str(item.get("object", "")),
                domain=domain,
                source_type=str(item.get("source_type", "user_direct")),
                confidence=float(item.get("confidence", 0.0)),
                is_negation=bool(item.get("is_negation", False)),
                raw_text=original_text,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed fact item: %s — %r", exc, item)
            continue

        # Discard low-confidence candidates (CONTEXT.md rule)
        if fact.confidence < CONFIDENCE_FLOOR:
            logger.debug(
                "Discarding low-confidence fact (%.2f < %.2f): %s → %s → %s",
                fact.confidence, CONFIDENCE_FLOOR,
                fact.subject, fact.relation, fact.object,
            )
            continue

        # Discard facts with empty required fields
        if not fact.subject or not fact.relation or not fact.object:
            logger.debug("Discarding fact with empty required fields: %r", item)
            continue

        facts.append(fact)

    return ExtractionResult(facts)


# ---------------------------------------------------------------------------
# FactExtractor
# ---------------------------------------------------------------------------

class FactExtractor:
    """Extracts structured facts from raw text via an LLM.

    Usage::

        extractor = FactExtractor()
        facts = await extractor.extract("I switched from Python to Rust for systems work.")
        for f in facts:
            print(f.subject, f.relation, f.object, f.is_negation)

    The LLM provider is selected via:
    * ``CLARA_LLM_PROVIDER`` env var: ``"openai"`` (default) or ``"anthropic"``
    * ``CLARA_OPENAI_MODEL`` / ``CLARA_ANTHROPIC_MODEL`` override the model name.

    All extraction calls are async to avoid blocking the event loop.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._provider = (
            provider or os.environ.get(ENV_LLM_PROVIDER, DEFAULT_PROVIDER)
        ).strip().lower()
        self._ollama_base_url = (
            ollama_base_url
            or os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
        )

        if self._provider == "openai":
            self._model = model or os.environ.get(
                ENV_OPENAI_MODEL, DEFAULT_OPENAI_MODEL
            )
        elif self._provider == "anthropic":
            self._model = model or os.environ.get(
                ENV_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_MODEL
            )
        elif self._provider == "ollama":
            if _ollama_lib is None:
                raise ImportError(
                    "The 'ollama' package is required for the Ollama extraction provider. "
                    "Install it with: pip install 'clara-memory[ollama]'"
                )
            self._model = model or os.environ.get(
                ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL
            )
        else:
            raise ValueError(
                f"Unknown LLM provider {self._provider!r}. "
                f"Supported: 'openai', 'anthropic', 'ollama'."
            )

    async def extract(self, text: str) -> ExtractionResult:
        """Extract structured facts from *text* asynchronously.

        Args:
            text: Raw natural-language input.

        Returns:
            An :class:`ExtractionResult` (a ``list`` of
            :class:`ExtractedFact` plus a ``status`` attribute). The list is
            empty when the text is empty, the LLM call fails, the output is
            unparseable, or no facts meet the confidence threshold — the
            ``status`` field distinguishes those cases so callers can
            surface pipeline failures instead of silently storing nothing.
        """
        if not text or not text.strip():
            return ExtractionResult(status="empty")

        try:
            if self._provider == "openai":
                raw_response = await _call_openai(text, self._model)
            elif self._provider == "anthropic":
                raw_response = await _call_anthropic(text, self._model)
            elif self._provider == "ollama":
                # Ollama client is synchronous — run in executor
                import asyncio
                loop = asyncio.get_running_loop()
                raw_response = await loop.run_in_executor(
                    None, _call_ollama, text, self._model, self._ollama_base_url,
                )
            else:
                return ExtractionResult(
                    status="llm_unavailable",
                    detail=f"unknown provider {self._provider!r}",
                )
        except (OSError, ImportError) as exc:
            logger.error("LLM provider unavailable: %s", exc)
            return ExtractionResult(status="llm_unavailable", detail=str(exc))
        except Exception as exc:
            logger.exception("LLM call failed for text: %r", text[:200])
            return ExtractionResult(status="llm_error", detail=str(exc))

        return _parse_llm_response(raw_response, text)
