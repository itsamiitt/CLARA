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
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

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
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Optional dependency imports (guarded)
# ---------------------------------------------------------------------------

try:
    import openai as _openai  # type: ignore[import-untyped]
except ImportError:
    _openai: Any = None  # type: ignore[assignment]

try:
    import anthropic as _anthropic  # type: ignore[import-untyped]
except ImportError:
    _anthropic: Any = None  # type: ignore[assignment]

try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib: Any = None  # type: ignore[assignment]

_OLLAMA_MODELS_READY: set[tuple[str, str]] = set()


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

def _call_openai(text: str, model: str) -> str:
    """Call OpenAI chat completion and return the raw response content."""
    if _openai is None:
        raise ImportError(
            "The 'openai' package is required for the OpenAI LLM provider. "
            "Install it with:  pip install openai"
        )
    api_key = os.environ.get(ENV_OPENAI_KEY)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {ENV_OPENAI_KEY!r} is not set."
        )

    client = _openai.OpenAI(
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "[]"


def _call_anthropic(text: str, model: str) -> str:
    """Call Anthropic messages API and return the raw response content."""
    if _anthropic is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic LLM provider. "
            "Install it with:  pip install anthropic"
        )
    api_key = os.environ.get(ENV_ANTHROPIC_KEY)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {ENV_ANTHROPIC_KEY!r} is not set."
        )

    client = _anthropic.Anthropic(
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        temperature=0.0,
    )
    # Anthropic returns content as a list of content blocks
    return response.content[0].text or "[]"


def _ensure_ollama_model(base_url: str, model: str) -> None:
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama extraction provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )

    key = (base_url, model)
    if key in _OLLAMA_MODELS_READY:
        return

    client = _ollama_lib.Client(host=base_url)
    listing = client.list()
    models = listing.get("models", []) if isinstance(listing, dict) else getattr(
        listing, "models", []
    )
    names = {
        str(item.get("name") or item.get("model") or "")
        for item in models
        if isinstance(item, dict)
    }
    if model not in names:
        client.pull(model)
    _OLLAMA_MODELS_READY.add(key)


def _call_ollama(text: str, model: str, base_url: str) -> str:
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama extraction provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )

    _ensure_ollama_model(base_url, model)
    client = _ollama_lib.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Input:\n{text}\n\nRespond with valid JSON only.",
            },
        ],
        options={"temperature": 0.1, "num_predict": 2048},
        format="json",
    )
    message = getattr(response, "message", None)
    if message is not None:
        return getattr(message, "content", "") or ""
    if isinstance(response, dict):
        payload = response.get("message", {})
        if isinstance(payload, dict):
            return str(payload.get("content", "") or "")
    logger.warning(
        "Unexpected Ollama chat response shape (%s); extracting no facts.",
        type(response).__name__,
    )
    return ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, original_text: str) -> list[ExtractedFact]:
    """Parse the LLM's JSON response into ExtractedFact objects.

    Handles:
    * Bare JSON arrays ``[...]``
    * JSON objects wrapping an array ``{"facts": [...]}``
    * Markdown-fenced JSON (strips ` ```json ... ``` `)

    Returns an empty list and logs a warning on any parse failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if the LLM disobeyed instructions
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Failed to parse LLM response as JSON: %s — raw=%r",
            exc, raw[:200],
        )
        return []

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
            return []

    if not isinstance(parsed, list):
        logger.warning("LLM returned non-list JSON: %s", type(parsed).__name__)
        return []

    facts: list[ExtractedFact] = []
    for item in parsed:
        try:
            fact = ExtractedFact(
                subject=str(item.get("subject", "")),
                relation=str(item.get("relation", "")),
                object=str(item.get("object", "")),
                domain=item.get("domain"),
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

    return facts


# ---------------------------------------------------------------------------
# FactExtractor
# ---------------------------------------------------------------------------

class FactExtractor:
    """Extracts structured facts from raw text via an LLM.

    Usage::

        extractor = FactExtractor()
        facts = extractor.extract("I switched from Python to Rust for systems work.")
        for f in facts:
            print(f.subject, f.relation, f.object, f.is_negation)

    The LLM provider is selected via:
    * ``CLARA_LLM_PROVIDER`` env var: ``"openai"`` (default) or ``"anthropic"``
    * ``CLARA_OPENAI_MODEL`` / ``CLARA_ANTHROPIC_MODEL`` override the model name.
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
            self._call_fn = _call_openai
        elif self._provider == "anthropic":
            self._model = model or os.environ.get(
                ENV_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_MODEL
            )
            self._call_fn = _call_anthropic
        elif self._provider == "ollama":
            if _ollama_lib is None:
                raise ImportError(
                    "The 'ollama' package is required for the Ollama extraction provider. "
                    "Install it with: pip install 'clara-memory[ollama]'"
                )
            self._model = model or os.environ.get(
                ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL
            )
            self._call_fn = lambda text, selected_model: _call_ollama(
                text,
                selected_model,
                self._ollama_base_url,
            )
        else:
            raise ValueError(
                f"Unknown LLM provider {self._provider!r}. "
                f"Supported: 'openai', 'anthropic', 'ollama'."
            )

    def extract(self, text: str) -> list[ExtractedFact]:
        """Extract structured facts from *text*.

        Args:
            text: Raw natural-language input.

        Returns:
            A list of :class:`ExtractedFact` objects.  Returns an empty list
            if the text is empty, the LLM returns unparseable output, or no
            facts meet the confidence threshold.
        """
        if not text or not text.strip():
            return []

        try:
            raw_response = self._call_fn(text, self._model)
        except Exception:
            logger.exception("LLM call failed for text: %r", text[:200])
            return []

        return _parse_llm_response(raw_response, text)
