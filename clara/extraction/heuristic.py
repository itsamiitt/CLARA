"""
CLARA — Rule-based fact extraction (zero-LLM tier).

A deterministic extractor producing the same :class:`ExtractedFact` shape as
the LLM-backed :class:`~clara.extraction.extractor.FactExtractor`, so the
update engine, classification keywords, and conflict resolution work
unchanged. Used when no LLM provider is configured (``llm_provider="none"``),
as the automatic fallback when the LLM is unavailable, and by the
``clara remember`` CLI.

Doctrine: precision over recall. In CLI-agent use the host model calling the
``memory_save`` MCP tool is the primary write channel; these patterns are the
safety net for plain-text ingestion. Extract less, never wrongly.
"""

from __future__ import annotations

import re
from typing import Callable, Iterator

from clara.extraction.extractor import ExtractedFact, ExtractionResult

# Mirrors rule 2 of the LLM extraction prompt: hedged statements are skipped.
_HEDGES = re.compile(
    r"\b(maybe|might|perhaps|probably|possibly|i think|i guess|not sure|"
    r"i believe|i suppose|kind of|sort of)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")

# Object captures: bounded, non-greedy, stopped by clause delimiters.
_OBJ = r"(?P<object>[^,.;:!?]{2,80}?)"
_OBJ2 = r"(?P<object2>[^,.;:!?]{2,80}?)"
_END = r"(?=$|,|;|:| for | because | since | when | so that | which )"

# Optional trailing domain: "... for web work" → domain="web work".
_DOMAIN = r"(?: for (?P<domain>[a-z0-9 _-]{2,40}))?"

_EVENT_VERBS = (
    "deployed|shipped|released|launched|migrated|finished|completed|started|"
    "began|fixed|merged|pushed|installed|upgraded|created|deleted|removed"
)

_SKILL_VERBS = "know|knows|learned|learnt|mastered"

# Demonstrative/expletive subjects that carry no entity — never a real fact
# subject. First-person ("I", "we") is intentionally NOT here: those are
# meaningful and normalized elsewhere.
_NON_SUBJECTS = frozenset({
    "this", "that", "these", "those", "it", "there",
    "which", "what", "here", "they", "he", "she",
})


class _Pattern:
    """One extraction rule: a compiled regex plus a fact builder."""

    def __init__(
        self,
        regex: str,
        build: Callable[[re.Match[str]], list[dict]],
        confidence: float,
    ) -> None:
        self.regex = re.compile(regex, re.IGNORECASE)
        self.build = build
        self.confidence = confidence


def _clean(value: str | None) -> str:
    return (value or "").strip().strip("\"'`").rstrip(" .")


def _negation_switch(m: re.Match[str]) -> list[dict]:
    return [
        {"subject": "user", "relation": "uses", "object": _clean(m["object"]),
         "is_negation": True},
        {"subject": "user", "relation": "uses", "object": _clean(m["object2"]),
         "is_negation": False},
    ]


def _negation_simple(m: re.Match[str]) -> list[dict]:
    return [{"subject": "user", "relation": "uses",
             "object": _clean(m["object"]), "is_negation": True}]


def _simple(relation: str, subject_from: str = "user") -> Callable[[re.Match[str]], list[dict]]:
    def build(m: re.Match[str]) -> list[dict]:
        groups = m.groupdict()
        subject = _clean(groups.get("subject")) if subject_from == "match" else "user"
        return [{
            "subject": subject or "user",
            "relation": relation,
            "object": _clean(m["object"]),
            "domain": _clean(groups.get("domain")) or None,
        }]
    return build


def _verb_fact(m: re.Match[str]) -> list[dict]:
    return [{
        "subject": "user",
        "relation": _clean(m["verb"]).lower(),
        "object": _clean(m["object"]),
        "domain": _clean(m.groupdict().get("domain")) or None,
    }]


def _world_fact(relation: str) -> Callable[[re.Match[str]], list[dict]]:
    def build(m: re.Match[str]) -> list[dict]:
        return [{
            "subject": _clean(m["subject"]),
            "relation": relation,
            "object": _clean(m["object"]),
        }]
    return build


_PATTERNS: list[_Pattern] = [
    # -- negation transitions first (most specific) ----------------------
    _Pattern(
        rf"\b(?:i|we)\s+switched\s+from\s+{_OBJ}\s+to\s+{_OBJ2}{_END}",
        _negation_switch, 0.9,
    ),
    _Pattern(
        rf"\b(?:i|we)\s+(?:no longer use|stopped using|quit using|quit|"
        rf"don'?t use|do not use)\s+{_OBJ}(?:\s+anymore)?{_END}",
        _negation_simple, 0.9,
    ),
    # -- preferences / usage ---------------------------------------------
    _Pattern(
        rf"\b(?:i|we)\s+(?:use|am using|are using)\s+{_OBJ}{_DOMAIN}{_END}",
        _simple("uses"), 0.85,
    ),
    _Pattern(
        rf"\b(?:i|we)\s+prefer\s+{_OBJ}{_DOMAIN}{_END}",
        _simple("prefers"), 0.85,
    ),
    _Pattern(
        rf"\bmy favorite\s+(?P<domain>[a-z0-9 _-]{{2,30}})\s+is\s+{_OBJ}{_END}",
        _simple("prefers"), 0.85,
    ),
    # -- skills ------------------------------------------------------------
    _Pattern(
        rf"\b(?:i|we)\s+(?:{_SKILL_VERBS})\s+(?:how to\s+)?{_OBJ}{_END}",
        _simple("learned"), 0.8,
    ),
    _Pattern(
        rf"\b(?:i am|i'm)\s+(?:proficient|experienced|skilled)\s+"
        rf"(?:in|with)\s+{_OBJ}{_END}",
        _simple("proficient_in"), 0.85,
    ),
    # -- events ------------------------------------------------------------
    _Pattern(
        rf"\b(?:i|we)\s+(?:just\s+|finally\s+)?(?P<verb>{_EVENT_VERBS})\s+{_OBJ}{_END}",
        _verb_fact, 0.8,
    ),
    # -- world model ---------------------------------------------------------
    _Pattern(
        rf"\b(?P<subject>[A-Za-z0-9 _./-]{{2,40}}?)\s+runs on\s+{_OBJ}{_END}",
        _world_fact("runs_on"), 0.75,
    ),
    _Pattern(
        rf"\b(?P<subject>[A-Za-z0-9 _./-]{{2,40}}?)\s+depends on\s+{_OBJ}{_END}",
        _world_fact("depends_on"), 0.75,
    ),
    _Pattern(
        rf"\b(?P<subject>[A-Za-z0-9 _./-]{{2,40}}?)\s+is\s+(?:a|an)\s+{_OBJ}{_END}",
        _world_fact("is_a"), 0.7,
    ),
]


class HeuristicExtractor:
    """Deterministic, dependency-free fact extraction.

    API-compatible with :class:`FactExtractor` (async ``extract`` returning
    an :class:`ExtractionResult`); never raises, never touches the network.
    """

    # Read by ClaraMemory for provider bookkeeping.
    _provider = "none"
    _model = None

    async def extract(self, text: str) -> ExtractionResult:
        return self.extract_sync(text)

    def extract_sync(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(status="empty")

        facts: list[ExtractedFact] = []
        seen: set[tuple] = set()
        for sentence in self._sentences(text):
            if _HEDGES.search(sentence):
                continue
            for pattern in _PATTERNS:
                match = pattern.regex.search(sentence)
                if not match:
                    continue
                for raw in pattern.build(match):
                    subject = raw.get("subject") or ""
                    relation = raw.get("relation") or ""
                    object_ = raw.get("object") or ""
                    if not subject or not relation or not object_:
                        continue
                    # Drop demonstrative/expletive subjects: "This is a problem"
                    # would otherwise yield world_model "This is_a problem",
                    # which contradicts the module's precision-first doctrine.
                    if subject.strip().lower() in _NON_SUBJECTS:
                        continue
                    key = (
                        subject.lower(), relation.lower(), object_.lower(),
                        bool(raw.get("is_negation")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(ExtractedFact(
                        subject=subject,
                        relation=relation,
                        object=object_,
                        domain=raw.get("domain"),
                        source_type="user_direct",
                        confidence=pattern.confidence,
                        is_negation=bool(raw.get("is_negation")),
                        raw_text=text,
                    ))
                # First matching pattern wins per sentence: patterns are
                # ordered most-specific first, and one sentence yielding two
                # overlapping interpretations is how wrong facts get stored.
                break

        return ExtractionResult(facts)

    @staticmethod
    def _sentences(text: str) -> Iterator[str]:
        for part in _SENTENCE_SPLIT.split(text):
            part = part.strip()
            if part:
                yield part
