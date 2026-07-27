"""
Extraction result types.

Plain dataclasses, deliberately in their own module. They used to live in
clara.extraction.extractor, which imports the OpenAI SDK (2.2 s) -- so
clara.update.engine importing ``ExtractedFact``, a dataclass with eight
fields, pulled that whole stack onto the store path of every `clara` command.

clara.extraction.extractor re-exports both names, so the established
``from clara.extraction.extractor import ExtractedFact`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass


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
