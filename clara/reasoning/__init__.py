"""Reasoning and context-assembly helpers for CLARA.

``ReasoningEngine`` is resolved on attribute access rather than imported here.
Importing it pulls clara.extraction.extractor and, through it, the OpenAI SDK
(2.2 s). That cost applied to *any* import of this package -- including
``from clara.reasoning.context import format_context``, which LocalMemory needs
and which uses no LLM at all -- so the zero-key tier paid for the reasoning
stack on every `clara` command that opened the store.

``from clara.reasoning import ReasoningEngine`` keeps working; the import just
happens at that moment instead of at package import.
"""

from typing import TYPE_CHECKING, Any

from clara.reasoning.context import ContextAssembler, format_context

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    from clara.reasoning.engine import ReasoningEngine, ReasoningResponse

_LAZY = frozenset({"ReasoningEngine", "ReasoningResponse"})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from clara.reasoning import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ContextAssembler",
    "ReasoningEngine",
    "ReasoningResponse",
    "format_context",
]
