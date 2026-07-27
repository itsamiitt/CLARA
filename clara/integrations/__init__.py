"""CLARA integrations: the zero-key LocalMemory facade and the OpenClaw bridge.

The bridge is resolved on attribute access. It imports clara.agent -- the full
LLM tier -- which pulls the extractor, the reasoning engine and the OpenAI SDK
(2.2 s). Because that import lived here, *any* ``clara.integrations.*`` import
paid for it, including ``from clara.integrations.local_memory import
LocalMemory``, which is the zero-key path that uses none of it.

``from clara.integrations import OpenClawMemoryBridge`` keeps working; the
import happens at that moment instead of at package import.
"""

from typing import TYPE_CHECKING, Any

from .local_memory import LocalMemory

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    from .openclaw_bridge import BridgeConfig, OpenClawMemoryBridge

_LAZY = frozenset({"BridgeConfig", "OpenClawMemoryBridge"})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import openclaw_bridge

        return getattr(openclaw_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["BridgeConfig", "LocalMemory", "OpenClawMemoryBridge"]
