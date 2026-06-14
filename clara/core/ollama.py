"""Shared Ollama model readiness check.

All CLARA modules that use Ollama (extraction, reasoning, reflection) import
``ensure_model()`` from here instead of maintaining their own per-module
``_OLLAMA_MODELS_READY`` sets.  This avoids redundant ``client.pull()`` calls
when multiple subsystems start concurrently.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import ollama as _ollama_lib  # type: ignore[import-untyped]
except ImportError:
    _ollama_lib: Any = None  # type: ignore[assignment]

_lock = threading.Lock()
_models_ready: set[tuple[str, str]] = set()


def ensure_model(base_url: str, model: str) -> None:
    """Pull *model* on the Ollama instance at *base_url* if not already present.

    Thread-safe: only one pull per ``(base_url, model)`` pair will ever execute,
    regardless of how many modules call this concurrently.

    Raises:
        ImportError: If the ``ollama`` package is not installed.
    """
    if _ollama_lib is None:
        raise ImportError(
            "The 'ollama' package is required for the Ollama provider. "
            "Install it with: pip install 'clara-memory[ollama]'"
        )

    key = (base_url, model)
    if key in _models_ready:
        return

    with _lock:
        # Double-checked locking
        if key in _models_ready:
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
            logger.info("Pulling Ollama model %r on %s …", model, base_url)
            client.pull(model)
        _models_ready.add(key)
