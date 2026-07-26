"""
CLARA — add-on feature flags (kill switches).

``CLARA_GRAPH_ENABLED`` and ``CLARA_DOCS_ENABLED`` (default on) disable the
knowledge-graph and doc-curator add-ons independently; the memory core keeps
working with both off. Mirrors ``ClaraConfig``'s ``_bool`` env semantics but
re-reads the environment on every call — long-lived processes (the MCP
server) and monkeypatching tests must see flag changes without a restart.
Stdlib-only so the fastpath may import it.
"""

from __future__ import annotations

import os

# The single definition of "off" for every CLARA boolean env var. ``config.py``
# imports this so a value like "enabled" cannot mean disabled in one process
# and enabled in another (the hook scripts test the same four strings).
FALSY = {"0", "false", "no", "off"}
_FALSY = FALSY

GRAPH_DISABLED_HINT = (
    "the knowledge-graph add-on is disabled (CLARA_GRAPH_ENABLED=0); "
    "unset it or set CLARA_GRAPH_ENABLED=1 to re-enable"
)
DOCS_DISABLED_HINT = (
    "the doc-curator add-on is disabled (CLARA_DOCS_ENABLED=0); "
    "unset it or set CLARA_DOCS_ENABLED=1 to re-enable"
)


def _flag(key: str) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return True
    return raw not in _FALSY


def graph_enabled() -> bool:
    """Is the knowledge-graph add-on enabled? (CLARA_GRAPH_ENABLED, default 1)"""
    return _flag("CLARA_GRAPH_ENABLED")


def docs_enabled() -> bool:
    """Is the doc-curator add-on enabled? (CLARA_DOCS_ENABLED, default 1)"""
    return _flag("CLARA_DOCS_ENABLED")
