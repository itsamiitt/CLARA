"""The CLI must stay cheap to import.

`clara statusline` runs on the Claude Code status bar's refreshInterval --
every 5 seconds by default. Importing clara.cli cost 7.75 s, so each render
took ~7.6 s and the bar could never keep up with itself; Claude Code kept
spawning a new process before the last had finished.

Almost none of that was the status line's own work. It only reads the stdlib
stats sidecar, but clara.cli imported LocalMemory, HeuristicExtractor and
friends at module scope, which pulled in lancedb (3.9 s, via pyarrow and a
generated urllib3 client) and the OpenAI SDK (1.8 s). Deferring those into the
commands that use them took the import to 0.36 s and a render to ~0.4 s.

A top-level `from clara.integrations.local_memory import ...` added back for
convenience would silently undo it, so this asserts the absence of the heavy
modules rather than a wall-clock number, which would be flaky on CI.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# lancedb and openai are optional extras; if they are not installed the
# absence assertions below pass trivially and prove nothing.
HEAVY = ("lancedb", "pyarrow", "openai")


def _probe(module: str, names: tuple[str, ...]) -> dict[str, bool]:
    """Import *module* in a fresh interpreter, report which *names* got loaded."""
    code = (
        "import sys, json;"
        f"__import__({module!r});"
        f"print(json.dumps({{n: n in sys.modules for n in {list(names)!r}}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("heavy", HEAVY)
def test_importing_the_cli_does_not_pull_in(heavy):
    """The status line needs none of these; it must not pay for them."""
    loaded = _probe("clara.cli", HEAVY)
    assert loaded[heavy] is False, (
        f"importing clara.cli now loads {heavy}, which puts the status-line "
        "render back over the 5s refresh interval. Import it inside the "
        "command that needs it instead."
    )


def test_the_statusline_path_stays_stdlib():
    """stats_cache is what the status line actually reads.

    It is deliberately stdlib-only so the hot path never opens SQLAlchemy.
    """
    loaded = _probe("clara.stats_cache", (*HEAVY, "sqlalchemy"))
    for name, was_loaded in loaded.items():
        assert was_loaded is False, f"clara.stats_cache pulled in {name}"


def test_vector_deps_are_loaded_on_demand_not_at_import():
    """clara.retrieval.engine holds the lancedb globals but must not import
    them until a vector operation actually runs."""
    loaded = _probe("clara.retrieval.engine", ("lancedb", "pyarrow"))
    assert loaded == {"lancedb": False, "pyarrow": False}
