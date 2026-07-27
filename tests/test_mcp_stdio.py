"""End-to-end MCP wire tests: spawn `clara-mcp` and speak real stdio JSON-RPC.

Everything else exercises the server in-process, which cannot catch the failure
mode that actually takes CLARA down in the field: anything printed to **stdout**
corrupts the JSON-RPC stream, and the memory server simply stops working. A
stray `print()` — in CLARA or in a dependency — passes every in-process test.

These spawn the real module the plugin's shim runs, so they also cover the
handshake, the advertised tool surface, and a genuine save/search round-trip.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

from tests.test_mcp_server import EXPECTED_TOOLS

# The whole module needs the MCP SDK and a subprocess; skip cleanly without it.
pytest.importorskip("mcp")

# Generous: a cold interpreter plus schema creation on a slow shared runner.
_WIRE_TIMEOUT_S = 90.0


def _server_env(tmp_path) -> dict[str, str]:
    """Isolate the store, the home directory and every add-on side effect."""
    env = dict(os.environ)
    env.update(
        {
            "CLARA_DB_PATH": str(tmp_path / "clara.db"),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            # Keep the run hermetic: no native-memory writes, no doc scanning.
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
    )
    env.pop("CLARA_HOME", None)
    return env


async def _with_session(tmp_path, body):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "clara.integrations.mcp_server"],
        env=_server_env(tmp_path),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await body(session)


def _run(tmp_path, body):
    return asyncio.run(asyncio.wait_for(_with_session(tmp_path, body), _WIRE_TIMEOUT_S))


def _payload(result) -> dict:
    """Decode a tool result's JSON body."""
    return json.loads(result.content[0].text)


class TestHandshake:
    def test_server_initializes_over_stdio(self, tmp_path):
        async def body(session):
            # Reaching here at all means the handshake framing was clean.
            return await session.list_tools()

        tools = _run(tmp_path, body)
        assert tools.tools, "server advertised no tools"

    def test_advertised_tools_match_the_declared_surface(self, tmp_path):
        async def body(session):
            return await session.list_tools()

        names = {tool.name for tool in _run(tmp_path, body).tools}
        assert names == EXPECTED_TOOLS, (
            f"wire surface differs: +{names - EXPECTED_TOOLS} "
            f"-{EXPECTED_TOOLS - names}"
        )

    def test_every_tool_advertises_a_description(self, tmp_path):
        # The description is how the model decides to call a tool at all.
        async def body(session):
            return await session.list_tools()

        for tool in _run(tmp_path, body).tools:
            assert (tool.description or "").strip(), f"{tool.name} has no description"


class TestRoundTrip:
    def test_save_then_search_over_the_wire(self, tmp_path):
        async def body(session):
            saved = await session.call_tool(
                "memory_save",
                {
                    "mem_type": "belief",
                    "subject": "user",
                    "relation": "uses",
                    "object": "fly.io",
                },
            )
            found = await session.call_tool("memory_search", {"query": "fly.io"})
            return _payload(saved), _payload(found)

        saved, found = _run(tmp_path, body)
        assert saved["action"] == "saved"
        assert found["total"] == 1
        assert "fly.io" in found["context"]

    def test_logging_tools_do_not_corrupt_the_stream(self, tmp_path):
        # memory_save logs ("Saved <type> memory <id>") and maintenance logs
        # more besides. If any of that reached stdout the transport would fail
        # long before these assertions; a later call proves the stream survived.
        async def body(session):
            await session.call_tool(
                "memory_save",
                {
                    "mem_type": "belief",
                    "subject": "user",
                    "relation": "prefers",
                    "object": "sqlite",
                },
            )
            return _payload(await session.call_tool("memory_stats", {}))

        stats = _run(tmp_path, body)
        assert stats["active_by_type"]["belief"] == 1

    def test_project_profile_reads_the_repo_over_the_wire(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "wire-demo", "dependencies": {"react": "^18"}}),
            encoding="utf-8",
        )

        async def body(session):
            return _payload(
                await session.call_tool("project_profile", {"repo": str(tmp_path)})
            )

        profile = _run(tmp_path, body)
        assert profile["name"] == "wire-demo"
        assert "react" in profile["detected"]["framework"]


class TestNoInheritedStdin:
    """Regression: every tool call used to cost 5 s over stdio.

    The server holds stdin as the MCP transport pipe, and ``subprocess.run``
    pipes stdout/stderr but *inherits* stdin. `git rev-parse`, which
    ``resolve_store`` runs on every call, therefore blocked on a pipe that
    never closes and was killed by its 5 s timeout — on every single request.
    In-process tests could not see this, because there is no transport pipe.
    """

    def test_tool_calls_do_not_hit_the_git_timeout(self, tmp_path):
        from clara.store import _GIT_TIMEOUT_S

        async def body(session):
            # First call warms the store; measure the ones after it.
            await session.call_tool("memory_stats", {})
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                await session.call_tool("memory_stats", {})
                timings.append(time.perf_counter() - started)
            return timings

        timings = _run(tmp_path, body)
        worst = max(timings)
        # Half the timeout is a wide margin: observed ~0.05 s after the fix,
        # ~5.0 s before it. Anything near the timeout means stdin leaked again.
        assert worst < _GIT_TIMEOUT_S / 2, (
            f"tool call took {worst:.2f}s, close to the {_GIT_TIMEOUT_S}s git "
            "timeout — a subprocess is probably inheriting the transport stdin"
        )


class TestErrorsStayOnTheProtocol:
    def test_invalid_arguments_return_an_error_not_a_crash(self, tmp_path):
        # A bad call must come back as a tool error over JSON-RPC; a traceback
        # printed to stdout would break every subsequent request.
        async def body(session):
            result = await session.call_tool(
                "memory_save", {"mem_type": "belief", "subject": "only-a-subject"}
            )
            follow_up = await session.call_tool("memory_stats", {})
            return result, _payload(follow_up)

        result, stats = _run(tmp_path, body)
        assert result.isError, "expected a protocol-level tool error"
        # The session is still usable afterwards.
        assert stats["total_rows"] == 0
