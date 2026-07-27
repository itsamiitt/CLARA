"""The README's documented defaults must match the code.

The README states concrete numbers -- top-k 8, cap 16384, keep 7 -- and users
tune against them. Nothing stopped a default moving in clara/config.py while
the README kept quoting the old one, and a config doc that is subtly wrong is
worse than none: it is believed.

This is the same class of defect as the two found by testing the docs rather
than reading them: the README told users to run `clara doctor`, which a plugin
install never puts on PATH, and quoted a 30-60 s install that measured 113 s.
Those needed a human to notice. These do not.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]


def _readme_env_defaults() -> dict[str, str]:
    """Every ``CLARA_FOO=value`` shown in a README shell block."""
    text = (_ROOT / "README.md").read_text("utf-8")
    found: dict[str, str] = {}
    for name, value in re.findall(r"^(CLARA_[A-Z_]+)=(\S+)", text, re.MULTILINE):
        found.setdefault(name, value)
    return found


@pytest.fixture
def clean_env(monkeypatch):
    """Defaults are only observable with every CLARA_* override removed."""
    for key in list(os.environ):
        if key.startswith("CLARA_"):
            monkeypatch.delenv(key, raising=False)


class TestDocumentedDefaultsMatchTheCode:
    def test_readme_documents_the_tuning_vars(self):
        """Guard the guard: if the README stops showing these, the checks below
        would pass vacuously."""
        documented = _readme_env_defaults()
        for name in (
            "CLARA_RETRIEVAL_TOP_K",
            "CLARA_SIMILARITY_THRESHOLD",
            "CLARA_ARCHIVAL_THRESHOLD",
            "CLARA_EVENT_STALE_DAYS",
            "CLARA_SKILL_UNUSED_DAYS",
        ):
            assert name in documented, f"{name} is no longer documented"

    @pytest.mark.parametrize(
        ("env_name", "attr"),
        [
            ("CLARA_RETRIEVAL_TOP_K", "retrieval_top_k"),
            ("CLARA_SIMILARITY_THRESHOLD", "similarity_threshold"),
            ("CLARA_ARCHIVAL_THRESHOLD", "archival_threshold"),
            ("CLARA_EVENT_STALE_DAYS", "event_stale_days"),
            ("CLARA_SKILL_UNUSED_DAYS", "skill_unused_days"),
        ],
    )
    def test_tuning_default(self, env_name, attr, clean_env):
        from clara.config import ClaraConfig

        documented = _readme_env_defaults()[env_name]
        actual = getattr(ClaraConfig.from_env(), attr)
        assert float(documented) == pytest.approx(float(actual)), (
            f"README says {env_name}={documented}, code default is {actual}"
        )

    def test_max_content_bytes_default(self):
        """Read in a clean subprocess.

        The constant is computed from the environment at import, so observing
        its default needs a fresh interpreter. importlib.reload() would do it
        too, but rebinding a module the rest of the suite has already imported
        (and patches) is a poor trade for one assertion.
        """
        import json
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if not k.startswith("CLARA_")}
        result = subprocess.run(
            [sys.executable, "-c",
             "import json;from clara.integrations import local_memory as m;"
             "print(json.dumps(m._MAX_CONTENT_BYTES))"],
            capture_output=True, text=True, env=env,
            stdin=subprocess.DEVNULL, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout.strip().splitlines()[-1])
        documented = int(_readme_env_defaults()["CLARA_MAX_CONTENT_BYTES"])
        assert actual == documented, (
            f"README says CLARA_MAX_CONTENT_BYTES={documented}, code default "
            f"is {actual}"
        )

    def test_backup_keep_default(self, clean_env):
        from clara.db import backup

        documented = int(_readme_env_defaults()["CLARA_BACKUP_KEEP"])
        keep = getattr(backup, "DEFAULT_KEEP", None) or getattr(backup, "_KEEP", None)
        assert keep is not None, "backup module exposes no default-keep constant"
        assert keep == documented, (
            f"README says CLARA_BACKUP_KEEP={documented}, code default is {keep}"
        )

    def test_secret_policy_default_is_reject(self, clean_env):
        """The default is the safe one; the README says so in two places."""
        from clara import security

        assert security.secret_policy() == "reject"


class TestDocumentedCommandsExist:
    def test_every_documented_cli_subcommand_is_real(self):
        """`clara <verb>` shown in the README must be a registered subcommand.

        A documented command that does not exist sends the user to a dead end,
        which is exactly what `clara doctor` was for plugin installs.
        """
        import argparse
        import contextlib
        import io

        from clara import cli

        parser_help = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(parser_help):
            cli.main(["--help"])
        text = parser_help.getvalue()
        match = re.search(r"\{([a-z,]+)\}", text)
        assert match, "could not read the subcommand list from --help"
        real = set(match.group(1).split(","))

        readme = (_ROOT / "README.md").read_text("utf-8")
        documented = set(re.findall(r"^\s*clara ([a-z]+)", readme, re.MULTILINE))
        documented -= {"memory"}  # `clara memory` appears only in prose examples
        phantom = documented - real
        assert not phantom, f"README documents non-existent commands: {sorted(phantom)}"
        assert isinstance(argparse.ArgumentParser, type)  # import used


class TestSkillMatchesTheToolSurface:
    """The skill is instructions the model acts on, so a stale tool name there
    is worse than a stale README line: it tells Claude to call something that
    does not exist, or hides one that does."""

    def _skill_text(self) -> str:
        return (
            _ROOT / "skills" / "using-clara-memory" / "SKILL.md"
        ).read_text("utf-8")

    def _named_tools(self) -> set[str]:
        return set(
            re.findall(
                r"`(memory_[a-z_]+|docs_[a-z_]+|graph_[a-z_]+|statusline_[a-z_]+"
                r"|project_[a-z_]+)`",
                self._skill_text(),
            )
        )

    def _real_tools(self) -> set[str]:
        import asyncio

        from clara.integrations.mcp_server import build_server

        return {tool.name for tool in asyncio.run(build_server().list_tools())}

    def test_skill_names_no_tool_that_does_not_exist(self):
        phantom = self._named_tools() - self._real_tools()
        assert not phantom, f"skill tells Claude to call missing tools: {sorted(phantom)}"

    def test_skill_names_every_real_tool(self):
        missing = self._real_tools() - self._named_tools()
        assert not missing, (
            f"these tools exist but the skill never mentions them: {sorted(missing)}"
        )

    def test_skill_tool_count_matches(self):
        """The skill opens with a count; it must not drift from the surface."""
        text = self._skill_text()
        match = re.search(r"(\d+)\s+MCP tools", text)
        assert match, "the skill no longer states a tool count"
        assert int(match.group(1)) == len(self._real_tools())


class TestHousekeepingClaim:
    """Housekeeping runs from the MCP server only.

    _run_maintenance_if_due lives in clara/integrations/mcp_server.py and has
    exactly one production call site, there. Verified against a CLI-only store:
    `clara remember` plus four `clara stats` produced no .maintenance marker and
    no daily backup. The README used to say it ran "the first time the store is
    opened each day", which reads as any open and left a CLI-only user
    expecting decay that never happens.
    """

    def test_only_the_mcp_server_triggers_maintenance(self):
        call_sites = []
        for path in (_ROOT / "clara").rglob("*.py"):
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                if "_run_maintenance_if_due(" in line and "def " not in line:
                    call_sites.append(f"{path.relative_to(_ROOT)}:{lineno}")
        assert call_sites, "maintenance is never triggered at all"
        assert all("mcp_server.py" in site for site in call_sites), (
            f"maintenance now runs from somewhere else too: {call_sites} — the "
            "README says the MCP server is the only trigger, so update it"
        )

    def test_readme_says_where_housekeeping_runs(self):
        readme = (_ROOT / "README.md").read_text("utf-8")
        section = readme[readme.index("- **Housekeeping**"):][:600]
        assert "MCP server" in section, (
            "the README must name the MCP server as the trigger, not imply any "
            "store open will do"
        )


class TestReadmeMechanismClaims:
    """The README now explains how CLARA works and compares with/without it.

    Those are factual claims about mechanism, so they get the same treatment as
    the documented defaults: asserted against the code, not trusted.
    """

    def _readme(self) -> str:
        return (_ROOT / "README.md").read_text("utf-8")

    def test_fenced_section_line_cap_matches_the_code(self):
        from clara.bridge.exporter import SECTION_MAX_LINES

        assert f"{SECTION_MAX_LINES}-line" in self._readme() or \
               f"≤{SECTION_MAX_LINES}" in self._readme() or \
               f"≤ {SECTION_MAX_LINES}" in self._readme(), (
            f"README should quote the real cap ({SECTION_MAX_LINES})"
        )

    def test_session_start_matcher_is_quoted_correctly(self):
        import json

        hooks = json.loads((_ROOT / "hooks" / "hooks.json").read_text("utf-8"))
        matcher = hooks["hooks"]["SessionStart"][0]["matcher"]
        assert matcher in self._readme(), (
            "the README quotes which session events re-inject memory; it must "
            f"match hooks.json ({matcher})"
        )

    def test_tool_count_claim_matches_the_server(self):
        import asyncio
        import re

        from clara.integrations.mcp_server import build_server

        real = len(asyncio.run(build_server().list_tools()))
        claimed = {int(n) for n in re.findall(r"(\d+) MCP tools", self._readme())}
        assert claimed, "the README no longer states a tool count"
        assert claimed == {real}, f"README claims {claimed} tools, server has {real}"

    def test_no_semantic_search_claim_holds(self):
        """The README states the zero-key tier does keyword search, not vectors.

        If LocalMemory.search ever grows an embedding path, that sentence
        becomes false and this fails.
        """
        import inspect

        from clara.integrations.local_memory import LocalMemory

        source = inspect.getsource(LocalMemory.search)
        assert "embed" not in source.lower(), (
            "search now embeds; the README's 'no semantic search in the "
            "zero-key tier' claim needs updating"
        )

    def test_comparison_table_disclaims_unmeasured_benefits(self):
        """Guard against the table drifting into marketing.

        The with/without section exists to describe mechanism. It must keep
        saying that productivity and token effects were not measured.
        """
        readme = self._readme()
        assert "not been measured" in readme or "has not been benchmarked" in readme
        assert "does not make the model smarter" in readme
