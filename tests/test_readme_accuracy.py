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
