"""Regression tests for findings from the 2026-07-23 adversarial review.

Each guards a bug that shipped in 0.2.0 and was fixed in response to the
review (see docs/reviews/2026-07-23/FINDINGS.md F1, F2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clara.integrations.local_memory import LocalMemory

_ROOT = Path(__file__).parents[1]
_SCRIPTS = [
    "scripts/session-start.sh",
    "scripts/session-stop.sh",
    "scripts/read-annotate.sh",
    "scripts/bootstrap.sh",
    "scripts/clara-mcp-launch.sh",
]


def _bash() -> str | None:
    import shutil

    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe"):
        if not cand:
            continue
        try:
            probe = subprocess.run([cand, "-c", "echo ok"], capture_output=True,
                                   text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return cand
    return None


class TestF1HomeUnbound:
    """F1: hooks must not die on `set -u` when HOME (and CLARA_HOME) is unset.
    They reference ${CLARA_HOME:-${HOME:-/tmp}/.clara}; a bare $HOME aborted
    the script with 'unbound variable' and a non-zero exit, breaking the
    'SessionStart always exits 0' contract."""

    @pytest.mark.parametrize("script", ["session-start.sh", "session-stop.sh",
                                        "read-annotate.sh"])
    def test_hook_exits_zero_without_home(self, script):
        bash = _bash()
        if bash is None:
            pytest.skip("no working bash")
        import os

        env = {k: v for k, v in os.environ.items()
               if k not in ("HOME", "CLARA_HOME", "CLAUDE_PLUGIN_DATA")}
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        proc = subprocess.run(
            [bash, str(_ROOT / "scripts" / script)],
            capture_output=True, text=True, timeout=60, env=env, input="{}",
        )
        assert proc.returncode == 0, (
            f"{script} exited {proc.returncode} with HOME unset: {proc.stderr[-200:]}"
        )
        assert "unbound variable" not in proc.stderr

    def test_no_bare_home_under_set_u(self):
        """Static guard: no script references $HOME without a :- fallback."""
        import re

        pattern = re.compile(r"\$HOME(?![:}])|\$\{HOME\}")
        offenders = []
        for rel in _SCRIPTS:
            text = (_ROOT / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                # allow ${HOME:-...} and ${HOME:=...}
                stripped = re.sub(r"\$\{HOME:[-=][^}]*\}", "", line)
                if pattern.search(stripped):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert not offenders, "bare $HOME under set -u:\n" + "\n".join(offenders)


class TestF2TopKClamp:
    """F2: negative top_k became a Python slice scored[:-1] and silently
    dropped the lowest-ranked hit instead of returning the requested count."""

    async def test_negative_top_k_returns_empty_not_dropped(self, tmp_path):
        mem = await LocalMemory.create(str(tmp_path / "tk.db"))
        try:
            for i in range(5):
                await mem.save(mem_type="belief", subject=f"s{i}",
                               relation="r", object=f"thing{i}")
            neg = await mem.search("thing", top_k=-1)
            assert neg["total"] == 0, "negative top_k must clamp, not slice[:-1]"
            zero = await mem.search("thing", top_k=0)
            assert zero["total"] == 0
            three = await mem.search("thing", top_k=3)
            assert three["total"] == 3
        finally:
            await mem.close()
