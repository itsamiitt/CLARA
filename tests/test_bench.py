"""Latency benchmarks over synthetic stores (marker: bench — excluded from
the default run; CI runs `pytest -m bench` on a dedicated job).

Assertions carry ~2x headroom over observed dev-box numbers and use
median-of-3 to damp runner variance. They exist to catch order-of-magnitude
regressions (a scan sneaking back in, an unbatched pass), not to be a
precise performance suite.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.synth import build_store
from clara.integrations.local_memory import LocalMemory

pytestmark = pytest.mark.bench

_ROOT = Path(__file__).parents[1]
ROWS = 10_000

# Windows process spawn + filesystem overhead runs 2-4x POSIX; budgets scale.
_WIN = sys.platform == "win32"
FASTPATH_BUDGET_S = 3.0 if _WIN else 1.5   # end-to-end incl. interpreter start
SEARCH_BUDGET_S = 0.35 if _WIN else 0.25
RECENT_BUDGET_S = 0.40 if _WIN else 0.30
SAVE_BUDGET_S = 0.20 if _WIN else 0.10
DECAY_BUDGET_S = 20.0 if _WIN else 10.0


@pytest.fixture(scope="module")
def bench_store(tmp_path_factory):
    db = tmp_path_factory.mktemp("bench") / "bench.db"
    build_store(str(db), ROWS)
    return db


def _median(samples):
    return statistics.median(samples)


class TestBench10k:
    def test_fastpath_end_to_end(self, bench_store, monkeypatch):
        monkeypatch.setenv("CLARA_DB_PATH", str(bench_store))
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-m", "clara.fastpath.context", "--cwd", str(_ROOT)],
                capture_output=True,
                text=True,
                timeout=60,
                env={**__import__("os").environ, "CLARA_DB_PATH": str(bench_store)},
            )
            timings.append(time.perf_counter() - start)
            assert proc.returncode == 0
            assert "MEMORY CONTEXT" in proc.stdout
        assert _median(timings) < FASTPATH_BUDGET_S, f"fastpath {timings}"

    async def test_search_fts(self, bench_store):
        memory = await LocalMemory.create(str(bench_store))
        try:
            timings = []
            for query in ("postgresql deployment", "kubernetes", "dark mode"):
                start = time.perf_counter()
                result = await memory.search(query, top_k=8)
                timings.append(time.perf_counter() - start)
                assert result["total"] > 0
            assert _median(timings) < SEARCH_BUDGET_S, f"search {timings}"
        finally:
            await memory.close()

    async def test_recent(self, bench_store):
        memory = await LocalMemory.create(str(bench_store))
        try:
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                result = await memory.recent(n=10)
                timings.append(time.perf_counter() - start)
                assert result["total"] > 0
            assert _median(timings) < RECENT_BUDGET_S, f"recent {timings}"
        finally:
            await memory.close()

    async def test_save(self, bench_store):
        memory = await LocalMemory.create(str(bench_store))
        try:
            timings = []
            for i in range(3):
                start = time.perf_counter()
                await memory.save(
                    mem_type="belief",
                    subject=f"bench{i}",
                    relation="measures",
                    object="save latency",
                )
                timings.append(time.perf_counter() - start)
            assert _median(timings) < SAVE_BUDGET_S, f"save {timings}"
        finally:
            await memory.close()

    async def test_decay_pass_batched(self, bench_store):
        """Full decay over 10k rows: bounded wall time AND no single
        transaction long enough to starve a concurrent writer."""
        from clara.scheduler.decay import DecayScheduler

        memory = await LocalMemory.create(str(bench_store))
        try:
            scheduler = DecayScheduler(memory._session_factory)
            start = time.perf_counter()
            summary = await scheduler.run_daily_decay()
            elapsed = time.perf_counter() - start
            assert summary["decayed"] + summary["archived"] > 0
            assert elapsed < DECAY_BUDGET_S, f"decay pass took {elapsed:.1f}s"
        finally:
            await memory.close()
