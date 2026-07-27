"""Tests for the status-line sidecar counter (clara/stats_cache.py).

The status line is pulled on a timer, so the counter must be cheap, always
fresh after a write, and incapable of breaking either the save path or the
status line itself (a status line that exits non-zero blanks the bar).
"""

from __future__ import annotations

import os
import sqlite3
import time

import pytest

from clara import stats_cache
from clara.db.migrations import ensure_schema
from clara.integrations.local_memory import LocalMemory


def _make_store(tmp_path) -> str:
    db = tmp_path / "clara.db"
    conn = sqlite3.connect(db)
    try:
        ensure_schema(conn)
    finally:
        conn.close()
    return str(db)


class TestSidecarFile:
    def test_write_then_read_roundtrip(self, tmp_path):
        db = _make_store(tmp_path)
        stats_cache.write(db, 42)
        assert stats_cache.read(db) == 42

    def test_read_missing_returns_none(self, tmp_path):
        assert stats_cache.read(str(tmp_path / "nope.db")) is None

    def test_read_stale_returns_none(self, tmp_path):
        db = _make_store(tmp_path)
        stats_cache.write(db, 7)
        # Age the sidecar explicitly rather than passing max_age_s=0: on
        # Windows the file's mtime can land a hair in the future relative to
        # time.time(), making "now - mtime >= 0" briefly false and the test
        # flaky. Anything older than max_age must read as absent so the caller
        # recounts instead of trusting a number from a previous CLARA version.
        cache = stats_cache.path_for(db)
        old = time.time() - 3600
        os.utime(cache, (old, old))
        assert stats_cache.read(db, max_age_s=60.0) is None

    def test_read_corrupt_returns_none(self, tmp_path):
        db = _make_store(tmp_path)
        stats_cache.path_for(db).write_text("not-a-number", encoding="utf-8")
        assert stats_cache.read(db) is None

    def test_write_is_atomic_no_temp_files_left(self, tmp_path):
        db = _make_store(tmp_path)
        stats_cache.write(db, 3)
        leftovers = list(tmp_path.glob(".stats-*"))
        assert leftovers == []

    def test_write_to_unwritable_dir_does_not_raise(self, tmp_path):
        # Cosmetic counter: a failure must never propagate into a save.
        stats_cache.write(str(tmp_path / "missing-dir" / "clara.db"), 1)

    def test_refresh_on_missing_store_returns_none(self, tmp_path):
        assert stats_cache.refresh(str(tmp_path / "absent.db")) is None


class TestRefreshCountsActiveRows:
    def test_refresh_counts_only_active(self, tmp_path):
        db = _make_store(tmp_path)
        conn = sqlite3.connect(db)
        try:
            rows = [
                ("a" * 32, "belief", "active"),
                ("b" * 32, "belief", "active"),
                ("c" * 32, "belief", "archived"),
            ]
            for mid, mtype, status in rows:
                conn.execute(
                    "INSERT INTO memories (memory_id, memory_type, content, "
                    "confidence, status, decay_rate, created_at, updated_at) "
                    "VALUES (?, ?, '{}', 0.9, ?, 0.02, "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    (mid, mtype, status),
                )
            conn.commit()
        finally:
            conn.close()

        assert stats_cache.refresh(db) == 2
        assert stats_cache.read(db) == 2


class TestCounterTracksWrites:
    """The regression this module exists for: the counter used to be written
    only by the status line itself, so it reported a stale number for up to
    five minutes after a save."""

    @pytest.mark.asyncio
    async def test_save_refreshes_counter(self, tmp_path):
        db = _make_store(tmp_path)
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="user", relation="uses", object="rust"
            )
            assert stats_cache.read(db) == 1

            await memory.save(
                mem_type="belief", subject="user", relation="uses", object="go"
            )
            assert stats_cache.read(db) == 2
        finally:
            await memory.close()

    @pytest.mark.asyncio
    async def test_forget_decrements_counter(self, tmp_path):
        db = _make_store(tmp_path)
        memory = await LocalMemory.create(db)
        try:
            saved = await memory.save(
                mem_type="belief", subject="user", relation="uses", object="rust"
            )
            assert stats_cache.read(db) == 1

            await memory.forget(saved["memory_id"])
            assert stats_cache.read(db) == 0
        finally:
            await memory.close()

    @pytest.mark.asyncio
    async def test_counter_is_fresh_not_merely_present(self, tmp_path):
        # Guard against a regression where the file exists but is never
        # updated: its mtime must move when the store changes.
        db = _make_store(tmp_path)
        memory = await LocalMemory.create(db)
        try:
            await memory.save(
                mem_type="belief", subject="user", relation="uses", object="rust"
            )
            first = stats_cache.path_for(db).stat().st_mtime
            time.sleep(0.01)
            await memory.save(
                mem_type="belief", subject="user", relation="uses", object="zig"
            )
            assert stats_cache.path_for(db).stat().st_mtime >= first
            assert stats_cache.read(db) == 2
        finally:
            await memory.close()
