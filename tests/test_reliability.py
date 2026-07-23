"""Tests for reliability hardening: retry, maintenance lock, pragmas,
access recording, and the create_all/migrations parity lock."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time

import pytest
from sqlalchemy.exc import OperationalError

from clara.db import backup as backup_mod
from clara.db.migrations import SCHEMA_VERSION, check_version, ensure_schema
from clara.db.retry import with_sqlite_retry
from clara.integrations.local_memory import LocalMemory


class TestRetry:
    async def test_returns_value_first_try(self):
        async def op():
            return 42

        assert await with_sqlite_retry(op) == 42

    async def test_retries_lock_then_succeeds(self):
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            if calls["n"] < 2:
                raise OperationalError("stmt", {}, Exception("database is locked"))
            return "ok"

        assert await with_sqlite_retry(op) == "ok"
        assert calls["n"] == 2

    async def test_non_lock_error_propagates_immediately(self):
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise OperationalError("stmt", {}, Exception("no such table: nope"))

        with pytest.raises(OperationalError):
            await with_sqlite_retry(op)
        assert calls["n"] == 1

    async def test_final_failure_message_is_actionable(self, monkeypatch):
        monkeypatch.setattr("clara.db.retry._DELAYS_S", (0.0, 0.0, 0.0))

        async def op():
            raise OperationalError("stmt", {}, Exception("database is locked"))

        with pytest.raises(RuntimeError, match="not persisted — retry shortly"):
            await with_sqlite_retry(op, what="save")


class TestMaintenanceLock:
    async def test_single_winner(self, tmp_path, monkeypatch):
        """Two concurrent due-maintenance calls: exactly one pass runs."""
        from clara.integrations import mcp_server

        db_path = str(tmp_path / "m.db")
        memory = await LocalMemory.create(db_path)
        runs = {"n": 0}

        class FakeScheduler:
            def __init__(self, *_a, **_k):
                pass

            async def run_daily_decay(self):
                runs["n"] += 1
                await asyncio.sleep(0.05)
                return "decay: fake"

            async def run_weekly_pruning(self):
                return "prune: fake"

        import clara.scheduler.decay as decay_mod

        monkeypatch.setattr(decay_mod, "DecayScheduler", FakeScheduler)
        monkeypatch.setattr(backup_mod, "backup_db", lambda *_a, **_k: None)
        try:
            await asyncio.gather(
                mcp_server._run_maintenance_if_due(memory, db_path),
                mcp_server._run_maintenance_if_due(memory, db_path),
            )
            assert runs["n"] == 1
            assert os.path.exists(db_path + ".maintenance")
            assert not os.path.exists(db_path + ".maintenance.lock")
        finally:
            await memory.close()

    async def test_stale_lock_recovered(self, tmp_path, monkeypatch):
        from clara.integrations import mcp_server

        db_path = str(tmp_path / "m.db")
        memory = await LocalMemory.create(db_path)
        lock = tmp_path / "m.db.maintenance.lock"
        lock.write_text("999,0")
        stale = time.time() - (7 * 3600)
        os.utime(lock, (stale, stale))
        ran = {"n": 0}

        class FakeScheduler:
            def __init__(self, *_a, **_k):
                pass

            async def run_daily_decay(self):
                ran["n"] += 1
                return "decay: fake"

            async def run_weekly_pruning(self):
                return "prune: fake"

        import clara.scheduler.decay as decay_mod

        monkeypatch.setattr(decay_mod, "DecayScheduler", FakeScheduler)
        monkeypatch.setattr(backup_mod, "backup_db", lambda *_a, **_k: None)
        try:
            await mcp_server._run_maintenance_if_due(memory, db_path)
            assert ran["n"] == 1
            assert not lock.exists()
        finally:
            await memory.close()


class TestPragmas:
    def test_fastpath_connection_is_read_only(self, tmp_path):
        from clara.fastpath.db import open_store

        db = tmp_path / "ro.db"
        seed = sqlite3.connect(db)
        ensure_schema(seed)
        seed.close()
        conn = open_store(db)
        assert conn is not None
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE nope (x)")
        conn.close()

    async def test_runtime_writer_uses_wal(self, tmp_path):
        db_path = str(tmp_path / "wal.db")
        memory = await LocalMemory.create(db_path)
        try:
            await memory.save(
                mem_type="belief", subject="a", relation="b", object="c"
            )
        finally:
            await memory.close()
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"


class TestAccessRecording:
    async def test_search_bumps_usage_not_updated_at(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        memory = await LocalMemory.create(db_path)
        try:
            saved = await memory.save(
                mem_type="belief",
                subject="user",
                relation="prefers",
                object="postgres",
            )
            conn = sqlite3.connect(db_path)
            before = conn.execute(
                "SELECT updated_at FROM memories WHERE memory_id = ?",
                (saved["memory_id"].replace("-", ""),),
            ).fetchone()
            conn.close()

            await memory.search("postgres", top_k=3)
            await memory.search("postgres", top_k=3)

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT updated_at, json_extract(metadata, '$.access_count') "
                "FROM memories WHERE memory_id = ?",
                (saved["memory_id"].replace("-", ""),),
            ).fetchone()
            conn.close()
            assert row is not None
            updated_at, access_count = row
            assert access_count == 2
            assert before is not None
            assert updated_at == before[0], "a read must not look like a write"
        finally:
            await memory.close()

    async def test_recent_does_not_bump(self, tmp_path):
        db_path = str(tmp_path / "recent.db")
        memory = await LocalMemory.create(db_path)
        try:
            saved = await memory.save(
                mem_type="belief", subject="a", relation="b", object="c"
            )
            await memory.recent(n=5)
            conn = sqlite3.connect(db_path)
            count = conn.execute(
                "SELECT json_extract(metadata, '$.access_count') FROM memories "
                "WHERE memory_id = ?",
                (saved["memory_id"].replace("-", ""),),
            ).fetchone()[0]
            conn.close()
            assert count is None
        finally:
            await memory.close()


class TestSchemaParity:
    async def test_create_all_matches_frozen_migrations(self, tmp_path):
        """The drift alarm: migration-4 DDL must equal what create_all emits."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from clara.db.models import Base

        orm_db = tmp_path / "orm.db"
        eng = create_async_engine(f"sqlite+aiosqlite:///{orm_db.as_posix()}")
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await eng.dispose()

        mig_db = tmp_path / "mig.db"
        conn2 = sqlite3.connect(mig_db)
        ensure_schema(conn2)
        conn2.close()

        def normalized(path):
            conn = sqlite3.connect(path)
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL "
                "AND name LIKE '%memories%' AND name NOT LIKE '%fts%' "
                "ORDER BY name"
            ).fetchall()
            conn.close()
            return {
                name: " ".join(
                    sql.replace("IF NOT EXISTS ", "").replace('"', "").split()
                )
                for name, sql in rows
            }

        assert normalized(orm_db) == normalized(mig_db)

    async def test_fresh_store_via_migrations_only_works(self, tmp_path):
        db = tmp_path / "fresh.db"
        conn = sqlite3.connect(db)
        version = ensure_schema(conn)
        conn.close()
        assert version == SCHEMA_VERSION
        memory = await LocalMemory.create(str(db))
        try:
            await memory.save(
                mem_type="belief", subject="x", relation="y", object="z"
            )
            found = await memory.search("z", top_k=3)
            assert found["total"] == 1
        finally:
            await memory.close()

    def test_check_version_never_writes(self, tmp_path):
        db = tmp_path / "check.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (x)")
        conn.commit()
        conn.close()
        before = db.read_bytes()
        conn = sqlite3.connect(db)
        assert check_version(conn) == 0
        conn.close()
        assert db.read_bytes() == before

    def test_legacy_v3_store_upgrades_with_rows_intact(self, tmp_path):
        """A pre-v4 store (create_all memories + v3 migrations) upgrades to v5
        with data preserved and FTS repopulated under the new text expr."""
        import clara.db.migrations as mig

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        # Simulate the legacy layout: memories from create_all-era DDL,
        # schema_info at v3.
        for stmt in mig._MEMORIES_DDL:
            conn.execute(stmt)
        conn.execute(
            "CREATE TABLE schema_info (version INTEGER NOT NULL PRIMARY KEY, "
            "migrated_at TEXT NOT NULL)"
        )
        for v in (1, 2, 3):
            # Graph/docs tables from their DDL so v2/v3 content exists.
            conn.execute(
                "INSERT INTO schema_info (version, migrated_at) VALUES (?, 'x')",
                (v,),
            )
        for stmt in mig._GRAPH_DDL:
            conn.execute(stmt)
        for stmt in mig._DOCS_DDL:
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO memories (memory_id, memory_type, content, confidence, "
            "status, decay_rate, metadata) VALUES ('abc123', 'belief', "
            "'{\"subject\": \"user\", \"relation\": \"uses\", \"object\": \"vim\"}', "
            "0.9, 'active', 0.02, '{\"tags\": [\"editor\"]}')"
        )
        conn.commit()
        version = ensure_schema(conn)
        assert version == SCHEMA_VERSION
        row = conn.execute(
            "SELECT count(*) FROM memories WHERE status = 'active'"
        ).fetchone()
        assert row[0] == 1
        fts = conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        assert fts == 1
        # v5 text expression indexes tags:
        hit = conn.execute(
            "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH '\"editor\"'"
        ).fetchall()
        conn.close()
        assert len(hit) == 1
