"""Tests for export/import (clara/porting.py) and backup/restore (clara/db/backup.py)."""

from __future__ import annotations

import io
import json
import sqlite3

import pytest

from clara.db.backup import backup_db, backup_dir, latest_backup
from clara.integrations.local_memory import LocalMemory
from clara.porting import ImportError_, export_records, import_records, write_export


async def _seed(db_path: str, n: int = 3) -> LocalMemory:
    memory = await LocalMemory.create(db_path)
    for i in range(n):
        await memory.save(
            mem_type="belief",
            subject=f"svc{i}",
            relation="deployed_to",
            object=f"host{i}",
            tags=[f"tag{i}"],
        )
    return memory


class TestExport:
    async def test_header_and_rows(self, tmp_path):
        db = str(tmp_path / "e.db")
        memory = await _seed(db)
        await memory.close()
        records = list(export_records(db))
        assert records[0]["kind"] == "header"
        assert records[0]["format"] == 1
        assert records[0]["counts"]["memories"] == 3
        memories = [r for r in records[1:] if r["kind"] == "memory"]
        assert len(memories) == 3
        assert all("metadata" in r for r in memories)
        assert all(isinstance(r["content"], dict) for r in memories)

    async def test_filters(self, tmp_path):
        db = str(tmp_path / "f.db")
        memory = await _seed(db)
        await memory.save(mem_type="event", subject="ci", event_type="released")
        await memory.close()
        only_events = [
            r for r in export_records(db, types=["event"]) if r["kind"] == "memory"
        ]
        assert len(only_events) == 1
        assert only_events[0]["memory_type"] == "event"


class TestImportRoundTrip:
    async def test_round_trip_preserves_everything(self, tmp_path):
        src = str(tmp_path / "src.db")
        memory = await _seed(src)
        await memory.close()
        buffer = io.StringIO()
        write_export(export_records(src), buffer)

        dst = str(tmp_path / "dst.db")
        stats = import_records(dst, buffer.getvalue().splitlines())
        assert stats["imported"] == 3
        assert stats["skipped_id"] == 0

        src_rows = sqlite3.connect(src).execute(
            "SELECT memory_id, content, created_at, updated_at FROM memories "
            "ORDER BY memory_id"
        ).fetchall()
        dst_rows = sqlite3.connect(dst).execute(
            "SELECT memory_id, content, created_at, updated_at FROM memories "
            "ORDER BY memory_id"
        ).fetchall()
        assert src_rows == dst_rows

        # FTS indexed the imported rows (triggers fired on raw INSERT).
        fts = sqlite3.connect(dst).execute(
            "SELECT count(*) FROM memories_fts"
        ).fetchone()[0]
        assert fts == 3

        found = None
        mem2 = await LocalMemory.create(dst)
        try:
            found = await mem2.search("svc1", top_k=3)
        finally:
            await mem2.close()
        assert found is not None and found["total"] >= 1

    async def test_conflict_skip_and_force(self, tmp_path):
        db = str(tmp_path / "c.db")
        memory = await _seed(db, n=1)
        await memory.close()
        buffer = io.StringIO()
        write_export(export_records(db), buffer)
        lines = buffer.getvalue().splitlines()

        again = import_records(db, lines)
        assert again["imported"] == 0
        assert again["skipped_id"] == 1

        forced = import_records(db, lines, on_conflict="force")
        assert forced["conflicts_updated"] == 1

    async def test_belief_identity_dedup(self, tmp_path):
        src = str(tmp_path / "s.db")
        memory = await _seed(src, n=1)
        await memory.close()
        buffer = io.StringIO()
        write_export(export_records(src), buffer)
        # Same belief content under a different memory_id.
        lines = []
        for line in buffer.getvalue().splitlines():
            record = json.loads(line)
            if record["kind"] == "memory":
                record["memory_id"] = "f" * 32
            lines.append(json.dumps(record))
        stats = import_records(src, lines)
        assert stats["skipped_content"] == 1
        assert stats["imported"] == 0

    def test_rejects_newer_format(self, tmp_path):
        db = str(tmp_path / "n.db")
        header = json.dumps({"kind": "header", "format": 99, "schema_version": 1})
        with pytest.raises(ImportError_, match="newer"):
            import_records(db, [header])

    def test_rejects_missing_header(self, tmp_path):
        db = str(tmp_path / "m.db")
        with pytest.raises(ImportError_, match="header"):
            import_records(db, ['{"kind": "memory"}'])

    async def test_dry_run_writes_nothing(self, tmp_path):
        src = str(tmp_path / "d.db")
        memory = await _seed(src, n=2)
        await memory.close()
        buffer = io.StringIO()
        write_export(export_records(src), buffer)
        dst = str(tmp_path / "d2.db")
        stats = import_records(dst, buffer.getvalue().splitlines(), dry_run=True)
        assert stats["imported"] == 2
        count = sqlite3.connect(dst).execute(
            "SELECT count(*) FROM memories"
        ).fetchone()[0]
        assert count == 0

    async def test_secret_rows_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_SECRET_POLICY", "reject")
        db = str(tmp_path / "sec.db")
        header = json.dumps({"kind": "header", "format": 1, "schema_version": 5})
        row = json.dumps({
            "kind": "memory", "memory_id": "a" * 32, "memory_type": "belief",
            "content": {"subject": "x", "relation": "uses",
                        "object": "AKIAIOSFODNN7EXAMPLE"},
            "confidence": 1.0, "status": "active", "decay_rate": 0.02,
            "created_at": "2026-01-01 00:00:00+00:00",
            "updated_at": "2026-01-01 00:00:00+00:00", "metadata": {},
        })
        stats = import_records(db, [header, row])
        assert stats["skipped_secret"] == 1
        assert stats["imported"] == 0


class TestBackup:
    async def test_backup_and_rotation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLARA_BACKUP_KEEP", "2")
        db = str(tmp_path / "b.db")
        memory = await _seed(db, n=1)
        await memory.close()
        paths = []
        for i in range(3):
            dest = backup_db(db, reason=f"r{i}")
            assert dest is not None
            # mtimes must differ for rotation ordering
            import os
            import time

            os.utime(dest, (time.time() + i, time.time() + i))
            paths.append(dest)
        remaining = sorted(backup_dir(db).glob("clara-*.db"))
        assert len(remaining) == 2
        assert paths[0] not in remaining

    async def test_backup_passes_integrity(self, tmp_path):
        db = str(tmp_path / "i.db")
        memory = await _seed(db, n=2)
        await memory.close()
        dest = backup_db(db)
        assert dest is not None
        check = sqlite3.connect(dest).execute("PRAGMA integrity_check").fetchone()[0]
        assert check == "ok"
        count = sqlite3.connect(dest).execute(
            "SELECT count(*) FROM memories"
        ).fetchone()[0]
        assert count == 2

    async def test_latest_backup(self, tmp_path):
        db = str(tmp_path / "l.db")
        memory = await _seed(db, n=1)
        await memory.close()
        assert latest_backup(db) is None
        dest = backup_db(db)
        assert latest_backup(db) == dest

    def test_backup_missing_store_returns_none(self, tmp_path):
        assert backup_db(str(tmp_path / "nope.db")) is None


class TestPreMigrationBackup:
    def test_upgrade_takes_backup_first(self, tmp_path):
        """A v3-era store upgrading to v5 gets a pre-migration snapshot."""
        import clara.db.migrations as mig

        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        for stmt in mig._MEMORIES_DDL:
            conn.execute(stmt)
        conn.execute(
            "CREATE TABLE schema_info (version INTEGER NOT NULL PRIMARY KEY, "
            "migrated_at TEXT NOT NULL)"
        )
        for v in (1, 2, 3):
            conn.execute(
                "INSERT INTO schema_info (version, migrated_at) VALUES (?, 'x')", (v,)
            )
        for stmt in mig._GRAPH_DDL:
            conn.execute(stmt)
        for stmt in mig._DOCS_DDL:
            conn.execute(stmt)
        conn.commit()

        from clara.db.backup import backup_before_migration

        backup_before_migration(conn, str(db))
        conn.close()
        snapshots = list(backup_dir(db).glob("clara-*pre-migration*"))
        assert len(snapshots) == 1

    def test_fresh_store_skips_backup(self, tmp_path):
        db = tmp_path / "fresh.db"
        conn = sqlite3.connect(db)
        from clara.db.backup import backup_before_migration

        backup_before_migration(conn, str(db))
        conn.close()
        assert not backup_dir(db).exists()
