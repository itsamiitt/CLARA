"""
The index pass must plan without the write lock and apply briefly.

SQLite allows one writer per database, and the store is shared between the
background index and every interactive memory_save. The two shapes that
preceded this one both failed people, measured cross-process on a 2,568-file
repo:

* one transaction for the whole walk — concurrent saves waited up to 6.4 s and
  two of five were refused outright after the 30 s busy timeout;
* parse work interleaved inside chunked transactions with a 20 ms yield — a
  waiting writer still starved for 7.1 s, because SQLite's busy handler polls
  ~100 ms apart and never landed inside a 20 ms gap.

Two-phase fixes it structurally: planning (stat, read, hash, parse, resolve)
runs with no transaction open, and only applying the accumulated plans takes
the lock, in bounded batches with a yield a waiter can actually catch. Same
harness afterwards: worst wait 0.5 s, 76 probe writes through instead of 11,
identical nodes and edges.

The tests here pin the structure, not the timings — timings belong to the
benchmark, structure is what regresses silently.
"""

from __future__ import annotations

import sqlite3

import pytest

from clara.db.migrations import ensure_schema
from clara.index import indexer

REPO = "twophase-repo"


class GuardedConnection:
    """Delegates to a real connection but refuses writes.

    If planning ever writes, index_repo is holding the lock during the parse
    phase again and the whole point is lost — better a loud test failure here
    than a 7 s stall found by a probe later.
    """

    WRITE_VERBS = ("insert", "update", "delete", "replace", "create", "drop")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.statements: list[str] = []

    def execute(self, sql: str, *args):
        self.statements.append(sql)
        verb = sql.split(None, 1)[0].lower() if sql.split() else ""
        assert verb not in self.WRITE_VERBS, (
            f"planning executed a write: {sql[:100]}"
        )
        return self._conn.execute(sql, *args)


@pytest.fixture()
def conn(tmp_path):
    connection = sqlite3.connect(tmp_path / "clara.db")
    ensure_schema(connection)
    yield connection
    connection.close()


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "a.py").write_text("from pkg import b\n", encoding="utf-8")
    (root / "pkg" / "b.py").write_text("import sys\n", encoding="utf-8")
    (root / "app.ts").write_text('import {a} from "./lib";\n', encoding="utf-8")
    (root / "lib.ts").write_text("export const a = 1;\n", encoding="utf-8")
    return root


class TestPlanningWritesNothing:
    def test_planning_a_fresh_file_only_reads(self, conn, repo) -> None:
        guarded = GuardedConnection(conn)
        for rel in indexer.walk_repo(repo):
            plan = indexer._plan_file(guarded, REPO, repo, rel)
            assert plan.action == "index"
        # The guard only proves something if planning consulted the store.
        assert guarded.statements, "planning never touched the connection"

    def test_planning_an_unchanged_file_only_reads(self, conn, repo) -> None:
        result = indexer.IndexResult()
        for rel in indexer.walk_repo(repo):
            indexer.index_file(conn, REPO, repo, rel, result)
        conn.commit()

        guarded = GuardedConnection(conn)
        for rel in indexer.walk_repo(repo):
            plan = indexer._plan_file(guarded, REPO, repo, rel)
            assert plan.action == "skip"

    def test_planning_a_vanished_file_only_reads(self, conn, repo) -> None:
        plan = indexer._plan_file(GuardedConnection(conn), REPO, repo, "gone.py")
        assert plan.action == "retire"


class TestPlanActions:
    def test_stat_moved_bytes_same_is_a_refresh(self, conn, repo) -> None:
        import os

        result = indexer.IndexResult()
        indexer.index_file(conn, REPO, repo, "pkg/b.py", result)
        conn.commit()
        target = repo / "pkg" / "b.py"
        os.utime(target, ns=(1, 1))  # same bytes, different mtime

        plan = indexer._plan_file(GuardedConnection(conn), REPO, repo, "pkg/b.py")
        assert plan.action == "refresh"

    def test_force_replans_everything(self, conn, repo) -> None:
        indexer.index_repo(conn, REPO, repo)
        plan = indexer._plan_file(conn, REPO, repo, "pkg/b.py", force=True)
        assert plan.action == "index"


class TestApplyBatching:
    def test_yields_between_batches_of_written_files(
        self, conn, repo, monkeypatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(indexer, "COMMIT_EVERY_FILES", 2)
        monkeypatch.setattr(indexer.time, "sleep", sleeps.append)

        result = indexer.index_repo(conn, REPO, repo)
        # 5 files at batch size 2 -> two mid-pass commits, each with a yield.
        assert result.processed == 5
        assert len(sleeps) == 2

    def test_a_warm_noop_pass_never_sleeps(self, conn, repo, monkeypatch) -> None:
        indexer.index_repo(conn, REPO, repo)
        sleeps: list[float] = []
        monkeypatch.setattr(indexer, "COMMIT_EVERY_FILES", 1)
        monkeypatch.setattr(indexer.time, "sleep", sleeps.append)

        result = indexer.index_repo(conn, REPO, repo)
        # Skips write nothing, so they must not count toward batches: a no-op
        # re-index sleeping its way through empty transactions would turn the
        # cheapest pass into the slowest.
        assert result.skipped_unchanged == 5
        assert sleeps == []

    def test_two_phase_and_per_file_agree(self, conn, repo, tmp_path) -> None:
        # index_repo (plan all, then apply) and index_file (plan+apply each)
        # must produce identical stores.
        indexer.index_repo(conn, REPO, repo)

        other = sqlite3.connect(tmp_path / "other.db")
        ensure_schema(other)
        result = indexer.IndexResult()
        for rel in indexer.walk_repo(repo):
            indexer.index_file(other, REPO, repo, rel, result)
        other.commit()

        def graph(c):
            nodes = c.execute(
                "SELECT kind, qualified_name, file_path, lang, attributes "
                "FROM code_nodes ORDER BY kind, qualified_name"
            ).fetchall()
            edges = c.execute(
                "SELECT s.qualified_name, d.qualified_name FROM code_edges e "
                "JOIN code_nodes s ON s.node_id = e.src_id "
                "JOIN code_nodes d ON d.node_id = e.dst_id "
                "WHERE e.invalid_at IS NULL ORDER BY 1, 2"
            ).fetchall()
            return nodes, edges

        assert graph(conn) == graph(other)
        other.close()
