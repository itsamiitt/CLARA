"""Change journal and index-state gate (docs/plans/memory-systems-plan.md §3.1-3.2).

The journal is the indexing pipeline's queue. Two properties matter more than
its API: a batch claimed by a worker that then dies must come back, and two
workers must never claim the same row. Both are tested against real
concurrency rather than by reading the SQL.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from clara.db.migrations import ensure_schema
from clara.index import journal, state

REPO = "repo-abc123"


@pytest.fixture
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "clara.db")
    ensure_schema(db)
    db.execute("PRAGMA busy_timeout = 30000")
    yield db
    db.close()


class TestEnqueue:
    def test_returns_increasing_sequence_numbers(self, conn):
        first = journal.enqueue(conn, REPO, change="added", path="a.py")
        second = journal.enqueue(conn, REPO, change="modified", path="b.py")
        assert second > first

    def test_repo_level_changes_need_no_path(self, conn):
        journal.enqueue(conn, REPO, change="git")
        journal.enqueue(conn, REPO, change="manifest")
        assert journal.pending_count(conn, REPO) == 2

    def test_file_changes_require_a_path(self, conn):
        with pytest.raises(ValueError, match="requires a path"):
            journal.enqueue(conn, REPO, change="modified")

    def test_unknown_change_is_rejected(self, conn):
        with pytest.raises(ValueError, match="unknown change"):
            journal.enqueue(conn, REPO, change="exploded", path="a.py")

    def test_rename_carries_the_old_path(self, conn):
        journal.enqueue(conn, REPO, change="renamed", path="new.py", old_path="old.py")
        (entry,) = journal.claim_batch(conn, REPO, worker="w1")
        assert entry.old_path == "old.py"
        assert entry.path == "new.py"


class TestClaiming:
    def test_claims_in_order_and_respects_the_limit(self, conn):
        for i in range(5):
            journal.enqueue(conn, REPO, change="added", path=f"{i}.py")
        batch = journal.claim_batch(conn, REPO, worker="w1", limit=3)
        assert [e.path for e in batch] == ["0.py", "1.py", "2.py"]
        assert journal.pending_count(conn, REPO) == 2

    def test_a_claimed_entry_is_not_handed_out_twice(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        assert len(journal.claim_batch(conn, REPO, worker="w1")) == 1
        assert journal.claim_batch(conn, REPO, worker="w2") == []

    def test_other_repos_are_untouched(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        journal.enqueue(conn, "other-repo", change="added", path="b.py")
        batch = journal.claim_batch(conn, REPO, worker="w1")
        assert [e.repo_id for e in batch] == [REPO]
        assert journal.pending_count(conn, "other-repo") == 1

    def test_completing_removes_the_rows(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        batch = journal.claim_batch(conn, REPO, worker="w1")
        assert journal.complete(conn, [e.seq for e in batch]) == 1
        assert conn.execute("SELECT count(*) FROM change_journal").fetchone()[0] == 0

    def test_zero_limit_claims_nothing(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        assert journal.claim_batch(conn, REPO, worker="w1", limit=0) == []
        assert journal.pending_count(conn, REPO) == 1


class TestCrashRecovery:
    """A worker that dies mid-batch must not take the work with it."""

    def test_a_stale_claim_is_released(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        journal.claim_batch(conn, REPO, worker="dead-worker")
        assert journal.pending_count(conn, REPO) == 0, "precondition: it is claimed"

        # Backdate the claim past the staleness window.
        conn.execute(
            "UPDATE change_journal SET claimed_at = datetime('now', '-1 day')"
        )
        conn.commit()

        assert journal.release_stale_claims(conn) == 1
        assert journal.pending_count(conn, REPO) == 1

    def test_a_fresh_claim_is_left_alone(self, conn):
        """Releasing eagerly would hand live work to a second worker."""
        journal.enqueue(conn, REPO, change="added", path="a.py")
        journal.claim_batch(conn, REPO, worker="w1")
        assert journal.release_stale_claims(conn) == 0
        assert journal.pending_count(conn, REPO) == 0

    def test_claiming_recovers_stale_work_automatically(self, conn):
        journal.enqueue(conn, REPO, change="added", path="a.py")
        journal.claim_batch(conn, REPO, worker="dead-worker")
        conn.execute(
            "UPDATE change_journal SET claimed_at = datetime('now', '-1 day')"
        )
        conn.commit()

        recovered = journal.claim_batch(conn, REPO, worker="live-worker")
        assert [e.path for e in recovered] == ["a.py"], (
            "a new worker must pick up work abandoned by a dead one"
        )


class TestConcurrentWorkers:
    def test_no_entry_is_claimed_twice_across_processes(self, tmp_path):
        """Real threads on separate connections, not a mocked race."""
        db_path = tmp_path / "clara.db"
        setup = sqlite3.connect(db_path)
        ensure_schema(setup)
        for i in range(60):
            journal.enqueue(setup, REPO, change="added", path=f"{i}.py")
        setup.commit()
        setup.close()

        claimed: list[list[int]] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(name: str) -> None:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA busy_timeout = 30000")
            try:
                got: list[int] = []
                while True:
                    batch = journal.claim_batch(conn, REPO, worker=name, limit=7)
                    if not batch:
                        break
                    got.extend(e.seq for e in batch)
                    journal.complete(conn, [e.seq for e in batch])
                with lock:
                    claimed.append(got)
            except BaseException as exc:  # noqa: BLE001 — surfaced below
                with lock:
                    errors.append(exc)
            finally:
                conn.close()

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert not errors, errors
        everything = [seq for got in claimed for seq in got]
        assert len(everything) == 60, f"lost or duplicated work: {len(everything)}"
        assert len(set(everything)) == 60, "an entry was claimed by two workers"


class TestIndexStateGate:
    def test_unchanged_content_is_skipped(self, conn, tmp_path):
        source = tmp_path / "a.py"
        source.write_text("import os\n", encoding="utf-8")
        digest = state.content_hash(source)

        assert not state.is_unchanged(
            conn, REPO, path="a.py", kind="file", current_hash=digest
        ), "never indexed yet"

        state.record_indexed(
            conn, REPO, path="a.py", kind="file", content_hash=digest, lang="python"
        )
        assert state.is_unchanged(
            conn, REPO, path="a.py", kind="file", current_hash=digest
        )

    def test_changed_content_is_not_skipped(self, conn, tmp_path):
        source = tmp_path / "a.py"
        source.write_text("import os\n", encoding="utf-8")
        state.record_indexed(
            conn, REPO, path="a.py", kind="file",
            content_hash=state.content_hash(source),
        )
        source.write_text("import sys\n", encoding="utf-8")
        assert not state.is_unchanged(
            conn, REPO, path="a.py", kind="file",
            current_hash=state.content_hash(source),
        )

    def test_an_unreadable_file_is_never_unchanged(self, conn):
        """A vanished file must be re-processed, not silently skipped."""
        assert state.content_hash(Path("does-not-exist.py")) is None
        state.record_indexed(
            conn, REPO, path="gone.py", kind="file", content_hash="whatever"
        )
        assert not state.is_unchanged(
            conn, REPO, path="gone.py", kind="file", current_hash=None
        )

    def test_recording_twice_updates_rather_than_duplicates(self, conn):
        for digest in ("aaa", "bbb"):
            state.record_indexed(
                conn, REPO, path="a.py", kind="file", content_hash=digest
            )
        rows = conn.execute(
            "SELECT content_hash FROM index_state WHERE repo_id = ? AND path = ?",
            (REPO, "a.py"),
        ).fetchall()
        assert rows == [("bbb",)]

    def test_forget_path_clears_every_kind(self, conn):
        for kind in ("file", "graph_shard"):
            state.record_indexed(
                conn, REPO, path="a.py", kind=kind, content_hash="x"
            )
        assert state.forget_path(conn, REPO, path="a.py") == 2
        assert state.current_generation(conn, REPO) == 0

    def test_hash_is_stable_and_content_sensitive(self, tmp_path):
        one = tmp_path / "one.py"
        two = tmp_path / "two.py"
        one.write_text("import os\n", encoding="utf-8")
        two.write_text("import os\n", encoding="utf-8")
        assert state.content_hash(one) == state.content_hash(two)
        two.write_text("import os  # changed\n", encoding="utf-8")
        assert state.content_hash(one) != state.content_hash(two)
