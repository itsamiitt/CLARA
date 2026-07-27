"""
Stale per-session sidecar files are swept by the maintenance pass.

Every session leaves files behind — the cwd hint, the Stop-hook debounce
flag, the prompt-recall ledger, the read-annotation flag directory — and
before this sweep, verified by reading every reference, nothing deleted any
of them, ever. Preventive: no observed damage, just growth with no owner.

The boundary that matters is the age cutoff. A sweep that deletes a *live*
session's recall ledger resurrects already-shown memories mid-session, so
young files must survive untouched.
"""

from __future__ import annotations

import os
import time

from clara.maintenance import SESSION_FILE_MAX_AGE_S, sweep_session_files

OLD = time.time() - SESSION_FILE_MAX_AGE_S - 3600
FRESH = time.time() - 60


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


class TestSweep:
    def test_old_files_go_young_files_stay(self, tmp_path) -> None:
        _touch(tmp_path / "session-flags" / "dead.done", OLD)
        _touch(tmp_path / "session-flags" / "dead.recalled", OLD)
        _touch(tmp_path / "session-cwd" / "dead", OLD)
        _touch(tmp_path / "session-flags" / "live.recalled", FRESH)
        _touch(tmp_path / "session-cwd" / "live", FRESH)

        removed = sweep_session_files(tmp_path)

        assert removed == 3
        assert not (tmp_path / "session-flags" / "dead.done").exists()
        assert not (tmp_path / "session-cwd" / "dead").exists()
        assert (tmp_path / "session-flags" / "live.recalled").exists()
        assert (tmp_path / "session-cwd" / "live").exists()

    def test_read_annotation_directories_are_swept_recursively(
        self, tmp_path
    ) -> None:
        flag_dir = tmp_path / "session-flags" / "dead.read"
        _touch(flag_dir / "doc_one", OLD)
        _touch(flag_dir / "doc_two", OLD)
        os.utime(flag_dir, (OLD, OLD))

        removed = sweep_session_files(tmp_path)

        assert removed == 3  # two children + the directory
        assert not flag_dir.exists()

    def test_a_live_read_directory_survives(self, tmp_path) -> None:
        flag_dir = tmp_path / "session-flags" / "live.read"
        _touch(flag_dir / "doc", FRESH)
        os.utime(flag_dir, (FRESH, FRESH))

        assert sweep_session_files(tmp_path) == 0
        assert (flag_dir / "doc").exists()

    def test_missing_directories_are_fine(self, tmp_path) -> None:
        assert sweep_session_files(tmp_path / "nothing-here") == 0

    def test_the_store_itself_is_never_touched(self, tmp_path) -> None:
        # The sweep must only ever look inside the two session dirs.
        store = tmp_path / "clara.db"
        _touch(store, OLD)
        _touch(tmp_path / "backups" / "old-backup.db", OLD)

        sweep_session_files(tmp_path)

        assert store.exists()
        assert (tmp_path / "backups" / "old-backup.db").exists()
