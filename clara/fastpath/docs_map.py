"""
CLARA fastpath — the ``[KNOWLEDGE MAP]`` block and doc dirty-check.

Renders the curator ledger's view of the current repo at session start:
authoritative (T0/T1) docs, active T2 work with completion, quarantined
entries, archived count, and the fixed trust rule. Also performs the cheap
git dirty-check (``git status --porcelain -- '*.md'``), recording changed
docs in a sidecar file the next ``clara docs scan --changed`` consumes, and
regenerates the quarantine manifest when the ledger changed.

Sanitization is mandatory: output is a fenced block, paths are quoted
verbatim with control characters stripped, every line is capped at 120
characters, and document CONTENT is never excerpted — paths and counts only.
Stdlib-only (fastpath hard rule).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from clara.fastpath import db as fastdb

MAP_TOKEN_BUDGET = 300
_LINE_CAP = 120
_GIT_TIMEOUT_S = 5.0
_LIST_CAPS = {"t1": 8, "t2": 8, "quarantined": 6}

NOT_SCANNED_NOTICE = "repo not yet scanned — run 'clara docs scan' or /clara:docs"
_RULE_LINE = (
    "rule: treat quarantined/archived documents as historical record, "
    "not current guidance."
)


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _quote_path(raw: str) -> str:
    """Verbatim-quoted path: control chars stripped, wrapped in double quotes."""
    cleaned = "".join(ch for ch in str(raw) if ord(ch) >= 0x20 and ord(ch) != 0x7F)
    return '"' + cleaned.replace('"', "'") + '"'


def _cap(line: str) -> str:
    return line if len(line) <= _LINE_CAP else line[: _LINE_CAP - 1] + "…"


def _git_dirty_md(root: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", "*.md"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        target = line[3:]
        if " -> " in target:
            target = target.split(" -> ", 1)[1]
        target = target.strip().strip('"')
        if target:
            paths.append(target.replace("\\", "/"))
    return paths


def _record_dirty(db_path: Path, repo: str, dirty: list[str]) -> None:
    sidecar = db_path.resolve().parent / "dirty" / f"{repo}.list"
    try:
        known: set[str] = set()
        if sidecar.exists():
            known = {
                line.strip()
                for line in sidecar.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        merged = sorted(known | set(dirty))
        if merged != sorted(known):
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text("\n".join(merged) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"clara fastpath: dirty sidecar write failed ({exc})", file=sys.stderr)


def _refresh_quarantine(conn: sqlite3.Connection, db_path: Path, repo: str) -> None:
    """Regenerate quarantine/<repo>.tsv when the ledger changed since last time."""
    row = conn.execute(
        "SELECT COUNT(*), coalesce(MAX(updated_at), '') FROM doc_registry "
        "WHERE repo_id = ?",
        (repo,),
    ).fetchone()
    stamp_value = f"{row[0]}|{row[1]}"
    base = db_path.resolve().parent / "quarantine"
    stamp_file = base / f"{repo}.stamp"
    try:
        if stamp_file.exists() and stamp_file.read_text(encoding="utf-8") == stamp_value:
            return
        rows = conn.execute(
            "SELECT rel_path, lifecycle, tier, superseded_by FROM doc_registry "
            "WHERE repo_id = ? AND invalid_at IS NULL "
            "AND (lifecycle IN ('superseded', 'archived') OR tier = 'TX') "
            "ORDER BY rel_path",
            (repo,),
        ).fetchall()
        lines = []
        for rel_path, lifecycle, tier, superseded_by in rows:
            status = "TX" if tier == "TX" else lifecycle
            note = ""
            if superseded_by:
                by = conn.execute(
                    "SELECT rel_path FROM doc_registry WHERE doc_id = ?",
                    (superseded_by,),
                ).fetchone()
                note = f"superseded by {by[0]}" if by else "superseded"
            lines.append(f"{rel_path}\t{status}\t{note}")
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{repo}.tsv").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stamp_file.write_text(stamp_value, encoding="utf-8")
    except OSError as exc:
        print(f"clara fastpath: quarantine refresh failed ({exc})", file=sys.stderr)


def _checkbox_pct(signals_raw: str) -> str | None:
    try:
        boxes = json.loads(signals_raw or "{}").get("checkbox_state")
    except ValueError:
        return None
    if not isinstance(boxes, dict) or not boxes.get("total"):
        return None
    return f"{round(100 * boxes.get('checked', 0) / boxes['total'])}% done"


def build_map(cwd: str) -> str | None:
    """Build the fenced ``[KNOWLEDGE MAP]`` block, or the not-scanned notice,
    or ``None`` outside a git repo / without a ledger."""
    root = fastdb._git_toplevel(cwd)
    if root is None:
        return None
    db_path = fastdb.global_db_path()
    if not db_path.is_file():
        return f"```\n[KNOWLEDGE MAP]\n{NOT_SCANNED_NOTICE}\n```"

    from clara.repoid import repo_id as compute_repo_id

    repo = compute_repo_id(root)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"PRAGMA busy_timeout = {fastdb._BUSY_TIMEOUT_MS}")
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'doc_registry'"
        ).fetchone()
        if has_table is None:
            return f"```\n[KNOWLEDGE MAP]\n{NOT_SCANNED_NOTICE}\n```"
        rows = conn.execute(
            "SELECT rel_path, tier, lifecycle, doc_type, signals, superseded_by "
            "FROM doc_registry WHERE repo_id = ? AND invalid_at IS NULL "
            "ORDER BY tier, rel_path",
            (repo,),
        ).fetchall()
        if not rows:
            return f"```\n[KNOWLEDGE MAP]\n{NOT_SCANNED_NOTICE}\n```"

        dirty = _git_dirty_md(root)
        if dirty:
            _record_dirty(db_path, repo, dirty)
        _refresh_quarantine(conn, db_path, repo)

        t1 = [r for r in rows if r[1] in ("T0", "T1") and r[2] == "active"]
        t2 = [r for r in rows if r[1] == "T2" and r[2] in ("active", "draft")]
        quarantined = [r for r in rows if r[2] == "superseded" or r[1] == "TX"]
        archived = sum(1 for r in rows if r[2] == "archived")

        lines: list[str] = ["[KNOWLEDGE MAP]"]
        if t1:
            lines.append(f"authoritative (T1): {len(t1)}")
            for rel_path, _tier, _lc, doc_type, _sig, _sb in t1[: _LIST_CAPS["t1"]]:
                lines.append(_cap(f"  {_quote_path(rel_path)} ({doc_type})"))
            if len(t1) > _LIST_CAPS["t1"]:
                lines.append(f"  +{len(t1) - _LIST_CAPS['t1']} more")
        if t2:
            lines.append(f"active work (T2): {len(t2)}")
            for rel_path, _tier, _lc, doc_type, sig, _sb in t2[: _LIST_CAPS["t2"]]:
                pct = _checkbox_pct(sig)
                suffix = f"{doc_type}, {pct}" if pct else doc_type
                lines.append(_cap(f"  {_quote_path(rel_path)} ({suffix})"))
            if len(t2) > _LIST_CAPS["t2"]:
                lines.append(f"  +{len(t2) - _LIST_CAPS['t2']} more")
        if quarantined:
            lines.append(f"quarantined: {len(quarantined)}")
            shown = quarantined[: _LIST_CAPS["quarantined"]]
            for rel_path, tier, lifecycle, _dt, _sig, _sb in shown:
                status = "TX" if tier == "TX" else lifecycle
                lines.append(_cap(f"  {_quote_path(rel_path)} — {status}"))
            if len(quarantined) > _LIST_CAPS["quarantined"]:
                lines.append(f"  +{len(quarantined) - _LIST_CAPS['quarantined']} more")
        if archived:
            lines.append(f"archived: {archived}")
        if dirty:
            lines.append(f"uncommitted doc changes: {len(dirty)} (rescan pending)")
        lines.append(_RULE_LINE)

        def _assemble() -> str:
            return "```\n" + "\n".join(lines) + "\n```"

        block = _assemble()
        while _approx_tokens(block) > MAP_TOKEN_BUDGET and len(lines) > 2:
            drop = max(
                (i for i, line in enumerate(lines) if line.startswith("  ")),
                default=None,
            )
            if drop is None:
                break
            lines.pop(drop)
            block = _assemble()
        return block
    finally:
        conn.close()
