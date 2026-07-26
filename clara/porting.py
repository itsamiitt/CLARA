"""
CLARA — JSONL export/import (`clara export` / `clara import`).

Format ``clara-export`` v1: line 1 is a header record, then one JSON object
per row. Graph rows are optional (the graph is rebuildable from memories via
``clara graph rebuild``); doc-ledger rows optional likewise.

    {"kind": "header", "format": 1, "schema_version": 5, ...}
    {"kind": "memory", "memory_id": "...", "memory_type": "belief", ...}

Raw stdlib ``sqlite3`` on both sides: import writes through plain SQL so the
FTS triggers index the rows and ORM defaults cannot clobber the original
timestamps. Stdlib-only.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from clara.db.migrations import SCHEMA_VERSION, ensure_schema, get_version

FORMAT_VERSION = 1


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp allowing the "T" or space separator.

    Returns ``None`` for anything unparseable so callers can fall back to a
    safe default (keep the existing row) rather than trust a lexical compare.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace(" ", "T", 1))
        except ValueError:
            return None

# Canonical column order for the memories table (see _MEMORIES_DDL in
# clara/db/migrations.py). The SQL column is named "metadata"; the ORM
# attribute metadata_ is an implementation detail that must not leak into
# the interchange format.
_MEMORY_COLUMNS = (
    "memory_id",
    "user_id",
    "memory_type",
    "content",
    "confidence",
    "status",
    "decay_rate",
    "created_at",
    "updated_at",
    "metadata",
)

_GRAPH_TABLES = ("graph_nodes", "graph_aliases", "graph_edges")
_DOCS_TABLES = ("doc_registry", "doc_refs", "doc_attestations")

IMPORT_BATCH = 500


def _json_or_none(raw: Any) -> Any:
    if raw is None or not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def export_records(
    db_path: str,
    *,
    types: list[str] | None = None,
    status: str = "all",
    since: str | None = None,
    include_graph: bool = False,
    include_docs: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield header + row records for the store at *db_path*."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if status != "all":
            clauses.append("status = ?")
            params.append(status)
        if types:
            clauses.append(
                "memory_type IN (%s)" % ", ".join("?" for _ in types)
            )
            params.extend(types)
        if since:
            clauses.append("updated_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        count = conn.execute(
            f"SELECT count(*) FROM memories {where}", params
        ).fetchone()[0]
        yield {
            "kind": "header",
            "format": FORMAT_VERSION,
            "schema_version": get_version(conn),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source": str(Path(db_path).resolve()),
            "counts": {"memories": count},
        }

        cols = ", ".join(f'"{c}"' for c in _MEMORY_COLUMNS)
        for row in conn.execute(
            f"SELECT {cols} FROM memories {where} ORDER BY created_at", params
        ):
            record: dict[str, Any] = {"kind": "memory"}
            for col in _MEMORY_COLUMNS:
                value = row[col]
                if col in ("content", "metadata"):
                    value = _json_or_none(value)
                record[col] = value
            yield record

        for enabled, tables, kind_prefix in (
            (include_graph, _GRAPH_TABLES, "graph"),
            (include_docs, _DOCS_TABLES, "docs"),
        ):
            if not enabled:
                continue
            for table in tables:
                try:
                    rows = conn.execute(f"SELECT * FROM {table}")
                except sqlite3.OperationalError:
                    continue
                for row in rows:
                    payload = {key: row[key] for key in row.keys()}
                    yield {"kind": f"{kind_prefix}:{table}", **payload}
    finally:
        conn.close()


def write_export(records: Iterable[dict[str, Any]], out: TextIO) -> int:
    written = 0
    for record in records:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    return written


class ImportError_(ValueError):
    """A structurally invalid or incompatible export file."""


def import_records(
    db_path: str,
    lines: Iterable[str],
    *,
    on_conflict: str = "skip",
    dry_run: bool = False,
    allow_secrets: bool = False,
) -> dict[str, int]:
    """Import a clara-export stream into the store at *db_path*.

    Conflict policy for an existing ``memory_id``: ``skip`` (default),
    ``newest`` (keep the row with the later ``updated_at``), ``force``
    (imported row wins). Rows without an id match are additionally checked
    against active belief identity (type + subject/relation/object) and
    skipped as ``duplicate_content`` when present.
    """
    if on_conflict not in ("skip", "newest", "force"):
        raise ValueError("on_conflict must be skip | newest | force")

    from clara import security

    iterator = iter(lines)
    header_line = next(iterator, None)
    if header_line is None:
        raise ImportError_("empty import file")
    try:
        header = json.loads(header_line)
    except ValueError as exc:
        raise ImportError_(f"line 1 is not JSON: {exc}") from exc
    if header.get("kind") != "header":
        raise ImportError_("line 1 must be the header record")
    if int(header.get("format", 0)) > FORMAT_VERSION:
        raise ImportError_(
            f"export format {header.get('format')} is newer than supported {FORMAT_VERSION}"
        )
    if int(header.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ImportError_(
            f"export schema v{header.get('schema_version')} is newer than this "
            f"CLARA supports (v{SCHEMA_VERSION}) — upgrade clara-memory first"
        )

    stats = {
        "imported": 0,
        "skipped_id": 0,
        "skipped_content": 0,
        "conflicts_updated": 0,
        "skipped_secret": 0,
        # Non-memory rows the export emits (graph:*, docs:*). Import does not
        # replay them yet; the CLI surfaces this count so an --include-docs
        # export doesn't silently lose attestations (which are unrebuildable
        # judgments, unlike the graph).
        "other_kinds": 0,
        # Body lines that were not valid JSON — surfaced so a truncated file
        # is not mistaken for a clean import.
        "skipped_malformed": 0,
    }
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        batch: list[dict[str, Any]] = []

        def flush() -> None:
            if not batch or dry_run:
                batch.clear()
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                for rec in batch:
                    cols = ", ".join(f'"{c}"' for c in _MEMORY_COLUMNS)
                    placeholders = ", ".join("?" for _ in _MEMORY_COLUMNS)
                    values = []
                    for col in _MEMORY_COLUMNS:
                        value = rec.get(col)
                        if col in ("content", "metadata") and isinstance(
                            value, (dict, list)
                        ):
                            value = json.dumps(value, ensure_ascii=False)
                        values.append(value)
                    conn.execute(
                        f"INSERT OR REPLACE INTO memories ({cols}) VALUES ({placeholders})",
                        values,
                    )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            batch.clear()

        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                stats["skipped_malformed"] += 1
                continue
            if record.get("kind") != "memory":
                stats["other_kinds"] += 1
                continue
            memory_id = str(record.get("memory_id") or "").replace("-", "")
            if not memory_id:
                continue
            record["memory_id"] = memory_id

            if not allow_secrets and security.secret_policy() != "off":
                # Scan metadata too (tags/provenance), matching the live write
                # path — a credential in a tag must not slip through on import.
                blob = json.dumps(
                    {
                        "content": record.get("content", {}),
                        "metadata": record.get("metadata", {}),
                    },
                    ensure_ascii=False,
                )
                if security.find_secret(blob) is not None:
                    stats["skipped_secret"] += 1
                    continue

            existing = conn.execute(
                "SELECT updated_at FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if existing is not None:
                if on_conflict == "skip":
                    stats["skipped_id"] += 1
                    continue
                if on_conflict == "newest":
                    theirs = _parse_ts(record.get("updated_at"))
                    ours = _parse_ts(existing[0])
                    # Parse before comparing: timestamps in the wild mix the
                    # "T" and space separators, and a lexical compare makes a
                    # newer space-separated time lose to an older "T" one
                    # (space 0x20 < "T" 0x54). Unparseable → keep existing.
                    if theirs is None or ours is None or theirs <= ours:
                        stats["skipped_id"] += 1
                        continue
                stats["conflicts_updated"] += 1
                batch.append(record)
                if len(batch) >= IMPORT_BATCH:
                    flush()
                continue

            content = record.get("content")
            if (
                record.get("memory_type") == "belief"
                and isinstance(content, dict)
                and conn.execute(
                    "SELECT 1 FROM memories WHERE memory_type = 'belief' "
                    "AND status = 'active' "
                    "AND json_extract(content, '$.subject') = ? "
                    "AND json_extract(content, '$.relation') = ? "
                    "AND json_extract(content, '$.object') = ? LIMIT 1",
                    (
                        content.get("subject"),
                        content.get("relation"),
                        content.get("object"),
                    ),
                ).fetchone()
                is not None
            ):
                stats["skipped_content"] += 1
                continue

            stats["imported"] += 1
            batch.append(record)
            if len(batch) >= IMPORT_BATCH:
                flush()
        flush()
    finally:
        conn.close()
    return stats
