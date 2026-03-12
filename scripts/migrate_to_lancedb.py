from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import lancedb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from clara.retrieval.engine import LANCE_SCHEMA, LANCE_TABLE_NAME, RetrievalEngine


def _ensure_lance_table(lance_path: str):
    Path(lance_path).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_path)
    table_names = db.list_tables()
    tables = set(getattr(table_names, "tables", table_names) or [])
    if LANCE_TABLE_NAME not in tables:
        return db.create_table(LANCE_TABLE_NAME, schema=LANCE_SCHEMA)
    return db.open_table(LANCE_TABLE_NAME)


async def _load_rows(db_url: str) -> list[dict]:
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT
                    memory_id::text AS memory_id,
                    user_id,
                    memory_type::text AS memory_type,
                    status::text AS status,
                    embedding
                FROM memories
                WHERE embedding IS NOT NULL
            """))
            return list(result.mappings().all())
    finally:
        await engine.dispose()


def _write_batch(table, rows: list[dict]) -> None:
    if not rows:
        return
    (
        table.merge_insert("memory_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(rows)
    )


async def migrate(*, db_url: str, lance_path: str, batch_size: int) -> tuple[int, int]:
    rows = await _load_rows(db_url)
    table = _ensure_lance_table(lance_path)

    migrated = 0
    skipped = 0
    batch: list[dict] = []
    for row in rows:
        vector = RetrievalEngine._embedding_to_list(row.get("embedding"))
        if vector is None:
            skipped += 1
            continue
        batch.append({
            "memory_id": str(row["memory_id"]),
            "vector": [float(value) for value in vector],
            "user_id": str(row.get("user_id") or ""),
            "memory_type": str(row["memory_type"]),
            "status": str(row["status"]),
        })
        if len(batch) >= batch_size:
            _write_batch(table, batch)
            migrated += len(batch)
            batch.clear()

    if batch:
        _write_batch(table, batch)
        migrated += len(batch)

    return migrated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate PostgreSQL pgvector embeddings into embedded LanceDB.",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="Existing PostgreSQL SQLAlchemy URL, e.g. postgresql+asyncpg://...",
    )
    parser.add_argument(
        "--lance-path",
        default="./clara_vectors",
        help="Destination LanceDB directory (default: ./clara_vectors).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows to merge per LanceDB batch (default: 500).",
    )
    args = parser.parse_args()

    migrated, skipped = asyncio.run(
        migrate(
            db_url=args.db_url,
            lance_path=args.lance_path,
            batch_size=max(1, args.batch_size),
        )
    )
    print(f"Migrated {migrated} embeddings into {args.lance_path!r}.")
    if skipped:
        print(f"Skipped {skipped} row(s) with missing or unreadable embeddings.")


if __name__ == "__main__":
    main()
