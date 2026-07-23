"""
Deterministic synthetic store generator for CLARA benchmarks.

Builds a store with a realistic type mix, tags, and timestamps spread over
two years, via raw executemany (fast, and it exercises the same v4/v5 schema
+ FTS triggers real writes hit). Seeded — identical stores every run.

    python -m benchmarks.synth /tmp/bench.db --rows 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from clara.db.migrations import ensure_schema

_SUBJECTS = [
    "user", "team", "api", "frontend", "backend", "database", "pipeline",
    "auth-service", "billing", "scheduler", "cache", "gateway",
]
_RELATIONS = [
    "prefers", "uses", "deployed_to", "depends_on", "owns", "migrated_to",
    "configured_with", "monitors", "replaced",
]
_OBJECTS = [
    "postgresql", "sqlite", "kubernetes", "pnpm", "typescript", "fastapi",
    "redis", "github actions", "fly.io", "tailwind", "pytest", "uv",
    "dark mode", "trunk-based development", "conventional commits",
]
_TAGS = ["infra", "frontend", "backend", "process", "tooling", "decision"]
_TYPE_MIX = [("belief", 0.55), ("event", 0.25), ("world_model", 0.12), ("skill", 0.08)]


def _pick_type(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for name, weight in _TYPE_MIX:
        cumulative += weight
        if roll <= cumulative:
            return name
    return "belief"


def build_store(db_path: str, rows: int, *, seed: int = 42) -> None:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    batch: list[tuple] = []
    for i in range(rows):
        mem_type = _pick_type(rng)
        subject = rng.choice(_SUBJECTS)
        relation = rng.choice(_RELATIONS)
        obj = f"{rng.choice(_OBJECTS)} #{i % 97}"
        if mem_type == "belief":
            content = {"subject": subject, "relation": relation, "object": obj}
            decay = 0.02
        elif mem_type == "event":
            content = {"subject": subject, "relation": "released",
                       "object": obj, "event_type": "released",
                       "event_status": "completed"}
            decay = 0.0
        elif mem_type == "world_model":
            content = {"entity_type": "service", "name": f"{subject}-{i}",
                       "subject": f"{subject}-{i}", "relation": "is_a",
                       "object": "service", "properties": {"port": 8000 + i % 1000}}
            decay = 0.005
        else:
            content = {"name": f"procedure-{i}", "subject": "skill",
                       "relation": "can_do", "object": f"procedure-{i}",
                       "trigger_conditions": ["on demand"], "steps": ["do it"]}
            decay = 0.02
        age_days = rng.uniform(0, 730)
        ts = (now - timedelta(days=age_days)).isoformat(sep=" ")
        meta = {"tags": rng.sample(_TAGS, k=rng.randint(0, 2))}
        batch.append((
            uuid.uuid4().hex,
            None,
            mem_type,
            json.dumps(content, ensure_ascii=False),
            round(rng.uniform(0.3, 1.0), 3),
            "active",
            decay,
            ts,
            ts,
            json.dumps(meta, ensure_ascii=False),
        ))
        if len(batch) >= 1000:
            _flush(conn, batch)
    _flush(conn, batch)
    conn.close()


def _flush(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    if not batch:
        return
    conn.executemany(
        "INSERT INTO memories (memory_id, user_id, memory_type, content, "
        "confidence, status, decay_rate, created_at, updated_at, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        batch,
    )
    conn.commit()
    batch.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_store(args.db_path, args.rows, seed=args.seed)
    print(f"built {args.rows}-row store at {args.db_path}")


if __name__ == "__main__":
    main()
