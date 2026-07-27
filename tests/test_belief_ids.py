"""Canonical belief ids (audit P4).

``graph_edges.belief_id`` was being written in two different spellings by two
different code paths:

    live projection -> ORM record -> memory_id is a uuid.UUID -> str() dashed
    graph rebuild   -> raw SQL row -> memory_id is str          -> str() dashless

``memories.memory_id`` is always the dashless form, so joining the two required
wrapping both sides in ``replace(CAST(...))``. That had two costs. It made the
join unindexable, and it hid a real bug: the rebuild's duplicate check compared
a dashless id against a set of dashed ones, never matched, and re-created every
edge on every rebuild.

These tests pin the canonical form at both writers, the absence of duplicates,
and the migration that repairs stores written before the fix.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from clara.core.ids import canonical_id
from clara.db.migrations import ensure_schema
from clara.integrations.local_memory import LocalMemory


class TestCanonicalId:
    def test_uuid_object_uses_hex(self):
        value = uuid.uuid4()
        assert canonical_id(value) == value.hex
        assert "-" not in canonical_id(value)

    def test_dashed_string_is_stripped(self):
        value = uuid.uuid4()
        assert canonical_id(str(value)) == value.hex

    def test_dashless_string_is_unchanged(self):
        value = uuid.uuid4()
        assert canonical_id(value.hex) == value.hex

    def test_matches_how_sqlite_stores_the_column(self, tmp_path):
        """The whole point: canonical_id must equal the stored representation."""
        db = str(tmp_path / "c.db")
        conn = sqlite3.connect(db)
        ensure_schema(conn)
        value = uuid.uuid4()
        conn.execute(
            "INSERT INTO memories (memory_id, memory_type, content, confidence,"
            " status, decay_rate, created_at, updated_at) VALUES"
            " (?,'belief','{}',0.9,'active',0.02,'2026-01-01','2026-01-01')",
            (value.hex,),
        )
        conn.commit()
        stored = conn.execute("SELECT memory_id FROM memories").fetchone()[0]
        assert canonical_id(value) == stored


@pytest.mark.asyncio
class TestWritersAgree:
    async def _store(self, tmp_path) -> str:
        db = str(tmp_path / "clara.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(subject="user", relation="uses", object="Rust")
            await memory.save(subject="api", relation="runs_on", object="fly.io")
            await memory.save(subject="team", relation="owns", object="billing")
            await memory.graph_rebuild()
        finally:
            await memory.close()
        return db

    async def test_no_dashed_belief_ids_are_written(self, tmp_path):
        conn = sqlite3.connect(await self._store(tmp_path))
        dashed = conn.execute(
            "SELECT count(*) FROM graph_edges WHERE belief_id LIKE '%-%'"
        ).fetchone()[0]
        assert dashed == 0

    async def test_plain_equality_join_finds_everything(self, tmp_path):
        """The normalised join must not find rows the plain one misses.

        If it does, some writer is still emitting a non-canonical id and the
        maintenance queries -- which now use plain equality -- would silently
        skip those edges.
        """
        conn = sqlite3.connect(await self._store(tmp_path))
        plain = conn.execute(
            "SELECT count(*) FROM memories m JOIN graph_edges e "
            "ON m.memory_id = e.belief_id"
        ).fetchone()[0]
        normalised = conn.execute(
            "SELECT count(*) FROM memories m JOIN graph_edges e ON "
            "replace(CAST(m.memory_id AS TEXT),'-','') = replace(e.belief_id,'-','')"
        ).fetchone()[0]
        assert plain == normalised
        assert plain > 0

    async def test_rebuild_does_not_duplicate_edges(self, tmp_path):
        """Rebuild is idempotent: it must not re-create edges it already has.

        The duplicate check reads existing belief ids out of graph_edges and
        compares them to the id it is about to write. While the two sides used
        different spellings the check never matched, so a rebuild silently
        doubled the edge set.
        """
        db = str(tmp_path / "clara.db")
        memory = await LocalMemory.create(db)
        try:
            await memory.save(subject="user", relation="uses", object="Rust")
            await memory.save(subject="api", relation="runs_on", object="fly.io")
            before = await memory.graph_rebuild()
            after = await memory.graph_rebuild()
        finally:
            await memory.close()

        assert before["edges_created"] == 0, before
        assert after["edges_created"] == 0, after
        conn = sqlite3.connect(db)
        edges = conn.execute("SELECT count(*) FROM graph_edges").fetchone()[0]
        distinct = conn.execute(
            "SELECT count(*) FROM (SELECT DISTINCT src_id, dst_id, relation, belief_id"
            " FROM graph_edges)"
        ).fetchone()[0]
        assert edges == distinct, "rebuild duplicated edges"


class TestMigration8:
    def test_legacy_dashed_ids_are_repaired(self, tmp_path):
        db = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db)
        ensure_schema(conn)
        conn.execute("DELETE FROM schema_info WHERE version > 7")  # pretend v7
        memory_id = uuid.uuid4()
        conn.execute(
            "INSERT INTO memories (memory_id, memory_type, content, confidence,"
            " status, decay_rate, created_at, updated_at) VALUES"
            " (?,'belief','{}',0.9,'active',0.02,'2026-01-01','2026-01-01')",
            (memory_id.hex,),
        )
        for belief_id in (str(memory_id), memory_id.hex):
            conn.execute(
                "INSERT INTO graph_edges (edge_id, src_id, dst_id, relation,"
                " belief_id, confidence, weight, valid_from) VALUES"
                " (?,'a','b','uses',?,0.8,1.0,'2026-01-01')",
                (uuid.uuid4().hex, belief_id),
            )
        conn.commit()
        assert conn.execute(
            "SELECT count(*) FROM graph_edges WHERE belief_id LIKE '%-%'"
        ).fetchone()[0] == 1
        conn.close()

        conn = sqlite3.connect(db)
        assert ensure_schema(conn) >= 8
        assert conn.execute(
            "SELECT count(*) FROM graph_edges WHERE belief_id LIKE '%-%'"
        ).fetchone()[0] == 0
        # Both edges now join, where only the dashless one did before.
        assert conn.execute(
            "SELECT count(*) FROM memories m JOIN graph_edges e "
            "ON m.memory_id = e.belief_id"
        ).fetchone()[0] == 2


class TestNoNormalisedBeliefJoins:
    """The normalisation must not creep back into belief joins.

    Re-adding ``replace(CAST(...))`` looks defensive but is actively harmful
    here: it defeats the memories primary key and ix_graph_edges_belief, and it
    masks a writer that has started emitting the dashed form again -- which is
    the failure this whole change exists to remove.
    """

    def test_no_module_normalises_a_belief_join(self):
        root = Path(__file__).parents[1] / "clara"
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "migrations.py":  # migration 8 repairs rows on purpose
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "replace(" in line and "belief_id" in line:
                    offenders.append(f"{path.relative_to(root)}:{lineno}")
        assert not offenders, (
            "belief ids are canonical; join on plain equality instead of "
            "normalising at query time:\n  " + "\n  ".join(offenders)
        )
