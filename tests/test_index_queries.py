"""Code-graph queries (plan §7).

Fixtures are tiny repos where the right answer is obvious by reading them, so
a wrong query fails rather than looking plausible. The traversals are
seed-restricted by construction; the belief graph's unrestricted version cost
1.4 s per query on a 100k-edge store (audit P3) and the plan forbids repeating
it, so depth bounding is asserted too.
"""

from __future__ import annotations

import sqlite3

import pytest

from clara.db.migrations import ensure_schema
from clara.index import indexer, queries

REPO = "q-repo"


@pytest.fixture
def graph(tmp_path):
    """a -> b -> c, plus d importing a. One clean chain and one fan-in."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "c.py").write_text("import os\n", encoding="utf-8")
    (root / "pkg" / "b.py").write_text("from pkg import c\n", encoding="utf-8")
    (root / "pkg" / "a.py").write_text("from pkg import b\n", encoding="utf-8")
    (root / "pkg" / "d.py").write_text("from pkg import a\n", encoding="utf-8")
    conn = sqlite3.connect(tmp_path / "clara.db")
    ensure_schema(conn)
    indexer.index_repo(conn, REPO, root)
    yield conn
    conn.close()


def names(rows) -> list[str]:
    return [r.qualified_name for r in rows]


class TestDependencies:
    def test_forward_one_hop(self, graph):
        found = names(queries.dependencies(graph, REPO, "pkg.a", depth=1))
        assert "pkg.b" in found
        assert "pkg.c" not in found, "depth 1 must not reach two hops"

    def test_forward_transitive(self, graph):
        found = names(queries.dependencies(graph, REPO, "pkg.a", depth=3))
        assert "pkg.b" in found and "pkg.c" in found

    def test_depth_is_reported(self, graph):
        by_name = {d.qualified_name: d.depth
                   for d in queries.dependencies(graph, REPO, "pkg.a", depth=3)}
        assert by_name["pkg.b"] == 1
        assert by_name["pkg.c"] == 2

    def test_reverse_is_the_impact_set(self, graph):
        found = names(queries.impact(graph, REPO, "pkg.c", depth=3))
        assert {"pkg.b", "pkg.a", "pkg.d"} <= set(found), (
            "everything upstream of c is affected by changing it"
        )

    def test_impact_of_a_leaf_is_small(self, graph):
        assert names(queries.impact(graph, REPO, "pkg.d", depth=3)) == []

    def test_accepts_a_file_path_as_the_seed(self, graph):
        by_path = names(queries.dependencies(graph, REPO, "pkg/a.py", depth=1))
        by_name = names(queries.dependencies(graph, REPO, "pkg.a", depth=1))
        assert by_path == by_name

    def test_unknown_seed_returns_nothing(self, graph):
        assert queries.dependencies(graph, REPO, "not.a.module") == []

    def test_direction_is_validated(self, graph):
        with pytest.raises(ValueError, match="forward"):
            queries.dependencies(graph, REPO, "pkg.a", direction="sideways")

    def test_depth_is_clamped(self, graph):
        """An unbounded walk over a real import graph returns the repo."""
        deep = queries.dependencies(graph, REPO, "pkg.a", depth=10_000)
        assert all(d.depth <= queries.MAX_DEPTH for d in deep)

    def test_limit_is_honoured(self, graph):
        assert len(queries.dependencies(graph, REPO, "pkg.a", depth=3, limit=1)) == 1


class TestUnusedModules:
    def test_reports_only_modules_nothing_imports(self, graph):
        unused = queries.unused_modules(graph, REPO)
        assert "pkg.d" in unused, "nothing imports d"
        assert "pkg.a" not in unused, "d imports a"
        assert "pkg.c" not in unused

    def test_a_package_imported_only_implicitly_is_not_unused(self, graph):
        """`from pkg import b` executes pkg/__init__.py — a real dependency."""
        assert "pkg" not in queries.unused_modules(graph, REPO)


class TestCycles:
    def test_no_cycle_in_an_acyclic_graph(self, graph):
        assert queries.find_cycles(graph, REPO) == []

    def test_finds_a_two_module_cycle(self, tmp_path):
        root = tmp_path / "r"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "x.py").write_text("from pkg import y\n", encoding="utf-8")
        (root / "pkg" / "y.py").write_text(
            "def f():\n    from pkg import x\n    return x\n", encoding="utf-8"
        )
        conn = sqlite3.connect(tmp_path / "c.db")
        ensure_schema(conn)
        indexer.index_repo(conn, REPO, root)
        try:
            cycles = queries.find_cycles(conn, REPO)
            assert cycles, "a deferred import still closes a cycle"
            members = {frozenset(c) for c in cycles}
            assert any({"pkg.x", "pkg.y"} <= m for m in members)
        finally:
            conn.close()

    def test_a_cycle_is_reported_once(self, tmp_path):
        root = tmp_path / "r"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "x.py").write_text("from pkg import y\n", encoding="utf-8")
        (root / "pkg" / "y.py").write_text("from pkg import x\n", encoding="utf-8")
        conn = sqlite3.connect(tmp_path / "c.db")
        ensure_schema(conn)
        indexer.index_repo(conn, REPO, root)
        try:
            cycles = queries.find_cycles(conn, REPO)
            pairs = [c for c in cycles if {"pkg.x", "pkg.y"} <= set(c)]
            assert len(pairs) == 1, f"x<->y reported {len(pairs)} times"
        finally:
            conn.close()


class TestRetiredEdgesAreExcluded:
    def test_a_removed_import_stops_showing_as_a_dependency(self, graph, tmp_path):
        from clara.index import journal

        root = tmp_path / "repo"
        (root / "pkg" / "a.py").write_text("import os\n", encoding="utf-8")
        journal.enqueue(graph, REPO, change="modified", path="pkg/a.py")
        indexer.drain_journal(graph, REPO, root, worker="w1")

        assert "pkg.b" not in names(queries.dependencies(graph, REPO, "pkg.a", depth=2))
        assert "pkg.a" not in names(queries.impact(graph, REPO, "pkg.b", depth=2))
