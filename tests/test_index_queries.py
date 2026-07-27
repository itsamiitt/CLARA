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


class TestEntrypointsAreNotDeadCode:
    """"Nothing imports it" is not the same as "nothing runs it".

    A CLI main, a `__main__` guard and every pytest module are legitimately
    unreferenced. Before this filter CLARA's own tree reported 69 unused
    modules, none of them actually unused; with it, 0 — and a genuinely
    orphaned module is still found.
    """

    @staticmethod
    def _project(tmp_path, *, scripts: str = "", testpaths: str = "") -> object:
        root = tmp_path / "proj"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "lib.py").write_text("import os\n", encoding="utf-8")
        (root / "pkg" / "cli.py").write_text("from pkg import lib\n", encoding="utf-8")
        (root / "pkg" / "runner.py").write_text(
            'def go():\n    pass\n\n\nif __name__ == "__main__":\n    go()\n',
            encoding="utf-8",
        )
        (root / "pkg" / "orphan.py").write_text("def nobody():\n    pass\n",
                                                encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "test_thing.py").write_text("def test_x():\n    pass\n",
                                                      encoding="utf-8")
        manifest = "[project]\nname = 'p'\n"
        if scripts:
            manifest += f"\n[project.scripts]\n{scripts}\n"
        if testpaths:
            manifest += f"\n[tool.pytest.ini_options]\ntestpaths = {testpaths}\n"
        (root / "pyproject.toml").write_text(manifest, encoding="utf-8")
        return root

    def _unused(self, tmp_path, root):
        conn = sqlite3.connect(tmp_path / "c.db")
        ensure_schema(conn)
        try:
            indexer.index_repo(conn, REPO, root)
            return queries.unused_modules(conn, REPO, repo_root=root)
        finally:
            conn.close()

    def test_a_console_script_is_not_unused(self, tmp_path):
        root = self._project(tmp_path, scripts='p = "pkg.cli:main"')
        assert "pkg.cli" not in self._unused(tmp_path, root)

    def test_a_main_guard_is_not_unused(self, tmp_path):
        root = self._project(tmp_path)
        assert "pkg.runner" not in self._unused(tmp_path, root), (
            "a module with a __main__ guard is run, not imported"
        )

    def test_pytest_testpaths_are_not_unused(self, tmp_path):
        root = self._project(tmp_path, testpaths='["tests"]')
        assert not [u for u in self._unused(tmp_path, root) if u.startswith("tests")]

    def test_a_genuinely_orphaned_module_is_still_reported(self, tmp_path):
        """The filter must not swallow the signal it exists to sharpen."""
        root = self._project(tmp_path, scripts='p = "pkg.cli:main"',
                             testpaths='["tests"]')
        assert "pkg.orphan" in self._unused(tmp_path, root)

    def test_without_a_manifest_nothing_is_excluded(self, tmp_path):
        """No declaration means no evidence; report the raw list rather than
        guessing which modules are entry points."""
        root = self._project(tmp_path)
        (root / "pyproject.toml").unlink()
        assert queries.declared_entrypoints(root) == set()
        assert queries.test_roots(root) == ()

    def test_a_malformed_manifest_is_survived(self, tmp_path):
        root = self._project(tmp_path)
        (root / "pyproject.toml").write_text("[project\nbroken", encoding="utf-8")
        assert queries.declared_entrypoints(root) == set()

    def test_reading_the_real_manifest(self):
        """CLARA's own declarations, as evidence the parsing is right."""
        from pathlib import Path as _Path

        root = _Path(__file__).parents[1]
        assert queries.declared_entrypoints(root) == {
            "clara.cli",
            "clara.integrations.mcp_server",
        }
        assert queries.test_roots(root) == ("tests",)
