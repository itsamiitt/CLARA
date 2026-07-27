"""Python source indexing and the code graph (plan §3.4, §4.2, §4.3).

The parser is checked against stdlib ``ast`` rather than against fixtures it
was written from: for every file in CLARA's own tree, the set of imports it
reports must equal what ``ast.walk`` finds. That caught the defect worth
catching — a depth-limited walk missed the 134 deferred imports this codebase
uses on purpose, including clara/maintenance.py importing clara.db.migrations
inside a nested function.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from clara.db.migrations import ensure_schema
from clara.index import indexer, journal, pysource

REPO = "test-repo"


@pytest.fixture
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "clara.db")
    ensure_schema(db)
    yield db
    db.close()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (root / "pkg" / "b.py").write_text("import sys\n", encoding="utf-8")
    (root / "pkg" / "a.py").write_text(
        "from pkg import b\nimport os\n\n\nclass Thing:\n    def method(self):\n"
        "        pass\n\n\ndef top():\n    pass\n",
        encoding="utf-8",
    )
    return root


def live_edges(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (src, dst)
        for src, dst in conn.execute(
            "SELECT s.qualified_name, d.qualified_name FROM code_edges e "
            "JOIN code_nodes s ON s.node_id = e.src_id "
            "JOIN code_nodes d ON d.node_id = e.dst_id "
            "WHERE e.invalid_at IS NULL ORDER BY 1, 2"
        )
    ]


class TestModuleNaming:
    @pytest.mark.parametrize(
        ("rel_path", "expected"),
        [
            ("clara/index/journal.py", "clara.index.journal"),
            ("clara/index/__init__.py", "clara.index"),
            ("top.py", "top"),
            ("a/b/c/__init__.py", "a.b.c"),
        ],
    )
    def test_dotted_name(self, rel_path, expected):
        assert pysource.module_name_for(rel_path) == expected

    @pytest.mark.parametrize(
        ("level", "module", "expected"),
        [
            (0, "os", "os"),
            (1, "state", "clara.index.state"),
            (2, "db.models", "clara.db.models"),
        ],
    )
    def test_relative_imports_resolve(self, level, module, expected):
        imported = pysource.SourceImport(module=module, level=level, names=(), line=1)
        assert pysource.resolve_import("clara.index.journal", imported) == expected


class TestParserMatchesAst:
    """Ground truth is the stdlib parser, not a fixture."""

    def _repo_files(self) -> list[Path]:
        root = Path(__file__).parents[1] / "clara"
        return sorted(root.rglob("*.py"))

    def test_every_import_in_the_tree_is_found(self):
        checked = 0
        for path in self._repo_files():
            rel = path.relative_to(Path(__file__).parents[1]).as_posix()
            source = path.read_text(encoding="utf-8-sig", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:  # pragma: no cover - tree is expected to parse
                continue
            checked += 1
            module = pysource.module_name_for(rel)

            truth: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    truth |= {alias.name for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    truth.add(
                        pysource.resolve_import(
                            module,
                            pysource.SourceImport(
                                module=node.module or "", level=node.level or 0,
                                names=(), line=node.lineno,
                            ),
                        )
                    )
            mine = {
                pysource.resolve_import(module, imported)
                for imported in pysource.parse_module(rel, source).imports
            }
            assert truth - mine == set(), f"{rel}: missed {truth - mine}"
            assert mine - truth == set(), f"{rel}: invented {mine - truth}"
        assert checked > 50, "the tree should have plenty of files to check"

    def test_deferred_imports_are_flagged_but_still_reported(self):
        source = (
            "import os\n\n\n"
            "def f():\n"
            "    from json import loads\n"
            "    return loads, os\n"
        )
        parsed = pysource.parse_module("m.py", source)
        by_module = {i.module: i for i in parsed.imports}
        assert by_module["os"].deferred is False
        assert by_module["json"].deferred is True, (
            "a function-level import is a real dependency and must be reported"
        )

    def test_a_syntax_error_costs_that_file_only(self):
        parsed = pysource.parse_module("bad.py", "def (:\n")
        assert parsed.syntax_error is not None
        assert [n.kind for n in parsed.nodes] == ["module"]
        assert parsed.imports == []

    def test_a_bom_does_not_break_parsing(self, tmp_path, conn):
        """Windows-authored sources carry one; CLARA's own tree has one."""
        root = tmp_path / "r"
        root.mkdir()
        (root / "m.py").write_bytes(b"\xef\xbb\xbfimport os\n")
        result = indexer.IndexResult()
        indexer.index_file(conn, REPO, root, "m.py", result)
        assert result.syntax_errors == 0, "a BOM must not fail the file"
        assert ("m", "os") in live_edges(conn)


class TestDefinitions:
    def test_classes_methods_and_functions_are_named(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        names = {
            row[0]
            for row in conn.execute(
                "SELECT qualified_name FROM code_nodes WHERE kind != 'module'"
            )
        }
        assert "pkg.a.Thing" in names
        assert "pkg.a.Thing.method" in names
        assert "pkg.a.top" in names

    def test_spans_are_recorded(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        span = conn.execute(
            "SELECT span FROM code_nodes WHERE qualified_name = 'pkg.a.top'"
        ).fetchone()[0]
        assert '"start_line"' in span


class TestImportEdges:
    def test_submodule_import_edges_to_the_submodule(self, repo, conn):
        """`from pkg import b` where b is a module means pkg.b, not pkg."""
        indexer.index_repo(conn, REPO, repo)
        assert ("pkg.a", "pkg.b") in live_edges(conn)

    def test_name_import_edges_to_the_package(self, repo, conn):
        (repo / "pkg" / "c.py").write_text("from pkg import helper\n", encoding="utf-8")
        indexer.index_repo(conn, REPO, repo)
        edges = live_edges(conn)
        assert ("pkg.c", "pkg") in edges
        assert ("pkg.c", "pkg.helper") not in edges, "helper is a function, not a module"

    def test_external_imports_are_recorded(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        assert ("pkg.a", "os") in live_edges(conn)

    def test_reverse_lookup_answers_who_imports_this(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        importers = [
            row[0]
            for row in conn.execute(
                "SELECT s.qualified_name FROM code_edges e "
                "JOIN code_nodes s ON s.node_id = e.src_id "
                "JOIN code_nodes d ON d.node_id = e.dst_id "
                "WHERE e.invalid_at IS NULL AND d.qualified_name = 'pkg.b'"
            )
        ]
        assert importers == ["pkg.a"]


class TestIncrementality:
    def test_unchanged_files_are_skipped(self, repo, conn):
        first = indexer.index_repo(conn, REPO, repo)
        assert first.processed == 3
        second = indexer.index_repo(conn, REPO, repo)
        assert second.processed == 0
        assert second.skipped_unchanged == 3

    def test_editing_one_file_leaves_other_edges_alone(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        (repo / "pkg" / "a.py").write_text("import json\n", encoding="utf-8")
        journal.enqueue(conn, REPO, change="modified", path="pkg/a.py")

        result = indexer.drain_journal(conn, REPO, repo, worker="w1")
        assert result.processed == 1, "only the edited file should be re-parsed"

        edges = live_edges(conn)
        assert ("pkg.a", "json") in edges, "the new import is recorded"
        assert ("pkg.a", "pkg.b") not in edges, "the dropped import is retired"
        assert ("pkg.b", "sys") in edges, "an untouched file keeps its edges"

    def test_retired_edges_are_kept_as_history(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        (repo / "pkg" / "a.py").write_text("import json\n", encoding="utf-8")
        journal.enqueue(conn, REPO, change="modified", path="pkg/a.py")
        indexer.drain_journal(conn, REPO, repo, worker="w1")

        retired = conn.execute(
            "SELECT count(*) FROM code_edges WHERE invalid_at IS NOT NULL"
        ).fetchone()[0]
        assert retired > 0, "edges are invalidated, never deleted"

    def test_a_deleted_file_retires_its_nodes(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        (repo / "pkg" / "b.py").unlink()
        journal.enqueue(conn, REPO, change="removed", path="pkg/b.py")
        indexer.drain_journal(conn, REPO, repo, worker="w1")

        status = conn.execute(
            "SELECT status FROM code_nodes WHERE qualified_name = 'pkg.b'"
        ).fetchone()[0]
        assert status == "removed"
        assert ("pkg.b", "sys") not in live_edges(conn)

    def test_draining_clears_the_queue(self, repo, conn):
        indexer.index_repo(conn, REPO, repo)
        journal.enqueue(conn, REPO, change="modified", path="pkg/a.py")
        indexer.drain_journal(conn, REPO, repo, worker="w1")
        assert journal.pending_count(conn, REPO) == 0

    def test_repo_level_entries_are_consumed_not_parsed(self, repo, conn):
        journal.enqueue(conn, REPO, change="git")
        result = indexer.drain_journal(conn, REPO, repo, worker="w1")
        assert result.processed == 0
        assert journal.pending_count(conn, REPO) == 0


class TestWalk:
    def test_vendored_directories_are_skipped(self, tmp_path):
        root = tmp_path / "r"
        (root / "node_modules" / "x").mkdir(parents=True)
        (root / "__pycache__").mkdir(parents=True)
        (root / "src").mkdir(parents=True)
        (root / "node_modules" / "x" / "dep.py").write_text("", encoding="utf-8")
        (root / "__pycache__" / "cached.py").write_text("", encoding="utf-8")
        (root / "src" / "real.py").write_text("", encoding="utf-8")
        assert indexer.walk_repo(root) == ["src/real.py"]


class TestNodeIds:
    def test_ids_are_stable_and_distinct(self):
        first = indexer.node_id("r", "module", "pkg.a")
        assert first == indexer.node_id("r", "module", "pkg.a")
        assert first != indexer.node_id("r", "module", "pkg.b")
        assert first != indexer.node_id("r", "function", "pkg.a")
        assert first != indexer.node_id("other", "module", "pkg.a")

    def test_a_module_referenced_before_it_is_indexed_converges(self, repo, conn):
        """Import order must not change the graph.

        `pkg/a.py` imports `pkg.b`, so the node for pkg.b may be created by the
        import before b.py itself is read. It must end up with a file_path
        either way, not two rows.
        """
        indexer.index_repo(conn, REPO, repo)
        rows = conn.execute(
            "SELECT file_path FROM code_nodes "
            "WHERE kind = 'module' AND qualified_name = 'pkg.b'"
        ).fetchall()
        assert rows == [("pkg/b.py",)], rows
