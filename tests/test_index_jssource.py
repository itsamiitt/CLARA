"""
JavaScript / TypeScript scanning and specifier resolution.

The scanner was built by comparing it against the real TypeScript compiler over
2,100 files of two production repos (8,313 specifiers, exact agreement). Those
repos are not shipped, so what is pinned here is one representative case per
construct that comparison actually caught getting wrong -- each of these
assertions failed at some point during development.
"""

from __future__ import annotations

import sqlite3

import pytest

from clara.index import indexer, jssource


def specifiers(source: str) -> list[str]:
    return jssource.parse_script("a.ts", source).specifiers


class TestFormsThatNameAModule:
    def test_static_default_and_named(self) -> None:
        assert specifiers('import x from "a";\nimport {b} from "b";') == ["a", "b"]

    def test_type_only_import(self) -> None:
        assert specifiers('import type {T} from "types";') == ["types"]

    def test_bare_side_effect_import(self) -> None:
        assert specifiers('import "./polyfill";') == ["./polyfill"]

    def test_re_export(self) -> None:
        assert specifiers('export * from "./all";\nexport {a} from "./one";') == [
            "./all", "./one"
        ]

    def test_require_and_dynamic_import(self) -> None:
        assert specifiers('const a = require("a");\nconst b = import("b");') == [
            "a", "b"
        ]

    def test_await_import(self) -> None:
        assert specifiers('const m = (await import("pg")) as unknown as M;') == ["pg"]

    def test_multiline_import_list(self) -> None:
        source = 'import {\n  a,\n  b,\n} from "./mod";'
        assert specifiers(source) == ["./mod"]


class TestThingsThatOnlyLookLikeImports:
    """Every one of these produced a phantom module before the fix beside it."""

    def test_import_inside_a_line_comment(self) -> None:
        assert specifiers('// import x from "ghost";\nlet a = 1;') == []

    def test_import_inside_a_block_comment(self) -> None:
        assert specifiers('/* import x from "ghost"; */') == []

    def test_string_containing_the_word_import(self) -> None:
        # Found by the compiler comparison: `case "import":` parsed as an
        # import when string contents were left visible to the patterns.
        assert specifiers('switch (k) { case "import": break; }') == []

    def test_code_stored_in_a_string(self) -> None:
        assert specifiers("""const t = 'import x from "ghost";';""") == []

    def test_specifier_is_not_erased_by_blanking(self) -> None:
        # The opposite failure: blanking strings wholesale found 0 of 3,153.
        assert specifiers('import x from "real";') == ["real"]


class TestScannerStateSurvives:
    def test_regex_literal_containing_a_quote(self) -> None:
        source = 'const r = /["\']/g;\nimport x from "after";'
        assert specifiers(source) == ["after"]

    def test_division_is_not_a_regex(self) -> None:
        source = 'const n = a / b / c;\nimport x from "after";'
        assert specifiers(source) == ["after"]

    def test_template_literal_with_substitution_containing_a_string(self) -> None:
        # The bug that cost the last 3 of 3,153: the cursor advanced by the
        # length of the *transformed* substitution, which tokenising strings
        # made different from the source it consumed, so everything after the
        # first such template was misread.
        source = (
            'const s = `+1${String(1).padStart(7, "0")}`;\n'
            'import x from "after";'
        )
        assert specifiers(source) == ["after"]

    def test_nested_template_literals(self) -> None:
        source = 'const s = `a${`b${c}`}d`;\nimport x from "after";'
        assert specifiers(source) == ["after"]

    def test_unterminated_string_does_not_run_away(self) -> None:
        source = 'const s = "oops\nimport x from "after";'
        assert "after" in specifiers(source)


class TestDeferred:
    def test_dynamic_import_is_deferred(self) -> None:
        parsed = jssource.parse_script("a.ts", 'const m = import("lazy");')
        assert parsed.deferred == {"lazy"}

    def test_static_import_is_not_deferred(self) -> None:
        parsed = jssource.parse_script("a.ts", 'import m from "eager";')
        assert parsed.deferred == set()


class TestResolution:
    @pytest.fixture()
    def repo(self, tmp_path):
        (tmp_path / "src" / "lib").mkdir(parents=True)
        (tmp_path / "src" / "app.ts").write_text("", encoding="utf-8")
        (tmp_path / "src" / "lib" / "util.ts").write_text("", encoding="utf-8")
        (tmp_path / "src" / "lib" / "index.ts").write_text("", encoding="utf-8")
        (tmp_path / "tsconfig.json").write_text(
            '{\n  // a comment, which tsconfig files really do contain\n'
            '  "compilerOptions": {\n    "paths": { "@/*": ["./src/*"] },\n  }\n}',
            encoding="utf-8",
        )
        return tmp_path

    def test_relative_specifier_gains_its_extension(self, repo) -> None:
        assert jssource.resolve_specifier(
            repo, "src/app.ts", "./lib/util", {}
        ) == "src/lib/util.ts"

    def test_directory_resolves_to_index(self, repo) -> None:
        assert jssource.resolve_specifier(
            repo, "src/app.ts", "./lib", {}
        ) == "src/lib/index.ts"

    def test_parent_traversal(self, repo) -> None:
        assert jssource.resolve_specifier(
            repo, "src/lib/util.ts", "../app", {}
        ) == "src/app.ts"

    def test_alias_from_tsconfig_with_comments_and_trailing_comma(self, repo) -> None:
        aliases = jssource.load_path_aliases(repo)
        assert aliases == {"@/*": ["src/*"]}
        assert jssource.resolve_specifier(
            repo, "src/app.ts", "@/lib/util", aliases
        ) == "src/lib/util.ts"

    def test_package_is_external(self, repo) -> None:
        assert jssource.resolve_specifier(repo, "src/app.ts", "react", {}) is None

    def test_relative_path_that_does_not_exist_is_external(self, repo) -> None:
        # A vendored bundle keeps its original internal paths; they resolve to
        # nothing here and must not be invented as repo files.
        assert jssource.resolve_specifier(repo, "src/app.ts", "./gone", {}) is None

    def test_missing_tsconfig_is_not_an_error(self, tmp_path) -> None:
        assert jssource.load_path_aliases(tmp_path) == {}

    def test_unparseable_tsconfig_is_not_an_error(self, tmp_path) -> None:
        (tmp_path / "tsconfig.json").write_text("{ not json", encoding="utf-8")
        assert jssource.load_path_aliases(tmp_path) == {}


class TestIndexing:
    @pytest.fixture()
    def conn(self, tmp_path):
        from clara.db.migrations import ensure_schema

        connection = sqlite3.connect(tmp_path / "clara.db")
        ensure_schema(connection)
        yield connection
        connection.close()

    @pytest.fixture()
    def repo(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.ts").write_text(
            'import {u} from "./util";\nimport React from "react";\n',
            encoding="utf-8",
        )
        (tmp_path / "src" / "util.ts").write_text("export const u = 1;\n", "utf-8")
        return tmp_path

    def test_internal_edge_joins_to_the_indexed_file(self, conn, repo) -> None:
        indexer.index_repo(conn, "r", repo)
        rows = conn.execute(
            "SELECT n.qualified_name, n.file_path FROM code_edges e "
            "JOIN code_nodes n ON n.node_id = e.dst_id "
            "WHERE e.invalid_at IS NULL ORDER BY n.qualified_name"
        ).fetchall()
        # The util node carries a file_path, so the edge points at the same
        # node the file gets when indexed itself -- not a second, pathless one.
        assert ("src/util.ts", "src/util.ts") in rows
        assert ("react", None) in rows

    def test_language_is_recorded(self, conn, repo) -> None:
        indexer.index_repo(conn, "r", repo)
        langs = dict(
            conn.execute(
                "SELECT qualified_name, lang FROM code_nodes "
                "WHERE file_path IS NOT NULL"
            ).fetchall()
        )
        assert langs["src/app.ts"] == "typescript"

    def test_declaration_files_are_not_indexed(self, conn, repo) -> None:
        (repo / "src" / "types.d.ts").write_text("export type A = 1;", "utf-8")
        indexer.index_repo(conn, "r", repo)
        names = [
            r[0] for r in conn.execute("SELECT qualified_name FROM code_nodes")
        ]
        assert "src/types.d.ts" not in names

    def test_vendored_directories_are_pruned(self, conn, repo) -> None:
        vendored = repo / "node_modules" / "pkg"
        vendored.mkdir(parents=True)
        (vendored / "index.js").write_text('import "x";', encoding="utf-8")
        assert "node_modules/pkg/index.js" not in indexer.walk_repo(repo)

    def test_reindex_replaces_edges_rather_than_appending(self, conn, repo) -> None:
        indexer.index_repo(conn, "r", repo)
        (repo / "src" / "app.ts").write_text('import "react";\n', encoding="utf-8")
        indexer.index_repo(conn, "r", repo, force=True)
        live = [
            r[0]
            for r in conn.execute(
                "SELECT n.qualified_name FROM code_edges e "
                "JOIN code_nodes n ON n.node_id = e.dst_id "
                "JOIN code_nodes s ON s.node_id = e.src_id "
                "WHERE e.invalid_at IS NULL AND s.qualified_name = 'src/app.ts'"
            )
        ]
        assert live == ["react"]

    def test_query_walk_finds_transitive_dependency(self, conn, repo) -> None:
        from clara.index import queries

        (repo / "src" / "util.ts").write_text('import "./deep";\n', encoding="utf-8")
        (repo / "src" / "deep.ts").write_text("export const d = 1;\n", "utf-8")
        indexer.index_repo(conn, "r", repo)
        found = queries.dependencies(conn, "r", "src/app.ts", depth=2)
        assert [(d.qualified_name, d.depth) for d in found] == [
            ("react", 1), ("src/util.ts", 1), ("src/deep.ts", 2),
        ]
