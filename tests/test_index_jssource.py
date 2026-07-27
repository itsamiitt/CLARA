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

from clara.index import indexer, jssource, queries


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

    def test_import_scripts_is_an_edge(self, conn, repo) -> None:
        # A service worker loads its code with importScripts, which the
        # TypeScript compiler does not model. Without it, 24 live files in the
        # corpus repo looked like dead code.
        (repo / "src" / "worker.js").write_text(
            'importScripts("util.js");\n', encoding="utf-8"
        )
        (repo / "src" / "util.js").write_text("var a = 1;\n", encoding="utf-8")
        indexer.index_repo(conn, "r", repo)
        targets = [
            r[0]
            for r in conn.execute(
                "SELECT n.qualified_name FROM code_edges e "
                "JOIN code_nodes n ON n.node_id = e.dst_id "
                "JOIN code_nodes s ON s.node_id = e.src_id "
                "WHERE e.invalid_at IS NULL AND s.qualified_name = 'src/worker.js'"
            )
        ]
        # Resolved as a sibling file, not mistaken for an npm package.
        assert targets == ["src/util.js"]

    def test_query_walk_finds_transitive_dependency(self, conn, repo) -> None:
        from clara.index import queries

        (repo / "src" / "util.ts").write_text('import "./deep";\n', encoding="utf-8")
        (repo / "src" / "deep.ts").write_text("export const d = 1;\n", "utf-8")
        indexer.index_repo(conn, "r", repo)
        found = queries.dependencies(conn, "r", "src/app.ts", depth=2)
        assert [(d.qualified_name, d.depth) for d in found] == [
            ("react", 1), ("src/util.ts", 1), ("src/deep.ts", 2),
        ]


class TestEntrypointEvidence:
    """What keeps `unused_modules` from calling a third of a repo dead.

    Before this, a real 2,568-file Node repo reported 848 files (33%) as
    unreferenced -- including vite.config.ts, every *.test.ts, and every app
    entry. Each rule below is evidence the project states about itself, or a
    convention a named tool defines, not a guess about what looks unimportant.
    """

    def test_package_json_main_bin_and_scripts(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text(
            '{"main": "./lib/index.js", "bin": {"cli": "bin/run.js"},'
            ' "scripts": {"start": "tsx server/index.ts --port 3000"}}',
            encoding="utf-8",
        )
        found = queries.script_entrypoints(tmp_path)
        assert found == {"lib/index.js", "bin/run.js", "server/index.ts"}

    def test_monorepo_paths_resolve_under_their_own_package(self, tmp_path) -> None:
        pkg = tmp_path / "apps" / "admin"
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text(
            '{"main": "src/main.tsx"}', encoding="utf-8"
        )
        assert queries.script_entrypoints(tmp_path) == {"apps/admin/src/main.tsx"}

    def test_any_html_page_not_only_index(self, tmp_path) -> None:
        # popup.html/sidepanel.html load extension code; scanning only
        # index.html left 42 live files looking dead.
        (tmp_path / "popup.html").write_text(
            '<script src="agent.js"></script>', encoding="utf-8"
        )
        assert "agent.js" in queries.script_entrypoints(tmp_path)

    def test_remote_script_src_is_not_a_repo_file(self, tmp_path) -> None:
        (tmp_path / "index.html").write_text(
            '<script src="https://cdn.example.com/x.js"></script>', encoding="utf-8"
        )
        assert queries.script_entrypoints(tmp_path) == set()

    def test_chrome_extension_manifest(self, tmp_path) -> None:
        (tmp_path / "manifest.json").write_text(
            '{"manifest_version": 3, "background": {"service_worker": "bg.js"},'
            ' "content_scripts": [{"js": ["content.js"]}]}',
            encoding="utf-8",
        )
        assert queries.script_entrypoints(tmp_path) == {"bg.js", "content.js"}

    def test_a_plain_manifest_json_is_not_an_extension(self, tmp_path) -> None:
        (tmp_path / "manifest.json").write_text('{"files": ["a.js"]}', encoding="utf-8")
        assert queries.script_entrypoints(tmp_path) == set()

    def test_vendored_packages_are_not_read(self, tmp_path) -> None:
        vendored = tmp_path / "node_modules" / "left-pad"
        vendored.mkdir(parents=True)
        (vendored / "package.json").write_text(
            '{"main": "index.js"}', encoding="utf-8"
        )
        assert queries.script_entrypoints(tmp_path) == set()

    def test_unreadable_manifest_is_not_an_error(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text("{ not json", encoding="utf-8")
        assert queries.script_entrypoints(tmp_path) == set()

    @pytest.mark.parametrize(
        "path",
        [
            "src/thing.test.ts",
            "src/thing.spec.tsx",
            "__tests__/helper.ts",
            "vite.config.ts",
            "eslint.config.js",          # root level: no slash in the name
            ".dependency-cruiser.cjs",   # a dotfile that is a tool's config
            "app/dashboard/page.tsx",
            "app/api/hook/route.ts",
            "pages/about.tsx",
        ],
    )
    def test_conventional_entrypoints(self, path) -> None:
        assert queries.is_conventional_entry(path) is True

    @pytest.mark.parametrize(
        "path", ["src/lib/utils.ts", "client/src/components/ui/slider.tsx"]
    )
    def test_ordinary_modules_are_not_exempt(self, path) -> None:
        # These must stay reportable -- both were verified to have no importer
        # in the corpus repo, so exempting them would hide a true finding.
        assert queries.is_conventional_entry(path) is False

    def test_unused_modules_applies_the_evidence(self, tmp_path) -> None:
        from clara.db.migrations import ensure_schema

        (tmp_path / "src").mkdir()
        (tmp_path / "package.json").write_text(
            '{"main": "src/main.ts"}', encoding="utf-8"
        )
        (tmp_path / "src" / "main.ts").write_text("export const m = 1;", "utf-8")
        (tmp_path / "src" / "orphan.ts").write_text("export const o = 1;", "utf-8")
        (tmp_path / "vite.config.ts").write_text("export default {};", "utf-8")
        conn = sqlite3.connect(tmp_path / "clara.db")
        ensure_schema(conn)
        indexer.index_repo(conn, "r", tmp_path)
        # main.ts is declared, vite.config.ts is conventional; only the real
        # orphan is left.
        assert queries.unused_modules(conn, "r", repo_root=tmp_path) == [
            "src/orphan.ts"
        ]
        conn.close()


class TestMcpToolsActuallyRun:
    """The code tools, exercised through the MCP surface rather than inspected.

    These exist because the suite previously only asserted that code_deps,
    code_impact and code_health were *registered*. All three raised
    "SQLite objects created in a thread can only be used in that same thread"
    on every call, for every repo, and nothing caught it: the connection was
    opened inside asyncio.to_thread and then used back on the event loop.
    Registration is not evidence a tool works.
    """

    @pytest.fixture()
    def indexed_repo(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp")
        from clara.db.migrations import open_db
        from clara.repoid import repo_id
        from clara.store import resolve_store

        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "store"))
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "app.ts").write_text(
            'import {u} from "./util";\nimport React from "react";\n', "utf-8"
        )
        (repo / "src" / "util.ts").write_text("export const u = 1;\n", "utf-8")
        (repo / "src" / "orphan.ts").write_text("export const o = 1;\n", "utf-8")

        resolution = resolve_store(str(repo), create=True)
        conn = open_db(str(resolution.db_path))
        indexer.index_repo(conn, repo_id(str(repo)), repo)
        conn.commit()
        conn.close()
        return repo

    def call(self, name, args):
        import asyncio

        from clara.integrations.mcp_server import build_server

        server = build_server()
        result = asyncio.run(server.call_tool(name, args))
        payload = result[1] if isinstance(result, tuple) else result
        assert "same thread" not in str(payload), (
            f"{name} raised a cross-thread sqlite error: {payload}"
        )
        return payload

    def test_code_deps_returns_real_dependencies(self, indexed_repo) -> None:
        payload = self.call(
            "code_deps", {"target": "src/app.ts", "depth": 1,
                          "repo": str(indexed_repo)}
        )
        assert payload["indexed"] is True
        names = {m["name"] for m in payload["modules"]}
        assert names == {"src/util.ts", "react"}

    def test_code_impact_returns_reverse_dependencies(self, indexed_repo) -> None:
        payload = self.call(
            "code_impact", {"target": "src/util.ts", "depth": 2,
                            "repo": str(indexed_repo)}
        )
        assert payload["indexed"] is True
        assert [m["name"] for m in payload["modules"]] == ["src/app.ts"]

    def test_code_health_runs_and_finds_the_orphan(self, indexed_repo) -> None:
        payload = self.call("code_health", {"repo": str(indexed_repo)})
        assert payload["indexed"] is True
        assert "src/orphan.ts" in payload["unused_modules"]
        # The note must describe the evidence actually applied, JS included.
        assert "package.json" in payload["note"]

    def test_unindexed_repo_says_so_rather_than_reporting_none(
        self, tmp_path, monkeypatch
    ) -> None:
        pytest.importorskip("mcp")
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "store2"))
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        payload = self.call("code_deps", {"target": "x.ts", "repo": str(fresh)})
        assert payload["indexed"] is False
        assert "clara index" in payload["hint"]
