"""
Python source → nodes and edges, via stdlib ``ast``.

No new dependency and no third-party parser: ``ast`` is exact for Python, and
CLARA's own tree is the ground truth this is tested against.

What it extracts, and nothing more:

* one ``module`` node per file;
* ``function`` and ``class`` nodes for top-level and one-level-nested defs
  (methods), with line spans;
* ``imports`` edges from the module to whatever it imports.

Import targets are recorded as written -- ``clara.index.journal``, ``.state`` --
and resolved to real modules later, where the repo layout is known. Resolution
belongs to the indexer, not the parser: the parser sees one file and cannot
know what else exists.

A file that does not parse yields the module node and no edges, so a syntax
error in one file costs that file's edges rather than the run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceNode:
    kind: str  # 'module' | 'function' | 'class'
    qualified_name: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class SourceImport:
    """One import as written, before resolution."""

    module: str  # 'clara.index.journal', or '.state' for a relative import
    level: int  # 0 = absolute, 1 = '.', 2 = '..'
    names: tuple[str, ...]  # imported symbols, empty for `import x`
    line: int
    # False = executed at import time; True = inside a function, `if
    # TYPE_CHECKING`, or a try/except. A deferred import is a real dependency
    # but a weaker one, and CLARA's own tree is full of them deliberately --
    # clara.cli defers four heavy imports to keep the status line fast.
    deferred: bool = False


@dataclass(slots=True)
class ParsedModule:
    rel_path: str
    is_package: bool = False
    nodes: list[SourceNode] = field(default_factory=list)
    imports: list[SourceImport] = field(default_factory=list)
    syntax_error: str | None = None


def module_name_for(rel_path: str) -> str:
    """Dotted module name for a repo-relative path.

    ``clara/index/journal.py`` -> ``clara.index.journal``
    ``clara/index/__init__.py`` -> ``clara.index``
    """
    parts = rel_path.replace("\\", "/").split("/")
    if not parts:
        return ""
    last = parts[-1]
    if last.endswith(".py"):
        last = last[:-3]
    parts = parts[:-1] if last == "__init__" else [*parts[:-1], last]
    return ".".join(p for p in parts if p)


def parse_module(rel_path: str, source: str) -> ParsedModule:
    """Parse one Python file. Never raises on bad syntax."""
    parsed = ParsedModule(
        rel_path=rel_path,
        is_package=rel_path.replace("\\", "/").endswith("/__init__.py")
        or rel_path == "__init__.py",
    )
    module_name = module_name_for(rel_path)
    parsed.nodes.append(
        SourceNode(kind="module", qualified_name=module_name, start_line=1, end_line=1)
    )
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        parsed.syntax_error = f"line {exc.lineno}: {exc.msg}"
        return parsed
    except ValueError as exc:  # e.g. source containing null bytes
        parsed.syntax_error = str(exc)
        return parsed

    # Imports are collected from the whole tree, not just the top level.
    # A deferred import is still a dependency: clara/maintenance.py imports
    # clara.db.migrations inside a nested function, and a depth-limited walk
    # missed it -- verified against ast.walk. Under-reporting dependencies is
    # the wrong failure for "what breaks if I change this".
    module_level = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _collect_import(node, parsed, deferred=id(node) not in module_level)

    # Definitions keep the depth rule: methods are worth naming, closures are
    # not, and a nested helper's qualified name is noise.
    for statement in tree.body:
        _collect_defs(statement, module_name, parsed, depth=0)
    return parsed


def _collect_defs(node: ast.AST, prefix: str, parsed: ParsedModule, *, depth: int) -> None:
    """Record function/class nodes. Imports are handled separately."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    qualified = f"{prefix}.{node.name}" if prefix else node.name
    parsed.nodes.append(
        SourceNode(
            kind=kind,
            qualified_name=qualified,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
        )
    )
    if depth == 0:
        for body_item in node.body:
            _collect_defs(body_item, qualified, parsed, depth=depth + 1)


def _collect_import(
    node: ast.Import | ast.ImportFrom,
    parsed: ParsedModule,
    *,
    deferred: bool = False,
) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            parsed.imports.append(
                SourceImport(
                    module=alias.name, level=0, names=(),
                    line=node.lineno, deferred=deferred,
                )
            )
        return
    parsed.imports.append(
        SourceImport(
            module=node.module or "",
            level=node.level or 0,
            names=tuple(alias.name for alias in node.names),
            line=node.lineno,
            deferred=deferred,
        )
    )


def resolve_import(
    source_module: str, imported: SourceImport, *, is_package: bool = False
) -> str:
    """Absolute dotted name for an import as written in *source_module*.

    ``from .state import x`` inside ``clara.index.journal`` is
    ``clara.index.state``.

    *is_package* matters and is easy to get wrong. Inside a package's
    ``__init__.py`` the current package is the module itself, so
    ``from . import openclaw_bridge`` in ``clara/integrations/__init__.py``
    means ``clara.integrations.openclaw_bridge``. Treating that file like an
    ordinary module drops a component and resolves it to ``clara`` -- which is
    how clara.integrations.openclaw_bridge came out looking unimported.
    """
    if imported.level == 0:
        return imported.module
    package_parts = source_module.split(".")
    # A package's own name is already the package; a module's is not.
    drop = imported.level - 1 if is_package else imported.level
    base = package_parts[: len(package_parts) - drop] if drop else package_parts
    if imported.module:
        base = [*base, *imported.module.split(".")]
    return ".".join(p for p in base if p)
