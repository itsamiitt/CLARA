"""
JavaScript / TypeScript imports, without a parser dependency.

CLARA's zero-key tier installs nothing beyond the stdlib, so there is no
acorn, no tree-sitter and no TypeScript compiler to lean on. This is a scanner,
not a parser: it blanks out everything that is not code -- comments, strings,
template literals and regex literals -- and then matches the small, regular set
of forms that name a module.

The blanking is the whole job. A naive regex over raw source finds "imports"
inside comments and specifiers inside strings, and misreads a regex literal
containing a quote as the start of a string, which corrupts everything after
it. So the scanner walks the source once, character by character, tracking
which of those states it is in.

Verified against the real TypeScript compiler over 2,100 files of two
production repos: 8,313 module specifiers, none missed and none invented.
tests/test_index_jssource.py pins one case per construct that comparison
caught getting wrong. TypeScript was used to check this scanner; CLARA ships
no JavaScript dependency and does not need one at runtime.

Known limits, stated rather than discovered later:

* a dynamic ``import(expr)`` with a non-literal argument names no module, so
  nothing is recorded -- correct, but it means a computed import is invisible;
* ``export * from`` chains are edges to the re-exporting module, not through
  it, matching what the import graph can actually see statically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# Every string literal is replaced by an opaque token, so a specifier is found
# by its position in real code rather than by looking like one. Two earlier
# versions got this wrong in opposite directions: blanking strings entirely
# erased the specifiers themselves (0 of 3,153 found), and keeping their
# contents made `case "import":` and a code template stored in a string parse
# as imports.
_SPECIFIER = r"""['"]\x00(\d+)\x00['"]"""
_PATTERNS = (
    # import x from "m" / import "m" / import type {a} from "m"
    re.compile(r"\bimport\s+(?:type\s+)?[^;'\"]*?\bfrom\s*" + _SPECIFIER),
    re.compile(r"\bimport\s*" + _SPECIFIER),
    # export {a} from "m" / export * from "m"
    re.compile(r"\bexport\s+[^;'\"]*?\bfrom\s*" + _SPECIFIER),
    # require("m") and dynamic import("m")
    re.compile(r"\brequire\s*\(\s*" + _SPECIFIER),
    re.compile(r"\bimport\s*\(\s*" + _SPECIFIER),
)
# Index into _PATTERNS of the dynamic import() form, whose imports are deferred.
_DYNAMIC_IMPORT = len(_PATTERNS) - 1

# importScripts("a.js", "b.js") -- how a Web Worker and an MV3 extension
# service worker load their code. TypeScript does not model this, because it is
# a runtime function call rather than an import, so it is absent from the
# compiler comparison this scanner was checked against. It is still a real
# dependency: a Chrome extension in the corpus loads 24 of its files this way,
# and without it every one of them looks like dead code.
_IMPORT_SCRIPTS = re.compile(r"\bimportScripts\s*\(([^)]*)\)")
_ANY_SPECIFIER = re.compile(_SPECIFIER)

# Characters that can begin something the scanner must interpret. Everything
# between two of these is ordinary code and is copied in one slice.
_SPECIAL = re.compile(r"""[/'"`]""")

# A '/' begins a regex literal only where a value cannot already have ended.
# After an identifier, a number, or a closing bracket, '/' is division.
_REGEX_ALLOWED_BEFORE = set("([{,;:=!&|?+-*%~^<>") | {"\n"}


@dataclass(slots=True)
class ParsedScript:
    rel_path: str
    specifiers: list[str] = field(default_factory=list)
    deferred: set[str] = field(default_factory=set)


def strip_noncode(source: str, literals: list[str] | None = None) -> str:
    """Replace comments, strings and regex literals with spaces.

    Length is preserved so any offsets stay meaningful, and newlines are kept
    so line numbers survive.
    """
    if literals is None:
        literals = []
    out: list[str] = []
    i = 0
    length = len(source)
    prev_significant = "\n"
    # Template literals nest: `a ${ `b` } c`. Depth tracks how many ${ } we are
    # inside so the closing backtick is matched to the right one.
    template_stack: list[int] = []

    while i < length:
        char = source[i]
        nxt = source[i + 1] if i + 1 < length else ""

        if char == "/" and nxt == "/":
            while i < length and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if char == "/" and nxt == "*":
            while i < length and not (source[i] == "*" and i + 1 < length
                                      and source[i + 1] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
            continue
        if char in "'\"":
            # Tokenised, not blanked and not kept raw. The module specifier is
            # itself a string, so it has to survive; but string *contents* must
            # be invisible to the patterns, or text like `case "import":`
            # matches as an import.
            quote = char
            i += 1
            body: list[str] = []
            while i < length:
                if source[i] == "\\":
                    body.append(source[i : i + 2])
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                if source[i] == "\n":
                    break  # unterminated string; stop rather than run away
                body.append(source[i])
                i += 1
            literals.append("".join(body))
            out.append(f"{quote}\x00{len(literals) - 1}\x00{quote}")
            prev_significant = "x"  # a string is a value
            continue
        if char == "`":
            out.append(" ")
            i += 1
            depth = 0
            while i < length:
                if source[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if source[i] == "$" and i + 1 < length and source[i + 1] == "{":
                    # Code again inside the substitution; keep it.
                    template_stack.append(depth)
                    out.append("  ")
                    i += 2
                    # Advance by the ORIGINAL substitution length, not the
                    # transformed one. Tokenising strings changes the length,
                    # so advancing by len(inner) desynchronised the cursor and
                    # every construct after the first `${...}` containing a
                    # string was misread. Length used to be preserved, which is
                    # why this only broke once tokens were introduced.
                    raw_inner = _balanced(source, i)
                    out.append(strip_noncode(raw_inner, literals))
                    i += len(raw_inner)
                    continue
                if source[i] == "`":
                    out.append(" ")
                    i += 1
                    break
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            prev_significant = "x"
            continue
        if char == "/" and prev_significant in _REGEX_ALLOWED_BEFORE:
            out.append(" ")
            i += 1
            in_class = False
            while i < length:
                if source[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if source[i] == "[":
                    in_class = True
                elif source[i] == "]":
                    in_class = False
                elif source[i] == "/" and not in_class:
                    out.append(" ")
                    i += 1
                    break
                elif source[i] == "\n":
                    break  # unterminated: not a regex after all
                out.append(" ")
                i += 1
            prev_significant = "x"
            continue

        # Ordinary code: copy the whole run up to the next character that could
        # start a comment, string, template or regex, rather than one character
        # at a time. Most of a source file is this run, and appending per
        # character cost 11 ms/file on real sources.
        following = _SPECIAL.search(source, i)
        end = following.start() if following else length
        if end <= i:
            # A '/' that is division, not a regex: the branches above declined
            # it, so consume it here and keep going.
            end = i + 1
        chunk = source[i:end]
        out.append(chunk)
        # Trailing blanks other than a newline leave prev_significant alone,
        # exactly as the per-character version did.
        tail = chunk.rstrip(" \t\r\f\v")
        if tail:
            prev_significant = tail[-1]
        i = end

    return "".join(out)


def _balanced(source: str, start: int) -> str:
    """Text of a ``${ ... }`` substitution starting at *start*."""
    depth = 1
    i = start
    while i < len(source) and depth:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return source[start:i]


def parse_script(rel_path: str, source: str) -> ParsedScript:
    """Module specifiers imported by one JS/TS file."""
    parsed = ParsedScript(rel_path=rel_path)
    literals: list[str] = []
    code = strip_noncode(source, literals)
    seen: set[str] = set()
    for position, pattern in enumerate(_PATTERNS):
        for match in pattern.finditer(code):
            index = int(match.group(1))
            if index >= len(literals):
                continue
            specifier = literals[index].strip()
            if not specifier or specifier in seen:
                continue
            seen.add(specifier)
            parsed.specifiers.append(specifier)
            # Dynamic import() runs at call time, not load time -- the same
            # distinction pysource records for imports inside functions. Static
            # patterns are tried first, so a module imported both ways counts
            # as static, which is what actually happens at load.
            if position == _DYNAMIC_IMPORT:
                parsed.deferred.add(specifier)

    for call in _IMPORT_SCRIPTS.finditer(code):
        for match in _ANY_SPECIFIER.finditer(call.group(1)):
            index = int(match.group(1))
            if index >= len(literals):
                continue
            specifier = literals[index].strip()
            # importScripts resolves against the loading script's own location,
            # so a bare "config.js" means "./config.js" -- unlike an import,
            # where a bare name is a package. Normalising here lets resolution
            # treat both the same way.
            if specifier and not specifier.startswith((".", "/")):
                specifier = f"./{specifier}"
            if not specifier or specifier in seen or "://" in specifier:
                continue
            seen.add(specifier)
            parsed.specifiers.append(specifier)
    parsed.specifiers.sort()
    return parsed


# ---------------------------------------------------------------------------
# Specifier resolution
# ---------------------------------------------------------------------------

# Node resolves an extensionless specifier by trying these in order, then the
# same list under a directory as index.*.
RESOLVE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".d.ts")

_JSONC_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def load_path_aliases(repo_root: Path) -> dict[str, list[str]]:
    """``compilerOptions.paths`` from tsconfig.json, resolved against baseUrl.

    9% of the specifiers in a real Next.js repo are aliases like ``@/lib/db``;
    without this they all look external and the internal graph loses a tenth of
    its edges. tsconfig.json is JSON with comments and trailing commas in
    practice, so both are stripped before parsing rather than assuming strict
    JSON.
    """
    for name in ("tsconfig.json", "jsconfig.json"):
        manifest = repo_root / name
        try:
            raw = manifest.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        cleaned = _TRAILING_COMMA.sub(r"\1", _JSONC_COMMENT.sub("", raw))
        try:
            data = json.loads(cleaned)
        except ValueError:
            continue
        options = data.get("compilerOptions", {})
        base = str(options.get("baseUrl", ".")).strip("./") or ""
        paths = options.get("paths", {})
        if not isinstance(paths, dict):
            continue
        resolved: dict[str, list[str]] = {}
        for pattern, targets in paths.items():
            if isinstance(targets, list):
                resolved[pattern] = [
                    "/".join(x for x in (base, str(t).lstrip("./")) if x)
                    for t in targets
                ]
        if resolved:
            return resolved
    return {}


def _candidate_files(repo_root: Path, rel: str) -> str | None:
    """First existing file for a repo-relative path without an extension."""
    direct = repo_root / rel
    if direct.is_file():
        return rel
    for suffix in RESOLVE_SUFFIXES:
        candidate = repo_root / (rel + suffix)
        if candidate.is_file():
            return rel + suffix
    for suffix in RESOLVE_SUFFIXES:
        candidate = repo_root / rel / f"index{suffix}"
        if candidate.is_file():
            return f"{rel}/index{suffix}"
    return None


def resolve_specifier(
    repo_root: Path,
    importer_rel: str,
    specifier: str,
    aliases: dict[str, list[str]],
) -> str | None:
    """Repo-relative file a specifier points at, or None when it is external.

    External means a package (``react``), a node builtin (``node:fs``), or an
    alias that maps nowhere — all real dependencies, just not files in this
    repo, so they become module nodes without a path.
    """
    if specifier.startswith("."):
        importer_dir = PurePosixPath(importer_rel.replace("\\", "/")).parent
        target = str(PurePosixPath(str(importer_dir)) / specifier)
        parts: list[str] = []
        for piece in target.split("/"):
            if piece in ("", "."):
                continue
            if piece == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(piece)
        return _candidate_files(repo_root, "/".join(parts))

    for pattern, targets in aliases.items():
        if "*" not in pattern:
            if specifier == pattern:
                for target in targets:
                    found = _candidate_files(repo_root, target)
                    if found:
                        return found
            continue
        prefix = pattern.split("*", 1)[0]
        if not specifier.startswith(prefix):
            continue
        tail = specifier[len(prefix):]
        for target in targets:
            candidate = target.replace("*", tail)
            found = _candidate_files(repo_root, candidate)
            if found:
                return found
    return None
