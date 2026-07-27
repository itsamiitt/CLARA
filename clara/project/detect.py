"""
CLARA — project detection from manifests.

Pure and stdlib-only: no network, no subprocess, no third-party parsers. This
runs on the session-start path, so import cost and wall time both matter, and
it must never be the reason a session fails to start.

Doctrine, mirroring ``clara.extraction.heuristic``: **claim less, never
wrongly.** Every fact points at the file and key that proved it. A dependency
that merely *appears* in a lockfile transitively is not evidence that the
project uses that framework, so only declared manifests are read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Cap the work: a manifest larger than this is pathological and not worth
# parsing on the session-start path.
_MAX_MANIFEST_BYTES = 2_000_000

# Package managers, strongest evidence first. A lockfile is a stronger claim
# than a field, because it reflects what was actually run.
_LOCKFILES: list[tuple[str, str]] = [
    ("bun.lock", "bun"),
    ("bun.lockb", "bun"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
    ("Cargo.lock", "cargo"),
    ("go.sum", "go"),
    ("composer.lock", "composer"),
    ("Gemfile.lock", "bundler"),
]

# npm package -> (category, canonical name). Only unambiguous, well-known
# packages belong here; anything requiring a judgement call is left out.
_NPM_SIGNALS: dict[str, tuple[str, str]] = {
    # frameworks
    "react": ("framework", "react"),
    "vue": ("framework", "vue"),
    "svelte": ("framework", "svelte"),
    "@angular/core": ("framework", "angular"),
    "next": ("framework", "next.js"),
    "nuxt": ("framework", "nuxt"),
    "@remix-run/react": ("framework", "remix"),
    "astro": ("framework", "astro"),
    "express": ("framework", "express"),
    "fastify": ("framework", "fastify"),
    "@nestjs/core": ("framework", "nestjs"),
    "hono": ("framework", "hono"),
    "koa": ("framework", "koa"),
    "electron": ("framework", "electron"),
    "react-native": ("framework", "react native"),
    # build tooling
    "vite": ("build_tool", "vite"),
    "webpack": ("build_tool", "webpack"),
    "rollup": ("build_tool", "rollup"),
    "esbuild": ("build_tool", "esbuild"),
    "parcel": ("build_tool", "parcel"),
    "turbo": ("build_tool", "turborepo"),
    "nx": ("build_tool", "nx"),
    "typescript": ("language", "typescript"),
    # test tooling
    "vitest": ("test_framework", "vitest"),
    "jest": ("test_framework", "jest"),
    "mocha": ("test_framework", "mocha"),
    "@playwright/test": ("test_framework", "playwright"),
    "cypress": ("test_framework", "cypress"),
    "@testing-library/react": ("test_framework", "testing-library"),
    # data / infra
    "prisma": ("database_tool", "prisma"),
    "@prisma/client": ("database_tool", "prisma"),
    "drizzle-orm": ("database_tool", "drizzle"),
    "typeorm": ("database_tool", "typeorm"),
    "mongoose": ("database_tool", "mongoose"),
    "pg": ("database", "postgresql"),
    "mysql2": ("database", "mysql"),
    "redis": ("database", "redis"),
    "ioredis": ("database", "redis"),
    "better-sqlite3": ("database", "sqlite"),
    # styling
    "tailwindcss": ("styling", "tailwind"),
    "styled-components": ("styling", "styled-components"),
    # linting
    "eslint": ("linter", "eslint"),
    "@biomejs/biome": ("linter", "biome"),
    "prettier": ("formatter", "prettier"),
}

# Python distribution name -> (category, canonical name).
_PY_SIGNALS: dict[str, tuple[str, str]] = {
    "django": ("framework", "django"),
    "flask": ("framework", "flask"),
    "fastapi": ("framework", "fastapi"),
    "starlette": ("framework", "starlette"),
    "sanic": ("framework", "sanic"),
    "tornado": ("framework", "tornado"),
    "pyramid": ("framework", "pyramid"),
    "litestar": ("framework", "litestar"),
    "sqlalchemy": ("database_tool", "sqlalchemy"),
    "alembic": ("database_tool", "alembic"),
    "psycopg2": ("database", "postgresql"),
    "psycopg": ("database", "postgresql"),
    "asyncpg": ("database", "postgresql"),
    "aiosqlite": ("database", "sqlite"),
    "redis": ("database", "redis"),
    "pymongo": ("database", "mongodb"),
    "pytest": ("test_framework", "pytest"),
    "unittest2": ("test_framework", "unittest"),
    "ruff": ("linter", "ruff"),
    "flake8": ("linter", "flake8"),
    "pylint": ("linter", "pylint"),
    "mypy": ("type_checker", "mypy"),
    "black": ("formatter", "black"),
    "pandas": ("library", "pandas"),
    "numpy": ("library", "numpy"),
    "torch": ("library", "pytorch"),
    "celery": ("library", "celery"),
    "pydantic": ("library", "pydantic"),
}

# Marker file -> language, for languages whose presence is proven by a manifest.
_LANGUAGE_MANIFESTS: list[tuple[str, str]] = [
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("build.gradle.kts", "kotlin"),
    ("Gemfile", "ruby"),
    ("composer.json", "php"),
    ("pubspec.yaml", "dart"),
    ("Package.swift", "swift"),
    ("mix.exs", "elixir"),
]


@dataclass(frozen=True, slots=True)
class ProjectFact:
    """One claim about the project, with the evidence that supports it.

    ``category``/``value`` are the claim ("framework" / "react"); ``evidence``
    names the file and key it came from so a human can check it.
    """

    category: str
    value: str
    evidence: str
    confidence: float = 0.9


@dataclass(slots=True)
class ProjectProfile:
    """Everything detected about one project root."""

    root: str
    name: str | None = None
    facts: list[ProjectFact] = field(default_factory=list)
    workspaces: list[str] = field(default_factory=list)

    @property
    def is_monorepo(self) -> bool:
        return bool(self.workspaces)

    def by_category(self, category: str) -> list[str]:
        """Distinct values claimed for *category*, in detection order."""
        seen: list[str] = []
        for fact in self.facts:
            if fact.category == category and fact.value not in seen:
                seen.append(fact.value)
        return seen

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe view for tools and the CLI."""
        categories: dict[str, list[str]] = {}
        for fact in self.facts:
            categories.setdefault(fact.category, [])
            if fact.value not in categories[fact.category]:
                categories[fact.category].append(fact.value)
        return {
            "root": self.root,
            "name": self.name,
            "monorepo": self.is_monorepo,
            "workspaces": self.workspaces,
            "detected": categories,
            "fact_count": len(self.facts),
        }


def _read_text(path: Path) -> str | None:
    """File contents, or ``None`` if unreadable/absent/too large."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


_PEP508_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_name(spec: str) -> str | None:
    """Distribution name from a PEP 508 requirement string."""
    match = _PEP508_NAME.match(spec)
    return match.group(1).lower().replace("_", "-") if match else None


def _detect_package_manager(root: Path, facts: list[ProjectFact]) -> None:
    for filename, manager in _LOCKFILES:
        if (root / filename).is_file():
            facts.append(
                ProjectFact("package_manager", manager, filename, confidence=0.95)
            )
            return


def _detect_node(root: Path, profile: ProjectProfile) -> None:
    manifest = _read_json(root / "package.json")
    if manifest is None:
        return
    facts = profile.facts

    name = manifest.get("name")
    if isinstance(name, str) and name.strip():
        profile.name = name.strip()

    # `packageManager: "bun@1.3.14"` is an explicit declaration; record the
    # tool, not the pinned version, which changes too often to be a memory.
    declared = manifest.get("packageManager")
    if isinstance(declared, str) and "@" in declared:
        tool = declared.split("@", 1)[0].strip()
        if tool:
            facts.append(
                ProjectFact(
                    "package_manager", tool, "package.json:packageManager", 0.95
                )
            )

    workspaces = manifest.get("workspaces")
    globs: list[str] = []
    if isinstance(workspaces, list):
        globs = [w for w in workspaces if isinstance(w, str)]
    elif isinstance(workspaces, dict):
        raw = workspaces.get("packages")
        if isinstance(raw, list):
            globs = [w for w in raw if isinstance(w, str)]
    if globs:
        profile.workspaces = globs
        facts.append(
            ProjectFact("structure", "monorepo", "package.json:workspaces", 0.95)
        )

    deps: dict[str, Any] = {}
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        section = manifest.get(key)
        if isinstance(section, dict):
            for dep in section:
                deps.setdefault(dep, key)

    if deps:
        # JavaScript is implied by having a package.json with dependencies;
        # TypeScript is only claimed when actually declared.
        facts.append(ProjectFact("language", "javascript", "package.json", 0.8))
    for dep, section in deps.items():
        signal = _NPM_SIGNALS.get(dep)
        if signal is not None:
            category, value = signal
            facts.append(
                ProjectFact(category, value, f"package.json:{section}:{dep}", 0.9)
            )

    scripts = manifest.get("scripts")
    if isinstance(scripts, dict) and scripts:
        for script in ("build", "test", "dev", "start", "lint"):
            if isinstance(scripts.get(script), str):
                facts.append(
                    ProjectFact("script", script, f"package.json:scripts:{script}", 0.95)
                )


# pyproject is read without tomllib so the same code path works on 3.10.
# Only the few fields below are needed, and each pattern is anchored to its
# table header so a value from another section cannot be misattributed.
_PY_NAME_RE = re.compile(r"^\s*name\s*=\s*[\"']([^\"']+)[\"']", re.M)
_PY_DEPS_BLOCK_RE = re.compile(r"^\s*dependencies\s*=\s*\[(.*?)\]", re.M | re.S)
_PY_STRING_RE = re.compile(r"[\"']([^\"']+)[\"']")
_PY_BUILD_BACKEND_RE = re.compile(
    r"^\s*build-backend\s*=\s*[\"']([^\"']+)[\"']", re.M
)
# Extras live under their own table as `name = [...]`, so the runtime
# `dependencies = [...]` pattern above does not see them. Dev tooling
# (pytest, ruff, mypy) is usually declared there and is worth knowing.
_PY_EXTRAS_TABLE_RE = re.compile(
    r"^\[project\.optional-dependencies\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S
)
_PY_EXTRA_ARRAY_RE = re.compile(r"^\s*[A-Za-z0-9_-]+\s*=\s*\[(.*?)\]", re.M | re.S)
# A `[tool.ruff]` section is proof the project uses ruff even when the tool is
# not a declared dependency (it is commonly installed by CI or pipx instead).
_PY_TOOL_TABLE_RE = re.compile(r"^\[tool\.([A-Za-z0-9_-]+)", re.M)
_PY_TOOL_SIGNALS: dict[str, tuple[str, str]] = {
    "ruff": ("linter", "ruff"),
    "mypy": ("type_checker", "mypy"),
    "pyright": ("type_checker", "pyright"),
    "black": ("formatter", "black"),
    "isort": ("formatter", "isort"),
    "pytest": ("test_framework", "pytest"),
    "coverage": ("test_framework", "coverage"),
    "poetry": ("package_manager", "poetry"),
    "hatch": ("build_tool", "hatch"),
    "setuptools": ("build_tool", "setuptools"),
}


def _detect_python(root: Path, profile: ProjectProfile) -> None:
    facts = profile.facts
    pyproject = _read_text(root / "pyproject.toml")
    requirements = _read_text(root / "requirements.txt")
    if pyproject is None and requirements is None:
        if not (root / "setup.py").is_file():
            return
        facts.append(ProjectFact("language", "python", "setup.py", 0.9))
        return

    facts.append(
        ProjectFact(
            "language", "python", "pyproject.toml" if pyproject else "requirements.txt", 0.95
        )
    )

    specs: list[tuple[str, str]] = []
    if pyproject is not None:
        if profile.name is None:
            match = _PY_NAME_RE.search(pyproject)
            if match:
                profile.name = match.group(1)
        backend = _PY_BUILD_BACKEND_RE.search(pyproject)
        if backend:
            tool = backend.group(1).split(".", 1)[0]
            facts.append(
                ProjectFact("build_tool", tool, "pyproject.toml:build-backend", 0.95)
            )
        for block in _PY_DEPS_BLOCK_RE.findall(pyproject):
            for spec in _PY_STRING_RE.findall(block):
                specs.append((spec, "pyproject.toml"))
        for table in _PY_EXTRAS_TABLE_RE.findall(pyproject):
            for block in _PY_EXTRA_ARRAY_RE.findall(table):
                for spec in _PY_STRING_RE.findall(block):
                    specs.append((spec, "pyproject.toml:optional-dependencies"))
        for tool in _PY_TOOL_TABLE_RE.findall(pyproject):
            signal = _PY_TOOL_SIGNALS.get(tool.lower())
            if signal is not None:
                category, value = signal
                facts.append(
                    ProjectFact(category, value, f"pyproject.toml:[tool.{tool}]", 0.9)
                )
    if requirements is not None:
        for line in requirements.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                specs.append((line, "requirements.txt"))

    for spec, source in specs:
        dep = _requirement_name(spec)
        if dep is None:
            continue
        signal = _PY_SIGNALS.get(dep)
        if signal is not None:
            category, value = signal
            facts.append(ProjectFact(category, value, f"{source}:{dep}", 0.9))


def _detect_other_languages(root: Path, facts: list[ProjectFact]) -> None:
    for filename, language in _LANGUAGE_MANIFESTS:
        if (root / filename).is_file():
            facts.append(ProjectFact("language", language, filename, 0.95))


def _detect_containers(root: Path, facts: list[ProjectFact]) -> None:
    for filename, value in (
        ("Dockerfile", "docker"),
        ("docker-compose.yml", "docker compose"),
        ("docker-compose.yaml", "docker compose"),
    ):
        if (root / filename).is_file():
            facts.append(ProjectFact("infrastructure", value, filename, 0.95))
    if (root / ".github" / "workflows").is_dir():
        facts.append(ProjectFact("ci", "github actions", ".github/workflows", 0.95))


def detect_project(root: str | Path) -> ProjectProfile:
    """Build a :class:`ProjectProfile` for the repository at *root*.

    Never raises: an unreadable or unparseable manifest yields fewer facts,
    never an error. Callers run this on the session-start path.
    """
    root_path = Path(root)
    profile = ProjectProfile(root=str(root_path))
    if not root_path.is_dir():
        return profile

    _detect_package_manager(root_path, profile.facts)
    _detect_node(root_path, profile)
    _detect_python(root_path, profile)
    _detect_other_languages(root_path, profile.facts)
    _detect_containers(root_path, profile.facts)
    return profile
