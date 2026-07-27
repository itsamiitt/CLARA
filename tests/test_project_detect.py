"""Tests for project detection (clara/project/detect.py).

The doctrine under test is "claim less, never wrongly": a fact must be backed
by something a manifest actually states. Detection runs on the session-start
path, so it must also never raise, whatever it is pointed at.
"""

from __future__ import annotations

import json

import pytest

from clara.project import detect_project


def _write(root, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _pkg(root, **manifest) -> None:
    _write(root, "package.json", json.dumps(manifest))


class TestPackageManager:
    @pytest.mark.parametrize(
        "lockfile, expected",
        [
            ("bun.lock", "bun"),
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("package-lock.json", "npm"),
            ("uv.lock", "uv"),
            ("poetry.lock", "poetry"),
            ("Cargo.lock", "cargo"),
        ],
    )
    def test_lockfile_identifies_manager(self, tmp_path, lockfile, expected):
        _write(tmp_path, lockfile, "")
        profile = detect_project(tmp_path)
        assert expected in profile.by_category("package_manager")

    def test_package_manager_field_is_read_without_the_version(self, tmp_path):
        _pkg(tmp_path, name="x", packageManager="bun@1.3.14")
        # The pinned version churns constantly; it is not durable knowledge.
        assert detect_project(tmp_path).by_category("package_manager") == ["bun"]


class TestNode:
    def test_detects_frameworks_and_tooling(self, tmp_path):
        _pkg(
            tmp_path,
            name="app",
            dependencies={"react": "^18", "express": "^4", "pg": "^8"},
            devDependencies={"vite": "^5", "vitest": "^1", "typescript": "^5"},
        )
        profile = detect_project(tmp_path)
        assert profile.name == "app"
        assert set(profile.by_category("framework")) == {"react", "express"}
        assert profile.by_category("build_tool") == ["vite"]
        assert profile.by_category("test_framework") == ["vitest"]
        assert profile.by_category("database") == ["postgresql"]
        assert "typescript" in profile.by_category("language")

    def test_workspaces_mark_a_monorepo(self, tmp_path):
        _pkg(tmp_path, name="root", workspaces=["apps/*", "packages/*"])
        profile = detect_project(tmp_path)
        assert profile.is_monorepo
        assert profile.workspaces == ["apps/*", "packages/*"]

    def test_workspaces_object_form(self, tmp_path):
        _pkg(tmp_path, name="root", workspaces={"packages": ["libs/*"]})
        assert detect_project(tmp_path).workspaces == ["libs/*"]

    def test_unknown_dependencies_are_not_claimed(self, tmp_path):
        _pkg(tmp_path, name="app", dependencies={"left-pad": "^1"})
        profile = detect_project(tmp_path)
        assert profile.by_category("framework") == []

    def test_javascript_not_claimed_without_dependencies(self, tmp_path):
        _pkg(tmp_path, name="empty")
        assert detect_project(tmp_path).by_category("language") == []


class TestPython:
    def test_pyproject_dependencies_and_extras(self, tmp_path):
        _write(
            tmp_path,
            "pyproject.toml",
            """
[project]
name = "svc"
dependencies = ["fastapi>=0.1", "sqlalchemy>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
build-backend = "setuptools.build_meta"
""",
        )
        profile = detect_project(tmp_path)
        assert profile.name == "svc"
        assert profile.by_category("language") == ["python"]
        assert profile.by_category("framework") == ["fastapi"]
        assert profile.by_category("database_tool") == ["sqlalchemy"]
        # Extras are a separate table and were previously missed entirely.
        assert "pytest" in profile.by_category("test_framework")
        assert "setuptools" in profile.by_category("build_tool")

    def test_tool_table_is_evidence_even_without_a_dependency(self, tmp_path):
        # ruff/mypy are usually installed by CI, not declared as dependencies.
        _write(
            tmp_path,
            "pyproject.toml",
            '[project]\nname = "x"\n\n[tool.ruff]\n\n[tool.mypy]\n',
        )
        profile = detect_project(tmp_path)
        assert profile.by_category("linter") == ["ruff"]
        assert profile.by_category("type_checker") == ["mypy"]

    def test_requirements_txt(self, tmp_path):
        _write(tmp_path, "requirements.txt", "django==5.0\n# comment\n-e .\nredis>=5\n")
        profile = detect_project(tmp_path)
        assert profile.by_category("framework") == ["django"]
        assert profile.by_category("database") == ["redis"]


class TestOtherSignals:
    @pytest.mark.parametrize(
        "filename, language",
        [("go.mod", "go"), ("Cargo.toml", "rust"), ("Gemfile", "ruby")],
    )
    def test_language_manifests(self, tmp_path, filename, language):
        _write(tmp_path, filename, "")
        assert language in detect_project(tmp_path).by_category("language")

    def test_infrastructure_and_ci(self, tmp_path):
        _write(tmp_path, "Dockerfile", "FROM scratch")
        _write(tmp_path, ".github/workflows/ci.yml", "on: push")
        profile = detect_project(tmp_path)
        assert "docker" in profile.by_category("infrastructure")
        assert "github actions" in profile.by_category("ci")


class TestRobustness:
    """Detection sits on the session-start path: it must never raise."""

    def test_missing_root(self, tmp_path):
        assert detect_project(tmp_path / "nope").facts == []

    def test_empty_directory(self, tmp_path):
        assert detect_project(tmp_path).facts == []

    def test_malformed_package_json(self, tmp_path):
        _write(tmp_path, "package.json", "{ not json")
        assert detect_project(tmp_path).facts == []

    def test_package_json_that_is_not_an_object(self, tmp_path):
        _write(tmp_path, "package.json", "[1, 2, 3]")
        assert detect_project(tmp_path).facts == []

    def test_malformed_pyproject_still_reports_python(self, tmp_path):
        _write(tmp_path, "pyproject.toml", "this is not toml [[[")
        # The file existing is itself evidence; nothing further is claimed.
        assert detect_project(tmp_path).by_category("language") == ["python"]

    def test_every_fact_carries_evidence(self, tmp_path):
        _pkg(tmp_path, name="app", dependencies={"react": "^18"})
        for fact in detect_project(tmp_path).facts:
            assert fact.evidence, f"{fact.category}={fact.value} has no evidence"


class TestAgainstThisRepository:
    """Detection run against CLARA itself — a real manifest, not a fixture."""

    def test_detects_clara(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        profile = detect_project(root)
        assert profile.name == "clara-memory"
        assert profile.by_category("language") == ["python"]
        assert "sqlalchemy" in profile.by_category("database_tool")
        assert "pytest" in profile.by_category("test_framework")
        assert "ruff" in profile.by_category("linter")
        assert "mypy" in profile.by_category("type_checker")
        assert "github actions" in profile.by_category("ci")
        assert not profile.is_monorepo
