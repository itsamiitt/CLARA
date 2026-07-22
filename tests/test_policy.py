"""Tests for the repo policy loader (clara/policy.py)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clara.policy import Policy, load_policy

_EXAMPLE = Path(__file__).parents[1] / "clara.yml.example"


def _write(tmp_path, text: str) -> None:
    (tmp_path / "clara.yml").write_text(text, encoding="utf-8")


class TestPolicyDefaults:
    def test_absent_file_yields_defaults(self, tmp_path):
        policy = load_policy(tmp_path)
        assert policy.version == 1
        assert policy.archive_dir == "docs/archive"
        assert policy.tiers == {
            "T1": ["docs/adr/**", "CODING_STANDARDS.md", "CLAUDE.md"],
            "T3": ["notes/**", "**/*brainstorm*.md"],
        }
        assert policy.types == {"plan": ["docs/plans/**", "**/PLAN*.md"]}
        assert policy.stale_after_days == {"plan": 60, "brainstorm": 14}
        assert policy.exclude == ["vendor/**"]
        assert policy.archive_mode == "git-mv"
        assert policy.aliases == {"pg": "postgresql"}

    def test_empty_file_yields_defaults(self, tmp_path):
        _write(tmp_path, "")
        assert load_policy(tmp_path) == Policy()

    def test_example_file_matches_defaults(self, tmp_path):
        shutil.copyfile(_EXAMPLE, tmp_path / "clara.yml")
        assert load_policy(tmp_path) == Policy()


class TestPolicyParsing:
    def test_partial_file_overrides_only_given_keys(self, tmp_path):
        _write(tmp_path, "archive_dir: custom/dir\n")
        policy = load_policy(tmp_path)
        assert policy.archive_dir == "custom/dir"
        assert policy.tiers == Policy().tiers
        assert policy.stale_after_days == Policy().stale_after_days

    def test_full_override_roundtrip(self, tmp_path):
        _write(
            tmp_path,
            "version: 2\n"
            "archive_dir: attic\n"
            "tiers:\n"
            '  A: ["a/**"]\n'
            "types:\n"
            '  spec: ["specs/**"]\n'
            "stale_after_days: { spec: 5 }\n"
            'exclude: ["gen/**"]\n'
            "archive_mode: copy\n"
            "aliases: { js: javascript }\n",
        )
        policy = load_policy(tmp_path)
        assert policy.version == 2
        assert policy.archive_dir == "attic"
        assert policy.tiers == {"A": ["a/**"]}
        assert policy.types == {"spec": ["specs/**"]}
        assert policy.stale_after_days == {"spec": 5}
        assert policy.exclude == ["gen/**"]
        assert policy.archive_mode == "copy"
        assert policy.aliases == {"js": "javascript"}

    def test_tiers_replace_not_merge(self, tmp_path):
        _write(tmp_path, 'tiers:\n  T1: ["x/**"]\n')
        assert load_policy(tmp_path).tiers == {"T1": ["x/**"]}

    def test_unknown_top_level_key_ignored(self, tmp_path):
        _write(tmp_path, "frobnicate: 1\narchive_dir: d\n")
        assert load_policy(tmp_path).archive_dir == "d"

    def test_flow_style_mappings_parse(self, tmp_path):
        _write(tmp_path, "stale_after_days: { plan: 30, brainstorm: 7 }\n")
        assert load_policy(tmp_path).stale_after_days == {"plan": 30, "brainstorm": 7}


class TestPolicyErrors:
    def test_malformed_yaml_raises_valueerror(self, tmp_path):
        _write(tmp_path, "tiers: [unclosed\n")
        with pytest.raises(ValueError, match="clara.yml"):
            load_policy(tmp_path)

    def test_non_mapping_top_level_raises(self, tmp_path):
        _write(tmp_path, "- a\n- b\n")
        with pytest.raises(ValueError, match="top level must be a mapping"):
            load_policy(tmp_path)

    def test_wrong_type_stale_days_raises(self, tmp_path):
        _write(tmp_path, "stale_after_days: { plan: sixty }\n")
        with pytest.raises(ValueError, match="stale_after_days"):
            load_policy(tmp_path)

    def test_bool_stale_days_raises(self, tmp_path):
        _write(tmp_path, "stale_after_days: { plan: true }\n")
        with pytest.raises(ValueError, match="stale_after_days"):
            load_policy(tmp_path)

    def test_wrong_type_tier_value_raises(self, tmp_path):
        _write(tmp_path, 'tiers: { T1: "notalist" }\n')
        with pytest.raises(ValueError, match="tiers"):
            load_policy(tmp_path)

    def test_empty_archive_mode_raises(self, tmp_path):
        _write(tmp_path, 'archive_mode: ""\n')
        with pytest.raises(ValueError, match="archive_mode"):
            load_policy(tmp_path)
