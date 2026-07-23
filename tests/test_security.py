"""Tests for secret detection (clara/security.py) and the save-path guards."""

from __future__ import annotations

import pytest

from clara import security
from clara.integrations.local_memory import LocalMemory


@pytest.fixture
async def memory(tmp_path):
    mem = await LocalMemory.create(str(tmp_path / "sec.db"))
    yield mem
    await mem.close()


class TestPatterns:
    @pytest.mark.parametrize(
        ("name", "sample"),
        [
            ("aws-access-key-id", "key AKIAIOSFODNN7EXAMPLE lives here"),
            ("api-key", "use sk-abcdefghijklmnopqrstuvwx123456"),
            ("github-token", "ghp_" + "a1B2" * 9),
            ("github-pat", "github_pat_" + "a1B2c3D4e5F6g7H8i9J0k1"),
            ("slack-token", "xoxb-1234567890-abcdefghij"),
            ("private-key", "-----BEGIN RSA PRIVATE KEY-----"),
            ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9P"),
            ("credential-assignment", "password: hunter2hunter2"),
            ("credential-assignment", "API_KEY=abcd1234efgh5678"),
        ],
    )
    def test_detects(self, name, sample):
        assert security.find_secret(sample) == name

    @pytest.mark.parametrize(
        "sample",
        [
            "the token bucket algorithm limits request rates",
            "rotate your api key monthly",
            "password requirements: 12 chars minimum",
            "user prefers pnpm over npm",
            "the secret to good soup is patience",
        ],
    )
    def test_prose_passes(self, sample):
        assert security.find_secret(sample) is None

    def test_redact_replaces_and_names(self):
        clean, names = security.redact("key AKIAIOSFODNN7EXAMPLE here")
        assert "AKIAIOSFODNN7EXAMPLE" not in clean
        assert "[REDACTED:aws-access-key-id]" in clean
        assert names == ["aws-access-key-id"]


class TestSavePathPolicy:
    async def test_reject_default(self, memory, monkeypatch):
        monkeypatch.delenv("CLARA_SECRET_POLICY", raising=False)
        with pytest.raises(ValueError, match="secret pattern"):
            await memory.save(
                mem_type="belief",
                subject="deploy",
                relation="uses",
                object="AKIAIOSFODNN7EXAMPLE",
            )

    async def test_off_stores_verbatim(self, memory, monkeypatch):
        monkeypatch.setenv("CLARA_SECRET_POLICY", "off")
        result = await memory.save(
            mem_type="belief",
            subject="deploy",
            relation="uses",
            object="AKIAIOSFODNN7EXAMPLE",
        )
        assert result["action"] == "saved"

    async def test_redact_mode(self, memory, monkeypatch):
        monkeypatch.setenv("CLARA_SECRET_POLICY", "redact")
        result = await memory.save(
            mem_type="belief",
            subject="deploy",
            relation="uses",
            object="key AKIAIOSFODNN7EXAMPLE",
        )
        assert result["action"] == "saved"
        found = await memory.search("deploy", top_k=3)
        assert "AKIAIOSFODNN7EXAMPLE" not in str(found)

    async def test_clean_content_unaffected(self, memory):
        result = await memory.save(
            mem_type="belief", subject="user", relation="prefers", object="tabs"
        )
        assert result["action"] == "saved"


class TestSizeCaps:
    async def test_oversize_content_rejected(self, memory):
        with pytest.raises(ValueError, match="bytes"):
            await memory.save(
                mem_type="belief",
                subject="dump",
                relation="contains",
                object="x" * 20000,
            )

    async def test_too_many_tags_rejected(self, memory):
        with pytest.raises(ValueError, match="tags"):
            await memory.save(
                mem_type="belief",
                subject="a",
                relation="b",
                object="c",
                tags=[f"t{i}" for i in range(100)],
            )

    async def test_env_cap_override(self, memory, monkeypatch):
        monkeypatch.setattr(
            "clara.integrations.local_memory._MAX_CONTENT_BYTES", 64
        )
        with pytest.raises(ValueError, match="cap 64"):
            await memory.save(
                mem_type="belief", subject="a", relation="b", object="c" * 100
            )
