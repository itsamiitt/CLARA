from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clara.agent import ClaraMemory


class _FakeBackend:
    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, text: str) -> list[float]:
        return [0.0] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


class TestInstallationDefaults:
    @pytest.mark.asyncio
    async def test_create_uses_file_backed_sqlite_by_default(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        with patch("clara.agent._create_backend", return_value=_FakeBackend()), \
             patch("clara.agent.FactExtractor") as mock_extractor:
            mock_extractor.return_value = MagicMock()
            agent = await ClaraMemory.create(start_scheduler=False)

        assert agent is not None
        await agent.close()
        assert (tmp_path / "clara.db").exists()
