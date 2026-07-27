"""
Per-prompt memory recall (the UserPromptSubmit hook).

Session start injects memory once; a topic that first comes up mid-session got
nothing until this hook. Its stdout is model-visible on exit 0 — verified
against the hooks documentation before building, because this repo has already
been burned once by assumed hook semantics (Stop stdout, finding S1).

The property these tests defend hardest is silence: a recall block on every
prompt is noise that teaches the model to ignore all of them.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

from clara.fastpath import prompt_recall


def _seed(db_path, facts):
    from clara.integrations.local_memory import LocalMemory

    async def go():
        memory = await LocalMemory.create(str(db_path))
        for subject, relation, obj in facts:
            await memory.save(
                mem_type="belief", subject=subject, relation=relation, object=obj
            )
        await memory.close()

    asyncio.run(go())


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "proj").mkdir()
    _seed(tmp_path / "home" / "clara.db", [
        ("user", "uses", "postgres"),
        ("user", "prefers", "pnpm"),
        ("api", "runs_on", "fly.io"),
    ])
    return tmp_path


def recall(home, prompt, session="s1"):
    return prompt_recall.recall(prompt, str(home / "proj"), session)


class TestRecallHits:
    def test_matching_prompt_recalls_the_fact(self, home) -> None:
        block = recall(home, "why is the postgres database slow today?")
        assert block is not None
        assert "postgres" in block
        assert block.startswith("[MEMORY RECALL")

    def test_each_memory_shows_once_per_session(self, home) -> None:
        first = recall(home, "debug the postgres database connection")
        second = recall(home, "more postgres database questions here")
        assert first is not None
        assert second is None, "the same fact must not reappear every prompt"

    def test_a_different_session_sees_it_again(self, home) -> None:
        assert recall(home, "postgres database tuning", session="a") is not None
        assert recall(home, "postgres database tuning", session="b") is not None


class TestSilenceIsTheDefault:
    def test_unrelated_prompt_is_silent(self, home) -> None:
        assert recall(home, "refactor the parser into a state machine") is None

    def test_single_short_word_overlap_is_not_enough(self, home) -> None:
        # "uses" alone must not drag a belief in; that is the conservative
        # two-words-or-one-long-word rule.
        assert recall(home, "what uses more memory here today") is None

    def test_slash_commands_are_skipped(self, home) -> None:
        assert recall(home, "/clara:memories postgres database things") is None

    def test_tiny_prompts_are_skipped(self, home) -> None:
        assert recall(home, "postgres?") is None

    def test_no_store_is_silent(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CLARA_HOME", str(tmp_path / "nowhere"))
        assert prompt_recall.recall(
            "long enough prompt about postgres databases", str(tmp_path), "s"
        ) is None


class TestHookProcess:
    """The real entry point, driven exactly as Claude Code drives it."""

    def run_hook(self, home, payload):
        env = {**os.environ, "CLARA_HOME": str(home / "home")}
        return subprocess.run(
            [sys.executable, "-m", "clara.fastpath.prompt_recall"],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True, env=env, timeout=60,
        )

    def test_emits_utf8_block_and_exits_zero(self, home) -> None:
        done = self.run_hook(home, {
            "prompt": "postgres database performance question",
            "cwd": str(home / "proj"), "session_id": "proc1",
        })
        assert done.returncode == 0
        text = done.stdout.decode("utf-8")  # must be valid UTF-8
        assert "[MEMORY RECALL" in text

    def test_malformed_stdin_is_silent_success(self, home) -> None:
        env = {**os.environ, "CLARA_HOME": str(home / "home")}
        done = subprocess.run(
            [sys.executable, "-m", "clara.fastpath.prompt_recall"],
            input=b"not json at all", capture_output=True, env=env, timeout=60,
        )
        assert done.returncode == 0
        assert done.stdout == b""

    def test_no_match_emits_nothing_at_all(self, home) -> None:
        done = self.run_hook(home, {
            "prompt": "write a haiku about compilers for me",
            "cwd": str(home / "proj"), "session_id": "proc2",
        })
        assert done.returncode == 0
        assert done.stdout == b"", "silence must be byte-silence, not chatter"


class TestHookWiring:
    def test_registered_in_hooks_json(self) -> None:
        from pathlib import Path

        root = Path(__file__).parents[1]
        config = json.loads((root / "hooks" / "hooks.json").read_text("utf-8"))
        events = config["hooks"]
        assert "UserPromptSubmit" in events
        command = events["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert "prompt-recall" in command
