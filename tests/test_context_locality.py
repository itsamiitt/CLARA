"""
The session-start block belongs to the project the session is in.

Verified on a real store before the fix: nine findings saved while auditing
one repository filled eight of the ten belief slots in every OTHER project's
session start, and nothing marked them as somebody else's. Same defect family
the per-prompt recall matcher had; same locality rule fixes both, shared in
clara.fastpath.db.is_local.

Three layers, each tested: this project's memories rank first at any score,
foreign survivors carry a "[from another project]" label, and at most
FOREIGN_CAP of them appear at all — labels alone still let a busy neighbour
project consume most of the token budget.
"""

from __future__ import annotations

import time

from clara.fastpath import context, db

NOW = time.time()
HERE = "aaaa1111aaaa1111"
ELSEWHERE = "bbbb2222bbbb2222"


def _memory(subject, obj, *, repo, confidence=0.9, mem_id=None):
    return {
        "type": "belief",
        "content": {"subject": subject, "relation": "uses", "object": obj},
        "confidence": confidence,
        "metadata": {"repo_id": repo} if repo else {},
        "created_at": "2026-07-27",
        "updated_epoch": int(NOW),
        "memory_id": mem_id or f"{subject}-{obj}",
    }


class TestIsLocal:
    def test_same_repo_is_local(self) -> None:
        assert db.is_local(_memory("api", "redis", repo=HERE), HERE)

    def test_unstamped_is_local(self) -> None:
        assert db.is_local(_memory("api", "redis", repo=None), HERE)

    def test_other_repo_is_foreign(self) -> None:
        assert not db.is_local(_memory("api", "redis", repo=ELSEWHERE), HERE)

    def test_user_facts_follow_the_user(self) -> None:
        assert db.is_local(_memory("user", "pnpm", repo=ELSEWHERE), HERE)

    def test_no_current_repo_means_everything_is_local(self) -> None:
        assert db.is_local(_memory("api", "redis", repo=ELSEWHERE), None)


class TestRankLocality:
    def test_local_facts_outrank_foreign_at_any_score(self) -> None:
        memories = [
            _memory("their service", "their queue", repo=ELSEWHERE, confidence=1.0),
            _memory("our service", "our queue", repo=HERE, confidence=0.5),
        ]
        ranked = context.rank(memories, NOW, HERE)
        assert ranked[0]["content"]["subject"] == "our service"

    def test_foreign_survivors_are_marked_for_the_label(self) -> None:
        memories = [_memory("their service", "kafka", repo=ELSEWHERE)]
        ranked = context.rank(memories, NOW, HERE)
        assert ranked and ranked[0].get("_foreign") is True
        block = context.format_block(ranked)
        assert "[from another project]" in block

    def test_local_lines_carry_no_label(self) -> None:
        ranked = context.rank([_memory("api", "redis", repo=HERE)], NOW, HERE)
        block = context.format_block(ranked)
        assert "[from another project]" not in block

    def test_foreign_entries_are_capped(self) -> None:
        memories = [
            _memory(f"their thing {i}", f"tool {i}", repo=ELSEWHERE, mem_id=f"f{i}")
            for i in range(9)
        ] + [_memory("user", "pnpm", repo=ELSEWHERE)]
        ranked = context.rank(memories, NOW, HERE)
        foreign = [m for m in ranked if m.get("_foreign")]
        assert len(foreign) == context.FOREIGN_CAP
        # The user preference is not foreign and must not be caught by the cap.
        assert any(m["content"]["subject"] == "user" for m in ranked)

    def test_no_repo_given_keeps_the_pure_score_order(self) -> None:
        # The bridge exporter writes user-global files and passes no repo;
        # its output must stay exactly as before this feature existed.
        memories = [
            _memory("their service", "their queue", repo=ELSEWHERE, confidence=1.0),
            _memory("our service", "our queue", repo=HERE, confidence=0.5),
        ]
        ranked = context.rank(memories, NOW)
        assert ranked[0]["content"]["subject"] == "their service"
        assert not any(m.get("_foreign") for m in ranked)
