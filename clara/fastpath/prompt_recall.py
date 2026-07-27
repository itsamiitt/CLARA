"""
CLARA fastpath — per-prompt memory recall (UserPromptSubmit hook).

Session start injects memory once; a topic that first comes up twenty minutes
in gets nothing. This hook closes that gap: on every user prompt it looks the
prompt's words up in the store and, when something genuinely matches, prints
a short block that Claude Code adds to the model's context (UserPromptSubmit
stdout is model-visible on exit 0 — verified against the hooks documentation,
same mechanism as SessionStart).

Discipline, in order of importance:

* **Silence is the default.** Most prompts match nothing and must emit
  nothing — a recall block on every prompt is noise that trains the model to
  ignore all of them. Matching is conservative on purpose: two distinct
  content words, or one store-rare word from the memory's naming fields.
  Facts stamped by a different repository clear a higher bar still, so one
  project's findings do not wander into another's sessions uninvited.
* **Never blocks, never writes the store.** Same contract as the session
  hook: exit 0 on every path, read-only connection, 3 s busy timeout,
  stdlib only.
* **Each memory is shown once per session.** A flags file keeps the ids
  already recalled; without it the same fact would reappear on every related
  prompt. The session-start block's items are not seeded into that file, so
  one of them can repeat once here — known, accepted for now.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from clara.fastpath import db
from clara.fastpath.context import _make_stdout_lossy, sanitize

MAX_HITS = 3
_MIN_PROMPT_LEN = 12
# A token is "rare" when this few stored memories mention it. Rarity, not
# length, is what makes a single-token match meaningful: "pnpm" (4 chars,
# probably one memory) identifies its fact outright, while "database"
# (8 chars, half the store) identifies nothing -- the original length-only
# rule had that exactly backwards, and missed a lone-"pnpm" prompt.
_RARE_DOC_COUNT = 3
_MIN_SINGLE_TOKEN = 4

# Words that match everything and mean nothing for recall.
_STOPWORDS = frozenset(
    """the and for with that this from have has are was were what when where
    which how why can could should would will just like want need make made
    using use used you your our their there here about into over under then
    than them they its his her not but all any some more most other only
    also very much many out get got run new now let see fix add set does
    please help write code file files change changes update test tests
    error work works working""".split()  # noqa: SIM905
)


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_\-\.]{2,}", text.lower())
        if word not in _STOPWORDS
    }


# Fields that NAME what a memory is about. A lone-token match may qualify
# only against these: "pnpm" in an object identifies its fact, while a verb
# like "uses" in the relation matches half the store and identifies nothing.
_IDENTIFIER_FIELDS = (
    "subject", "object", "name", "entity_type", "domain", "event_type"
)


def _memory_text(memory: dict[str, object]) -> str:
    content = memory.get("content")
    parts: list[str] = []
    if isinstance(content, dict):
        for value in content.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, dict)):
                parts.append(json.dumps(value))
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        tags = metadata.get("tags")
        if isinstance(tags, list):
            parts.extend(str(tag) for tag in tags)
    return " ".join(parts)


def _identifier_tokens(memory: dict[str, object]) -> set[str]:
    content = memory.get("content")
    parts: list[str] = []
    if isinstance(content, dict):
        for fieldname in _IDENTIFIER_FIELDS:
            value = content.get(fieldname)
            if isinstance(value, str):
                parts.append(value)
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        tags = metadata.get("tags")
        if isinstance(tags, list):
            parts.extend(str(tag) for tag in tags)
    return _tokens(" ".join(parts))


def _doc_frequency(memory_token_sets: list[set[str]]) -> dict[str, int]:
    """How many memories mention each token — the store's own rarity signal."""
    frequency: dict[str, int] = {}
    for tokens in memory_token_sets:
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def _matches(
    prompt_tokens: set[str],
    memory_tokens: set[str],
    identifier_tokens: set[str],
    frequency: dict[str, int],
    *,
    foreign: bool,
) -> int:
    """Conservative overlap score; 0 means "do not show".

    Qualifying overlap: two distinct content words anywhere in the memory, or
    a single word that both NAMES the memory (subject/object/name/tags, never
    the relation — "uses" matches half the store and identifies nothing) and
    is rare across the store. Rarity, not length: "pnpm" in one memory
    identifies its fact; "database" in thirty identifies none of them.

    Memories stamped with a *different* repository's id need both: two hits
    AND a rare identifying one. A generic overlap is enough to recall this
    project's own facts, not enough to drag another project's findings into
    the conversation uninvited. In a near-empty store every token is rare and
    cross-project facts surface on one strong name — the right behaviour at
    that scale, where there is nothing to drown in.
    """
    hits = prompt_tokens & memory_tokens
    if not hits:
        return 0
    rare_naming_hits = [
        hit for hit in (prompt_tokens & identifier_tokens)
        if frequency.get(hit, 0) <= _RARE_DOC_COUNT
        and len(hit) >= _MIN_SINGLE_TOKEN
    ]
    if foreign:
        qualified = len(hits) >= 2 and bool(rare_naming_hits)
    else:
        qualified = len(hits) >= 2 or bool(rare_naming_hits)
    if not qualified:
        return 0
    return len(hits) + len(rare_naming_hits)


def _render(memory: dict[str, object]) -> str:
    content = memory.get("content")
    content = content if isinstance(content, dict) else {}
    kind = str(memory.get("type", ""))
    raw_confidence = memory.get("confidence", 0.5)
    confidence = raw_confidence if isinstance(raw_confidence, float) else 0.5
    if kind == "belief":
        subject = sanitize(content.get("subject"))
        relation = sanitize(content.get("relation"))
        obj = sanitize(content.get("object"))
        negation = " (negated)" if content.get("is_negation") else ""
        return f"- {subject} {relation} {obj}{negation} (confidence: {confidence:.2f})"
    label = sanitize(
        content.get("name") or content.get("subject") or content.get("description")
    )
    detail = sanitize(content.get("description") or "")
    tail = f" — {detail}" if detail and detail != label else ""
    return f"- [{kind}] {label}{tail}"


def _recalled_file(session_id: str) -> Path:
    base = Path(os.environ.get("CLARA_HOME") or Path.home() / ".clara")
    return base / "session-flags" / f"{session_id}.recalled"


def _already_shown(session_id: str) -> set[str]:
    if not session_id:
        return set()
    try:
        return set(
            _recalled_file(session_id).read_text(encoding="utf-8").split()
        )
    except OSError:
        return set()


def _record_shown(session_id: str, memory_ids: list[str]) -> None:
    if not session_id or not memory_ids:
        return
    try:
        path = _recalled_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(memory_ids) + "\n")
    except OSError:
        pass  # dedupe is best-effort; a repeat beats a crash


def recall(prompt: str, cwd: str, session_id: str) -> str | None:
    """The recall block for *prompt*, or None when there is nothing to say."""
    if len(prompt) < _MIN_PROMPT_LEN or prompt.lstrip().startswith("/"):
        return None
    prompt_tokens = _tokens(prompt)
    if not prompt_tokens:
        return None

    db_path, current_repo = db.resolve_store(cwd)
    if db_path is None:
        return None
    conn = db.open_store(db_path)
    if conn is None:
        return None
    try:
        memories = db.fetch_active(conn)
    except Exception:  # noqa: BLE001 — recall must never block a prompt
        return None
    finally:
        conn.close()

    shown = _already_shown(session_id)
    token_sets = [_tokens(_memory_text(memory)) for memory in memories]
    identifier_sets = [_identifier_tokens(memory) for memory in memories]
    frequency = _doc_frequency(token_sets)

    scored: list[tuple[int, int, float, dict[str, object]]] = []
    for memory, memory_tokens, naming in zip(
        memories, token_sets, identifier_sets, strict=True
    ):
        if str(memory.get("memory_id")) in shown:
            continue
        local = db.is_local(memory, current_repo)
        score = _matches(
            prompt_tokens, memory_tokens, naming, frequency, foreign=not local
        )
        if score > 0:
            raw = memory.get("confidence", 0.0)
            rank_conf = raw if isinstance(raw, float) else 0.0
            scored.append((0 if local else 1, score, rank_conf, memory))
    if not scored:
        return None
    # This project's facts outrank another project's, whatever the overlap.
    scored.sort(key=lambda item: (item[0], -item[1], -item[2]))
    picked = [memory for _, _, _, memory in scored[:MAX_HITS]]

    _record_shown(session_id, [str(m.get("memory_id")) for m in picked])
    lines = [_render(memory) for memory in picked]
    return (
        "[MEMORY RECALL — stored facts matching this prompt]\n"
        + "\n".join(lines)
    )


def main() -> int:
    _make_stdout_lossy()
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    prompt = str(payload.get("prompt") or payload.get("prompt_text") or "")
    cwd = str(payload.get("cwd") or os.getcwd())
    session_id = str(payload.get("session_id") or "")
    try:
        block = recall(prompt, cwd, session_id)
    except Exception as exc:  # noqa: BLE001 — a hook failure must stay invisible
        print(f"clara prompt-recall: skipped ({exc})", file=sys.stderr)
        return 0
    if block:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
