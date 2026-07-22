"""
CLARA fastpath — emit the ``=== MEMORY CONTEXT ===`` block at session start.

Run as ``python -m clara.fastpath.context --cwd <dir>`` by the plugin's
SessionStart hook. Mirrors the empty-query recall path (`LocalMemory.recent`):
active memories ranked by the composite score with similarity 0 —
``0.20*confidence + 0.10*recency + 0.05*usage`` (weights from
``clara.retrieval.engine``) — top 12, grouped and formatted exactly like
``clara.reasoning.context.format_context``.

Stdlib-only (see the package docstring's hard rule): no ``math``, no
``datetime``, no ``re`` — recency uses ``E ** x``, the usage log-ratio uses a
few Newton steps on ``exp``, timestamps are parsed by SQLite, and the text
sanitizer is a scanner port of ``clara.core.text.sanitize_memory_text``
(kept in lockstep by a parity test).

Output contract: at most ``TOKEN_BUDGET`` (approx.) tokens — over budget the
oldest entries are dropped first; an empty or missing store emits nothing;
exit code is always 0.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

from clara.fastpath import db

TOP_K = 12
TOKEN_BUDGET = 900

# Scoring constants — keep in sync with clara.retrieval.engine.
_W_CONFIDENCE = 0.20
_W_RECENCY = 0.10
_W_USAGE = 0.05
_RECENCY_LAMBDA = 0.01

_E = 2.718281828459045
_SANITIZE_MAX_LEN = 500  # clara.core.text.DEFAULT_MAX_LEN


# ---------------------------------------------------------------------------
# math-free scoring helpers
# ---------------------------------------------------------------------------


def _exp(x: float) -> float:
    return float(_E**x)


def _ln(y: float) -> float:
    """Natural log via Newton on ``exp`` (math module is off-limits here)."""
    if y <= 0.0:
        return 0.0
    # Seed from the bit length (~log2) so large values converge in few steps.
    x = int(y).bit_length() * 0.6931471805599453
    for _ in range(8):
        x += y * _exp(-x) - 1.0
    return x


def _recency_score(updated_epoch: int | None, now_epoch: float) -> float:
    if updated_epoch is None:
        return 0.0
    days = max(0.0, (now_epoch - updated_epoch) / 86_400.0)
    return _exp(-_RECENCY_LAMBDA * days)


def _usage_frequency(access_count: int, max_access_count: int) -> float:
    if max_access_count <= 0:
        return 0.0
    denominator = _ln(1 + max_access_count)
    if denominator == 0.0:
        return 0.0
    return _ln(1 + access_count) / denominator


def _access_count(memory: dict[str, object]) -> int:
    meta = memory.get("metadata")
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("access_count", 0))
    except (TypeError, ValueError):
        return 0


def rank(memories: list[dict[str, object]], now_epoch: float) -> list[dict[str, object]]:
    """Composite-score ranking with similarity fixed at 0 (empty query)."""
    max_access = max((_access_count(m) for m in memories), default=0)
    scored: list[tuple[float, dict[str, object]]] = []
    for memory in memories:
        recency = _recency_score(memory.get("updated_epoch"), now_epoch)  # type: ignore[arg-type]
        usage = _usage_frequency(_access_count(memory), max_access)
        confidence = float(memory.get("confidence", 0.5))  # type: ignore[arg-type]
        score = _W_CONFIDENCE * confidence + _W_RECENCY * recency + _W_USAGE * usage
        scored.append((score, memory))
    # Stable sort over the updated_at-desc fetch order — same tie-breaking as
    # the LexicalRetriever path.
    scored.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in scored[:TOP_K]]


# ---------------------------------------------------------------------------
# sanitizer — scanner port of clara.core.text.sanitize_memory_text
# ---------------------------------------------------------------------------


def _defang_sections(text: str) -> str:
    """``[SECTION]`` -> ``(SECTION)`` for `/?[A-Z][A-Z0-9 _-]*` markers."""
    out: list[str] = []
    i = 0
    while True:
        start = text.find("[", i)
        if start == -1:
            out.append(text[i:])
            break
        end = text.find("]", start + 1)
        if end == -1:
            out.append(text[i:])
            break
        inner = text[start + 1 : end]
        body = inner[1:] if inner.startswith("/") else inner
        ok = bool(body) and "A" <= body[0] <= "Z"
        if ok:
            for ch in body[1:]:
                if not ("A" <= ch <= "Z" or "0" <= ch <= "9" or ch in " _-"):
                    ok = False
                    break
        if ok:
            out.append(text[i:start])
            out.append(f"({inner})")
            i = end + 1
        else:
            out.append(text[i : start + 1])
            i = start + 1
    return "".join(out)


def sanitize(value: object, *, max_len: int = _SANITIZE_MAX_LEN) -> str:
    text = "".join(
        " " if (ord(ch) < 0x20 or ord(ch) == 0x7F) else ch for ch in str(value)
    )
    while "==" in text:
        text = text.replace("==", "=")
    text = _defang_sections(text)
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# formatting — mirrors clara.reasoning.context.format_context
# ---------------------------------------------------------------------------


def _format_belief(memory: dict[str, object]) -> str:
    c = memory["content"]
    assert isinstance(c, dict)
    domain = c.get("domain")
    subject = sanitize(c.get("subject", "?"))
    relation = sanitize(c.get("relation", "?"))
    obj = sanitize(c.get("object", "?"))
    core = f"{subject} {relation} {obj}"
    if c.get("is_negation"):
        core = f"not ({core})"
    line = f"- {core}"
    line += f" (confidence: {float(memory['confidence']):.2f}"  # type: ignore[arg-type]
    if domain:
        line += f", domain: {sanitize(domain)}"
    line += ")"
    return line


def _format_event(memory: dict[str, object]) -> str:
    c = memory["content"]
    assert isinstance(c, dict)
    created = str(memory.get("created_at", ""))
    ts = created[:10] if len(created) >= 10 else "?"
    desc = sanitize(c.get("object", c.get("description", "")))
    subj = sanitize(c.get("subject", ""))
    rel = sanitize(c.get("relation", ""))
    return f"- {ts}: {subj} {rel} {desc}"


def _format_skill(memory: dict[str, object]) -> str:
    c = memory["content"]
    assert isinstance(c, dict)
    name = sanitize(c.get("name", c.get("object", "unnamed skill")))
    return f"- {name} (confidence: {float(memory['confidence']):.2f})"  # type: ignore[arg-type]


def _format_world_model(memory: dict[str, object]) -> str:
    c = memory["content"]
    assert isinstance(c, dict)
    parts: list[str] = []
    for key in ("name", "subject", "object"):
        if key in c and c[key]:
            parts.append(sanitize(c[key]))
            break
    props = c.get("properties", {})
    if props and isinstance(props, dict):
        parts.append(" | ".join(f"{sanitize(k)}: {sanitize(v)}" for k, v in props.items()))
    elif c.get("relation") and c.get("object"):
        parts.append(f"{sanitize(c.get('relation', ''))} {sanitize(c.get('object', ''))}")
    return f"- {' | '.join(parts)}" if parts else "- (world model entry)"


_SECTIONS: list[tuple[str, str, object]] = [
    ("[BELIEFS]", "belief", _format_belief),
    ("[WORLD MODEL]", "world_model", _format_world_model),
    ("[RECENT EVENTS]", "event", _format_event),
    ("[RELEVANT SKILLS]", "skill", _format_skill),
]


def format_block(memories: list[dict[str, object]]) -> str:
    sections: list[str] = ["=== MEMORY CONTEXT ===", ""]
    for header, mem_type, formatter in _SECTIONS:
        sections.append(header)
        bucket = [m for m in memories if m["type"] == mem_type]
        if bucket:
            for memory in bucket:
                sections.append(formatter(memory))  # type: ignore[operator]
        else:
            sections.append("- (none)")
        sections.append("")
    sections.append("=== END MEMORY CONTEXT ===")
    return "\n".join(sections)


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _updated_epoch(memory: dict[str, object]) -> int:
    value = memory.get("updated_epoch")
    return value if isinstance(value, int) else -1


def build_context(memories: list[dict[str, object]], now_epoch: float) -> str | None:
    """Rank, format, and enforce the token budget (drop oldest first)."""
    top = rank(memories, now_epoch)
    if not top:
        return None
    block = format_block(top)
    while top and _approx_tokens(block) > TOKEN_BUDGET:
        oldest = min(range(len(top)), key=lambda i: _updated_epoch(top[i]))
        top.pop(oldest)
        block = format_block(top)
    return block if top else None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    cwd = os.getcwd()
    for i, arg in enumerate(args):
        if arg == "--cwd" and i + 1 < len(args):
            cwd = args[i + 1]

    db_path, rid = db.resolve_store(cwd)
    if os.environ.get("CLARA_FASTPATH_DEBUG"):
        print(f"clara fastpath: repo_id={rid} store={db_path}", file=sys.stderr)
    if db_path is None:
        return 0
    conn = db.open_store(db_path)
    if conn is None:
        return 0
    try:
        memories = db.fetch_active(conn)
    except sqlite3.Error as exc:
        print(f"clara fastpath: {db_path}: read failed ({exc})", file=sys.stderr)
        return 0
    finally:
        conn.close()
    block = build_context(memories, time.time())
    if block:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
