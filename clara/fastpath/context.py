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
# At most this many memories stamped by OTHER repositories appear in a
# session's block; see rank() for the measurement that set it.
FOREIGN_CAP = 3

# Scoring constants — keep in sync with clara.retrieval.engine.
_W_CONFIDENCE = 0.20
_W_RECENCY = 0.10
_W_USAGE = 0.05
_RECENCY_LAMBDA = 0.01

# Doc-provenance tier weighting — keep in sync with clara.docs.TIER_MULTIPLIER
# and clara.retrieval.lexical (the recent()-parity test enforces this mirror).
_TIER_MULTIPLIER = {"T0": 1.2, "T1": 1.1, "T2": 1.0, "T3": 0.8}
_EXCLUDED_TIER = "TX"

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


def rank(
    memories: list[dict[str, object]],
    now_epoch: float,
    current_repo: str | None = None,
) -> list[dict[str, object]]:
    """Composite-score ranking with similarity fixed at 0 (empty query).

    When *current_repo* is given, this project's memories outrank another
    project's at any score: verified on a real store, nine findings saved
    from one repository filled eight of ten belief slots in every other
    project's session start. Foreign survivors are marked so the block can
    say where they came from. Callers that pass no repo get the pure
    score order.
    """
    max_access = max((_access_count(m) for m in memories), default=0)
    scored: list[tuple[int, float, dict[str, object]]] = []
    for memory in memories:
        meta = memory.get("metadata")
        doc_tier = meta.get("doc_tier") if isinstance(meta, dict) else None
        if doc_tier == _EXCLUDED_TIER:
            continue  # quarantined provenance is excluded from default context
        recency = _recency_score(memory.get("updated_epoch"), now_epoch)  # type: ignore[arg-type]
        usage = _usage_frequency(_access_count(memory), max_access)
        confidence = float(memory.get("confidence", 0.5))  # type: ignore[arg-type]
        score = _W_CONFIDENCE * confidence + _W_RECENCY * recency + _W_USAGE * usage
        if isinstance(doc_tier, str):
            score *= _TIER_MULTIPLIER.get(doc_tier, 1.0)
        foreign = current_repo is not None and not db.is_local(
            memory, current_repo
        )
        if foreign:
            memory["_foreign"] = True
        scored.append((1 if foreign else 0, score, memory))
    # Stable sort over the updated_at-desc fetch order — same tie-breaking as
    # the LexicalRetriever path. Locality first, score second.
    scored.sort(key=lambda item: (item[0], -item[1]))
    # Labels and ordering were not enough on their own: measured on a real
    # store, nine labeled foreign findings still consumed most of the token
    # budget of an unrelated project's block. A few foreign lines keep
    # cross-project awareness; more than that is another project's session.
    picked: list[dict[str, object]] = []
    foreign_taken = 0
    for is_foreign, _, memory in scored:
        if is_foreign:
            if foreign_taken >= FOREIGN_CAP:
                continue
            foreign_taken += 1
        picked.append(memory)
        if len(picked) >= TOP_K:
            break
    return picked


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


# A belief's triple says *what* was decided; the rationale saved alongside it
# says *why*. That reasoning was stored and then never shown, which is the one
# thing a decision is worth remembering for. Kept short so a long explanation
# cannot crowd out other memories.
RATIONALE_MAX_LEN = 120


def _rationale(memory: dict[str, object]) -> str:
    """The saved reasoning for a belief, or "" when there is none.

    Lives in ``metadata.evidence[0].text`` (BeliefMemory stores the caller's
    description as evidence), not in ``content``.
    """
    meta = memory.get("metadata")
    if not isinstance(meta, dict):
        return ""
    evidence = meta.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return ""
    first = evidence[0]
    if not isinstance(first, dict):
        return ""
    raw = first.get("text")
    # A save without a description stores None here, and str(None) is the
    # string "None" -- which rendered as an earnest '— None' rationale.
    if not raw:
        return ""
    return sanitize(raw, max_len=RATIONALE_MAX_LEN)


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
    rationale = _rationale(memory)
    # Skip a rationale that merely restates the triple — it costs tokens and
    # tells the model nothing it cannot already see on the line.
    if rationale and rationale.lower() not in core.lower():
        line += f" — {rationale}"
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
                line = formatter(memory)  # type: ignore[operator]
                if memory.get("_foreign"):
                    # Saved while working in a different repository; still
                    # shown when slots remain, but labeled so the model does
                    # not mistake another project's finding for this one's.
                    line += "  [from another project]"
                sections.append(line)
        else:
            sections.append("- (none)")
        sections.append("")
    sections.append("=== END MEMORY CONTEXT ===")
    return "\n".join(sections)


# The project header is a handful of lines; it must never crowd out actual
# memories, which are the point of the block.
PROJECT_TOKEN_BUDGET = 120

# Rendered in this order, so the most identifying facts come first. Categories
# absent from the profile are skipped entirely rather than printed empty.
_PROJECT_ROWS: tuple[tuple[str, str], ...] = (
    ("language", "language"),
    ("package_manager", "package manager"),
    ("framework", "frameworks"),
    ("build_tool", "build"),
    ("test_framework", "tests"),
    ("database", "database"),
    ("infrastructure", "infra"),
)
_PROJECT_MAX_PER_ROW = 4


def build_project_block(cwd: str) -> str | None:
    """Render a compact ``[PROJECT]`` header for the repo at *cwd*.

    Every value is sanitised before it is emitted: names and dependencies come
    from manifests inside a repository that may have been cloned from anywhere,
    and this text lands in the block the model reads as trusted background.
    Returns ``None`` when nothing was detected.
    """
    from clara.project.detect import detect_project

    profile = detect_project(cwd)
    if not profile.facts:
        return None

    lines = ["[PROJECT]"]
    headline = sanitize(profile.name, max_len=60) if profile.name else ""
    if profile.is_monorepo:
        workspaces = ", ".join(
            sanitize(w, max_len=30) for w in profile.workspaces[:_PROJECT_MAX_PER_ROW]
        )
        headline = f"{headline} (monorepo: {workspaces})" if headline else (
            f"monorepo: {workspaces}"
        )
    if headline:
        lines.append(headline)

    for category, label in _PROJECT_ROWS:
        values = profile.by_category(category)[:_PROJECT_MAX_PER_ROW]
        if values:
            rendered = ", ".join(sanitize(v, max_len=40) for v in values)
            lines.append(f"{label}: {rendered}")

    if len(lines) == 1:
        return None
    block = "\n".join(lines)
    while _approx_tokens(block) > PROJECT_TOKEN_BUDGET and len(lines) > 2:
        lines.pop()
        block = "\n".join(lines)
    return block


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _updated_epoch(memory: dict[str, object]) -> int:
    value = memory.get("updated_epoch")
    return value if isinstance(value, int) else -1


def build_context(
    memories: list[dict[str, object]],
    now_epoch: float,
    current_repo: str | None = None,
) -> str | None:
    """Rank, format, and enforce the token budget (drop oldest first)."""
    top = rank(memories, now_epoch, current_repo)
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


def _make_stdout_lossy() -> None:
    """Never let an un-encodable character take the session down.

    This block is rendered with a few non-ASCII characters (the truncation
    ellipsis, the rationale separator) and printed to whatever stdout the host
    handed us. Under an ASCII locale — ``LC_ALL=C``, or PYTHONIOENCODING=ascii —
    that raises UnicodeEncodeError and the hook exits non-zero, which breaks
    the "always exits 0, memory never blocks a session" contract. Verified: the
    ellipsis alone was already enough to crash it before this existed.

    Replacing the offending characters degrades one glyph; raising loses the
    entire memory block.

    Encoding matters as much as the error handler. When stdout is a pipe --
    which is exactly how a host captures a hook -- Python picks the *locale*
    encoding, so on a Windows machine this block went out as cp1252 and the em
    dash became byte 0x97. Verified: the SessionStart payload was not valid
    UTF-8, so a host decoding it as UTF-8 got a decode error or a replacement
    glyph in the injected memory. Captured output is for a machine and is
    written as UTF-8; a real console keeps its own encoding, so a legacy
    terminal still shows readable text rather than mojibake.
    """
    # noqa: SIM105 — contextlib.suppress would read better, but contextlib is
    # not on this package's allowed-import list and widening that list for a
    # style rule is the wrong trade on the session-start path.
    # An explicit PYTHONIOENCODING is a deliberate instruction and is left
    # alone; only the encoding Python inferred from the locale is overridden.
    try:  # noqa: SIM105
        piped = not sys.stdout.isatty() and not os.environ.get("PYTHONIOENCODING")
    except (AttributeError, ValueError, OSError):
        piped = False
    try:  # noqa: SIM105
        if piped:
            sys.stdout.reconfigure(  # type: ignore[union-attr]
                encoding="utf-8", errors="replace"
            )
        else:
            sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    _make_stdout_lossy()
    args = sys.argv[1:] if argv is None else argv
    cwd = os.getcwd()
    for i, arg in enumerate(args):
        if arg == "--cwd" and i + 1 < len(args):
            cwd = args[i + 1]

    db_path, rid = db.resolve_store(cwd)
    if os.environ.get("CLARA_FASTPATH_DEBUG"):
        print(f"clara fastpath: repo_id={rid} store={db_path}", file=sys.stderr)
    block: str | None = None
    if db_path is not None:
        conn = db.open_store(db_path)
        if conn is not None:
            try:
                memories = db.fetch_active(conn)
                block = build_context(memories, time.time(), rid)
            except sqlite3.Error as exc:
                print(f"clara fastpath: {db_path}: read failed ({exc})", file=sys.stderr)
            finally:
                conn.close()
    # The project header goes first: it frames everything below it, and it is
    # the one part that is useful even when the store is empty (a repo CLARA
    # has never seen still has manifests).
    try:
        project_block = build_project_block(cwd)
    except Exception as exc:  # noqa: BLE001 — must never block a session
        print(f"clara fastpath: project detection failed ({exc})", file=sys.stderr)
        project_block = None
    if project_block:
        print(project_block)

    if block:
        if project_block:
            print()
        print(block)

    try:
        from clara.fastpath import docs_map

        map_block = docs_map.build_map(cwd)
    except Exception as exc:  # noqa: BLE001 — the map must never block a session
        print(f"clara fastpath: knowledge map failed ({exc})", file=sys.stderr)
        map_block = None
    if map_block:
        if block or project_block:
            print()
        print(map_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
