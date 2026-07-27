"""
Import native memory files into the CLARA store.

Sources: project/user CLAUDE.md, MEMORY.md outside the CLARA fence, and
auto-memory topic files (excluding CLARA's own ``clara-memory.md``). Each
bullet/paragraph line runs through the precision-first
:class:`HeuristicExtractor`; lines that do not extract are skipped by
default (``verbatim=True`` stores them as ``("note", "states", <line>)``
beliefs).

Loop prevention: fenced lines and ``<!--clara:...-->``-tagged lines never
import; every imported memory carries ``origin_file`` + ``origin_hash``
provenance so re-imports dedup and exports skip the file's own lines.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from clara.bridge import markers, paths
from clara.extraction.heuristic import HeuristicExtractor
from clara.integrations.local_memory import LocalMemory

_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_HEADING_RE = re.compile(r"^\s*#")


def _origin_hash(line: str) -> str:
    normalized = " ".join(line.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def candidate_lines(text: str) -> list[str]:
    """Importable lines: bullets and plain paragraph lines, fences stripped,
    CLARA-tagged lines dropped, code fences skipped."""
    text = markers.strip_fences(text)
    out: list[str] = []
    in_code = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or _HEADING_RE.match(stripped):
            continue
        if markers.PROVENANCE_RE.search(stripped):
            continue
        match = _BULLET_RE.match(raw)
        line = match.group(1).strip() if match else stripped
        if len(line) >= 8:
            out.append(line)
    return out


async def _already_imported(memory: LocalMemory, origin_hash: str) -> bool:
    from sqlalchemy import text as sa_text

    async with memory.session_factory() as session:
        row = (
            await session.execute(
                sa_text(
                    "SELECT 1 FROM memories WHERE status = 'active' "
                    "AND json_extract(metadata, '$.origin_hash') = :h LIMIT 1"
                ),
                {"h": origin_hash},
            )
        ).first()
    return row is not None


async def _belief_exists(memory: LocalMemory, subject: str, relation: str, obj: str) -> bool:
    from sqlalchemy import text as sa_text

    async with memory.session_factory() as session:
        row = (
            await session.execute(
                sa_text(
                    "SELECT 1 FROM memories WHERE memory_type = 'belief' "
                    "AND status = 'active' "
                    "AND json_extract(content, '$.subject') = :s "
                    "AND json_extract(content, '$.relation') = :r "
                    "AND json_extract(content, '$.object') = :o LIMIT 1"
                ),
                {"s": subject, "r": relation, "o": obj},
            )
        ).first()
    return row is not None


async def import_file(
    memory: LocalMemory,
    path: Path,
    *,
    origin_label: str,
    verbatim: bool = False,
) -> dict[str, int]:
    stats = {"lines": 0, "imported": 0, "skipped_dup": 0, "skipped_no_extract": 0}
    try:
        text = paths.long_path(path).read_text(encoding="utf-8")
    except OSError:
        return stats
    extractor = HeuristicExtractor()
    for line in candidate_lines(text):
        stats["lines"] += 1
        origin_hash = _origin_hash(line)
        if await _already_imported(memory, origin_hash):
            stats["skipped_dup"] += 1
            continue
        facts = extractor.extract_sync(line)
        provenance: dict[str, Any] = {
            "origin": "native_import",
            "origin_file": origin_label,
            "origin_hash": origin_hash,
        }
        if facts:
            fact = facts[0]
            if fact.subject and fact.relation and fact.object:
                if await _belief_exists(memory, fact.subject, fact.relation, fact.object):
                    stats["skipped_dup"] += 1
                    continue
                saved = await memory.save(
                    mem_type="belief",
                    subject=fact.subject,
                    relation=fact.relation,
                    object=fact.object,
                    is_negation=bool(getattr(fact, "is_negation", False)),
                    description=line,
                    confidence=min(0.9, float(getattr(fact, "confidence", 0.7) or 0.7)),
                    source="user_direct",
                )
                await _stamp_provenance(memory, saved["memory_id"], provenance)
                stats["imported"] += 1
                continue
        if verbatim:
            saved = await memory.save(
                mem_type="belief",
                subject="note",
                relation="states",
                object=line[:400],
                description=line,
                confidence=0.6,
                source="user_direct",
            )
            await _stamp_provenance(memory, saved["memory_id"], provenance)
            stats["imported"] += 1
        else:
            stats["skipped_no_extract"] += 1
    return stats


async def _stamp_provenance(
    memory: LocalMemory, memory_id: str, provenance: dict[str, Any]
) -> None:
    from sqlalchemy import text as sa_text

    async with memory.session() as session:
        for key, value in provenance.items():
            await session.execute(
                sa_text(
                    "UPDATE memories SET metadata = json_set("
                    "coalesce(metadata, '{}'), :path, :value), "
                    "updated_at = updated_at "
                    "WHERE memory_id = :mid"
                ),
                {
                    "path": f"$.{key}",
                    "value": value,
                    "mid": memory_id.replace("-", ""),
                },
            )


async def import_native(
    memory: LocalMemory, anchor: str, *, verbatim: bool = False
) -> dict[str, Any]:
    """Import every native source for *anchor*'s project. Returns per-file stats."""
    results: dict[str, Any] = {}
    for source in paths.claude_md_paths(anchor):
        label = f"native:{source.name}"
        results[str(source)] = await import_file(
            memory, source, origin_label=label, verbatim=verbatim
        )
    memory_md = paths.memory_md_path(anchor)
    if memory_md is not None and memory_md.is_file():
        results[str(memory_md)] = await import_file(
            memory, memory_md, origin_label="native:MEMORY.md", verbatim=verbatim
        )
    directory = paths.auto_memory_dir(anchor)
    if directory is not None and directory.is_dir():
        for topic in sorted(directory.glob("*.md")):
            if topic.name in ("MEMORY.md", paths.CLARA_TOPIC_FILE):
                continue
            results[str(topic)] = await import_file(
                memory, topic, origin_label=f"native:{topic.name}", verbatim=verbatim
            )
    return results
