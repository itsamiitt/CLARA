"""
CLARA — LocalMemory: zero-backend memory facade

The engine behind the Claude Code / MCP integration. Unlike :class:`ClaraMemory`
it uses **no LLM, no embedding model, and no LanceDB** — the calling agent
(Claude) supplies already-structured memories and plain-text queries.

* Writes go straight to SQLite through the existing typed memory stores
  (``BeliefMemory``, ``EventStore``, ``SkillStore``, ``WorldModelStore``) with
  ``embedding=None``.
* Reads use :class:`LexicalRetriever` (keyword search over SQLite).

The only state on disk is a single SQLite file. LanceDB stays dormant via the
``_clara_disable_lance`` session flag (see ``clara/retrieval/engine.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy import text as sa_text
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from clara import security, stats_cache
from clara.core.ids import canonical_id
from clara.db.engine import _make_engine
from clara.db.fts import ensure_fts
from clara.db.migrations import SchemaTooNew, ensure_schema
from clara.db.models import Base, Memory, MemoryStatus, MemoryType
from clara.db.pragmas import apply_runtime
from clara.db.retry import with_sqlite_retry
from clara.flags import (
    DOCS_DISABLED_HINT,
    GRAPH_DISABLED_HINT,
    docs_enabled,
    graph_enabled,
)
from clara.graph import project as graph_project
from clara.graph import render as graph_render
from clara.graph import traverse as graph_traverse
from clara.graph.resolve import resolve_node
from clara.memory.belief import BeliefMemory, SourceType
from clara.memory.event import EventStore
from clara.memory.skill import SkillStore
from clara.memory.world_model import WorldModelStore
from clara.reasoning.context import format_context
from clara.repoid import repo_id
from clara.retrieval.engine import ScoredMemory
from clara.retrieval.lexical import LexicalRetriever

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in MemoryType}

# repo_id shells out to git; cache per working directory so long-lived MCP
# servers pay the subprocess cost once, not per save.
_REPO_ID_CACHE: dict[str, str] = {}


def _cached_repo_id() -> str:
    cwd = str(Path.cwd())
    cached = _REPO_ID_CACHE.get(cwd)
    if cached is None:
        cached = repo_id(cwd)
        _REPO_ID_CACHE[cwd] = cached
    return cached


class StoreReadOnly(RuntimeError):
    """A write was refused because the store is open read-only.

    Distinct from SQLite's own "attempt to write a readonly database", which
    means the *file* is unwritable (permissions, read-only mount). This one
    means the schema is newer than this build understands, so writing could
    corrupt it. The two need opposite advice -- upgrade CLARA versus fix the
    file -- and the CLI used to give the first for both.
    """


def _ensure_versioned_schema(db_path: str) -> None:
    """Apply versioned migrations (memories, graph, docs, FTS) to a file store.

    Since schema v4 this is the primary schema path. A backup is taken before
    any pending migration runs (see :mod:`clara.db.backup`).

    Fail-soft for migrations that *cannot* run (read-only mount, locked file):
    memory availability beats add-on availability, and those degrade to no-ops.

    NOT fail-soft for :class:`SchemaTooNew`, which is a different situation
    wearing the same exception costume. A store written by a newer CLARA may
    have constraints, columns and triggers this build knows nothing about;
    writing to it is how you corrupt someone's memory during a downgrade.
    clara/db/migrations.py states the rule plainly -- "if the database's
    version is NEWER than this code knows, never write" -- but this function
    used to swallow it along with everything else, print a traceback, and let
    the caller carry on writing. Verified before this change: against a store
    marked v99 the row count still went 1 -> 2. It is re-raised so
    :meth:`LocalMemory.create` can open the store read-only instead.
    """
    if db_path in ("", ":memory:"):
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            apply_runtime(conn)
            from clara.db.backup import backup_before_migration

            backup_before_migration(conn, db_path)
            ensure_schema(conn)
        finally:
            conn.close()
    except SchemaTooNew:
        raise
    except Exception:  # noqa: BLE001 — memory availability beats graph availability
        logger.warning("versioned schema migration failed for %s", db_path, exc_info=True)


# Write-path guardrails (see plan §3.3): a single memory is a compact fact,
# not a document dump — and never a credential.
_MAX_CONTENT_BYTES = int(os.environ.get("CLARA_MAX_CONTENT_BYTES", "16384") or 16384)
_MAX_TAGS = 64
_MAX_TAG_LENGTH = 256


def _clean_text(value: Any) -> Any:
    """Strip surrounding whitespace; map whitespace-only text to ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _redact_value(value: Any) -> tuple[Any, list[str]]:
    """Redact secrets anywhere inside a str, list, or dict value.

    Recursive because ``properties``/``steps``/``trigger_conditions`` carry
    nested user content; a credential in any of them used to survive the
    ``redact`` policy because only three top-level strings were scrubbed.
    """
    if isinstance(value, str):
        return security.redact(value)
    if isinstance(value, list):
        cleaned: list[Any] = []
        names: list[str] = []
        for item in value:
            clean_item, matched = _redact_value(item)
            cleaned.append(clean_item)
            names.extend(matched)
        return cleaned, names
    if isinstance(value, dict):
        cleaned_map: dict[Any, Any] = {}
        names = []
        for key, item in value.items():
            clean_key, key_matched = _redact_value(key)
            clean_item, matched = _redact_value(item)
            cleaned_map[clean_key] = clean_item
            names.extend(key_matched)
            names.extend(matched)
        return cleaned_map, names
    return value, []


def _guard_and_redact(
    fields: dict[str, Any], tags: list[str] | None
) -> tuple[dict[str, Any], list[str] | None, list[str]]:
    """Enforce tag caps, the size cap, and the secret policy on one save.

    Returns ``(fields, tags, redacted_names)``. Under ``reject`` (the default)
    a match raises and nothing is written; under ``redact`` *every* scanned
    field comes back scrubbed, so the caller can persist the return values
    directly. Lives here rather than in ``save()`` so that every write path —
    including ``docs_fulfill``, which calls ``_route_save`` directly — is
    covered; a guard one layer above the routing function is a guard that a
    future caller can forget.
    """
    if tags is not None:
        if len(tags) > _MAX_TAGS:
            raise ValueError(f"too many tags ({len(tags)} > {_MAX_TAGS}).")
        for tag in tags:
            if len(str(tag)) > _MAX_TAG_LENGTH:
                raise ValueError(
                    f"tag too long ({len(str(tag))} chars > {_MAX_TAG_LENGTH})."
                )
    payload = json.dumps(
        {k: v for k, v in fields.items() if v is not None}, ensure_ascii=False
    )
    if len(payload.encode("utf-8")) > _MAX_CONTENT_BYTES:
        raise ValueError(
            f"content is {len(payload.encode('utf-8'))} bytes "
            f"(cap {_MAX_CONTENT_BYTES}) — split it, or store a summary plus a "
            "file reference. Raise CLARA_MAX_CONTENT_BYTES to change the cap."
        )
    policy = security.secret_policy()
    if policy == "off":
        return fields, tags, []
    scan_text = payload + " " + " ".join(str(t) for t in (tags or []))
    if policy == "reject":
        name = security.find_secret(scan_text)
        if name is not None:
            raise security.SecretRejected(name)
        return fields, tags, []

    if security.find_secret(scan_text) is None:
        return fields, tags, []
    clean_fields: dict[str, Any] = {}
    names: list[str] = []
    for key, value in fields.items():
        clean_value, matched = _redact_value(value)
        clean_fields[key] = clean_value
        names.extend(matched)
    clean_tags = tags
    if tags:
        clean_tags, tag_matches = _redact_value(list(tags))
        names.extend(tag_matches)
    return clean_fields, clean_tags, sorted(set(names))


def _coerce_types(types: Sequence[str] | None) -> list[MemoryType] | None:
    """Map a list of type strings to :class:`MemoryType`, ignoring unknowns."""
    if not types:
        return None
    coerced: list[MemoryType] = []
    for raw in types:
        try:
            coerced.append(MemoryType(raw))
        except ValueError:
            logger.debug("Ignoring unknown memory type %r", raw)
    return coerced or None


def _source_type(source: str) -> SourceType:
    try:
        return SourceType(source)
    except ValueError:
        return SourceType.user_direct


def _serialize(sm: ScoredMemory) -> dict[str, Any]:
    return {
        "memory_id": str(sm.memory.memory_id),
        "type": sm.memory.memory_type.value,
        "score": round(sm.score, 4),
        "confidence": round(sm.confidence, 4),
        "content": sm.memory.content,
    }


class LocalMemory:
    """SQLite-only memory store driven entirely by the calling agent."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        db_path: str,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._db_path = db_path
        # Set by create() when the store's schema is newer than this build
        # understands; the engine is then bound to a mode=ro URL and SQLite
        # itself refuses writes.
        self._read_only = False

    @property
    def read_only(self) -> bool:
        """True when the store was opened read-only (schema newer than us)."""
        return self._read_only

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, db_path: str) -> LocalMemory:
        """Open (or create) the SQLite store at *db_path* and build the schema."""
        path_obj = Path(db_path)
        # Forward-slash form keeps the SQLAlchemy URL valid on Windows paths.
        db_url = f"sqlite+aiosqlite:///{path_obj.as_posix()}"
        db_path = str(path_obj)
        # Versioned migrations are the primary schema path for file stores
        # (memories DDL lives in migration 4, FTS in 5); create_all stays as
        # a checkfirst no-op safety net and the whole path for ":memory:".
        read_only = False
        try:
            await asyncio.to_thread(_ensure_versioned_schema, db_path)
        except SchemaTooNew as exc:
            # Open read-only rather than refusing outright: the user can still
            # read everything they have, which is the whole point of the store,
            # and a downgrade becomes a visible degradation instead of silent
            # corruption. create_all and ensure_fts are skipped deliberately --
            # both write, and running them here is what would apply this
            # build's older DDL on top of a newer schema.
            read_only = True
            db_url = f"sqlite+aiosqlite:///file:{path_obj.as_posix()}?mode=ro&uri=true"
            logger.warning(
                "%s - opening the store READ-ONLY. Memories can be read but "
                "not written until you upgrade clara-memory.", exc,
            )

        engine = _make_engine(db_url)
        if not read_only:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await ensure_fts(engine)

        # The info dict is shared by every session this factory makes, so the
        # LanceDB commit listeners short-circuit and never touch a vector store.
        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
            info={"_clara_disable_lance": True},
        )
        logger.info("LocalMemory ready (db=%s, backend=none)", db_path)
        instance = cls(engine=engine, session_factory=session_factory, db_path=db_path)
        instance._read_only = read_only
        return instance

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("LocalMemory closed")

    # ------------------------------------------------------------------
    # Public accessors (sanctioned entry points for other subsystems)
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> str:
        """Filesystem path of the backing SQLite store."""
        return self._db_path

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """The session factory, for subsystems that manage their own txn."""
        return self._session_factory

    @property
    def engine(self) -> AsyncEngine:
        """The underlying engine, for callers needing a raw connection."""
        return self._engine

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session in an open transaction, committed on clean exit.

        The sanctioned way to run a unit of work against the store. CLI, docs,
        bridge, and MCP previously reached through ``_session_factory`` /
        ``_engine`` because no public accessor existed — that made every
        internal refactor a breaking change for four modules.
        """
        async with self._session_factory() as session, session.begin():
            yield session

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _require_writable(self) -> None:
        """Refuse a write up front when the store was opened read-only.

        Without this the write still fails -- SQLite rejects it -- but the
        caller gets "(sqlite3.OperationalError) attempt to write a readonly
        database" followed by the whole INSERT and every bound parameter. Over
        MCP that dump is what the model receives as the tool result, which is
        both unactionable and needlessly loud. Failing here states the cause
        and the fix instead, and touches no SQL.
        """
        if self._read_only:
            raise StoreReadOnly(
                "this store was written by a newer version of CLARA, so it is "
                "open read-only and nothing was written. Upgrade with "
                "'pip install -U clara-memory' to write to it again; reading "
                "keeps working meanwhile."
            )

    async def save(
        self,
        *,
        mem_type: str = "belief",
        # belief
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        is_negation: bool = False,
        # event
        event_type: str | None = None,
        # skill
        name: str | None = None,
        trigger_conditions: list[str] | None = None,
        steps: list[str] | None = None,
        # world_model
        entity_type: str | None = None,
        properties: dict[str, Any] | None = None,
        # shared
        description: str | None = None,
        domain: str | None = None,
        confidence: float | None = None,
        source: str = "user_direct",
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one structured memory and return ``{memory_id, type, action}``.

        Raises ``ValueError`` if *mem_type* is unknown or required fields for the
        chosen type are missing.
        """
        self._require_writable()
        if mem_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown mem_type {mem_type!r}. Expected one of {sorted(VALID_TYPES)}."
            )

        async def _attempt() -> tuple[str, str]:
            async with self._session_factory() as session, session.begin():
                record = await self._route_save(
                    session,
                    mem_type=mem_type,
                    subject=subject,
                    relation=relation,
                    object=object,
                    is_negation=is_negation,
                    event_type=event_type,
                    name=name,
                    trigger_conditions=trigger_conditions,
                    steps=steps,
                    entity_type=entity_type,
                    properties=properties,
                    description=description,
                    domain=domain,
                    confidence=confidence,
                    source=source,
                    user_id=user_id,
                    tags=tags,
                )
                if confidence is not None:
                    record.confidence = max(0.0, min(1.0, float(confidence)))
                meta = dict(record.metadata_) if record.metadata_ else {}
                # Stamp provenance on new writes; world-model upserts of an
                # existing record keep their original repo_id.
                meta.setdefault("repo_id", _cached_repo_id())
                record.metadata_ = meta
                return str(record.memory_id), record.memory_type.value

        memory_id, rec_type = await with_sqlite_retry(_attempt, what="save")
        logger.info("Saved %s memory %s", rec_type, memory_id)
        await self._refresh_stats_cache()
        return {"memory_id": memory_id, "type": rec_type, "action": "saved"}

    async def _refresh_stats_cache(self) -> None:
        """Update the status-line counter after a write. Fail-soft.

        The status line is pulled on a timer and must not open SQLite on that
        cadence, so the count lives in a sidecar file that write paths refresh.
        Runs after the transaction has committed (never inside it) and off the
        event loop, so a cosmetic counter cannot slow or fail a save.
        """
        try:
            await asyncio.to_thread(stats_cache.refresh, self._db_path)
        except Exception:  # noqa: BLE001 — the counter is cosmetic
            logger.debug("stats cache refresh skipped", exc_info=True)

    @staticmethod
    async def _route_save(
        session: AsyncSession,
        *,
        mem_type: str,
        subject: str | None,
        relation: str | None,
        object: str | None,
        is_negation: bool,
        event_type: str | None,
        name: str | None,
        trigger_conditions: list[str] | None,
        steps: list[str] | None,
        entity_type: str | None,
        properties: dict[str, Any] | None,
        description: str | None,
        domain: str | None,
        confidence: float | None,
        source: str,
        user_id: str | None,
        tags: list[str] | None = None,
    ) -> Memory:
        if mem_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown mem_type {mem_type!r}. Expected one of {sorted(VALID_TYPES)}."
            )
        fields, tags, redacted = _guard_and_redact(
            {
                "subject": subject,
                "relation": relation,
                "object": object,
                "event_type": event_type,
                "name": name,
                "trigger_conditions": trigger_conditions,
                "steps": steps,
                "entity_type": entity_type,
                "properties": properties,
                "description": description,
                "domain": domain,
            },
            tags,
        )
        # Whitespace-only text is not a fact: normalize before the required-field
        # checks below so "   " fails validation instead of being stored.
        subject = _clean_text(fields["subject"])
        relation = _clean_text(fields["relation"])
        object = _clean_text(fields["object"])
        event_type = _clean_text(fields["event_type"])
        name = _clean_text(fields["name"])
        entity_type = _clean_text(fields["entity_type"])
        description = _clean_text(fields["description"])
        domain = _clean_text(fields["domain"])
        trigger_conditions = fields["trigger_conditions"]
        steps = fields["steps"]
        properties = fields["properties"]

        record = await LocalMemory._route_save_record(
            session,
            mem_type=mem_type,
            subject=subject,
            relation=relation,
            object=object,
            is_negation=is_negation,
            event_type=event_type,
            name=name,
            trigger_conditions=trigger_conditions,
            steps=steps,
            entity_type=entity_type,
            properties=properties,
            description=description,
            domain=domain,
            confidence=confidence,
            source=source,
            user_id=user_id,
        )
        if tags or redacted:
            meta = dict(record.metadata_) if record.metadata_ else {}
            if tags:
                meta["tags"] = list(tags)
            if redacted:
                meta["redacted"] = redacted
            record.metadata_ = meta
        return record

    @staticmethod
    async def _route_save_record(
        session: AsyncSession,
        *,
        mem_type: str,
        subject: str | None,
        relation: str | None,
        object: str | None,
        is_negation: bool,
        event_type: str | None,
        name: str | None,
        trigger_conditions: list[str] | None,
        steps: list[str] | None,
        entity_type: str | None,
        properties: dict[str, Any] | None,
        description: str | None,
        domain: str | None,
        confidence: float | None,
        source: str,
        user_id: str | None,
    ) -> Memory:
        """Dispatch to the per-type store. Guarded callers only — use
        :meth:`_route_save`, which applies the write guards first."""
        if mem_type == MemoryType.belief.value:
            if not (subject and relation and object):
                raise ValueError(
                    "belief requires 'subject', 'relation', and 'object'."
                )
            return await BeliefMemory(session).store(
                subject=subject,
                relation=relation,
                object_=object,
                domain=domain,
                is_negation=is_negation,
                source=_source_type(source),
                raw_text=description,
                user_id=user_id,
            )

        if mem_type == MemoryType.event.value:
            if not (subject and event_type):
                raise ValueError("event requires 'subject' and 'event_type'.")
            return await EventStore(session).create(
                subject=subject,
                event_type=event_type,
                description=description or "",
                domain=domain,
                user_id=user_id,
                confidence=confidence if confidence is not None else 1.0,
                source_type=source,
                raw_text=description,
            )

        if mem_type == MemoryType.skill.value:
            if not name:
                raise ValueError("skill requires 'name'.")
            return await SkillStore(session).create(
                name=name,
                trigger_conditions=trigger_conditions,
                steps=steps,
                description=description or "",
                domain=domain,
                user_id=user_id,
                confidence=confidence if confidence is not None else 0.5,
                source_type=source,
                raw_text=description,
            )

        # world_model
        if not (entity_type and name):
            raise ValueError("world_model requires 'entity_type' and 'name'.")
        return await WorldModelStore(session).upsert(
            entity_type=entity_type,
            name=name,
            properties=properties,
            domain=domain,
            user_id=user_id,
            confidence=confidence if confidence is not None else 0.9,
            source_type=source,
            raw_text=description,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        types: Sequence[str] | None = None,
        user_id: str | None = None,
        graph_depth: int = 0,
    ) -> dict[str, Any]:
        """Keyword-search memories. Returns a formatted context block + hits.

        ``graph_depth > 0`` appends a ``[GRAPH]`` section built by traversing
        the knowledge graph from the entities of the top hits (fail-soft).
        """
        async with self._session_factory() as session:
            retriever = LexicalRetriever(session)
            result = await retriever.search(
                query,
                top_k=top_k,
                memory_types=_coerce_types(types),
                user_id=user_id,
            )
        if query.strip() and result.total:
            # Real queries feed the usage score term; the empty-query recency
            # feed (recent()) does not — bumping there would be noise.
            await self._record_accesses(result.all)
        payload: dict[str, Any] = {
            "query": query,
            "total": result.total,
            "context": format_context(result),
            "hits": [_serialize(sm) for sm in result.all],
        }
        if graph_depth > 0 and result.total and graph_enabled():
            try:
                graph = await self._graph_augment(result, depth=graph_depth, user_id=user_id)
            except Exception:  # noqa: BLE001 — graph must never break search
                logger.warning("graph augmentation failed", exc_info=True)
                graph = None
            if graph:
                payload["context"] += "\n\n" + graph["section"]
                payload["graph"] = {"edges": graph["edges"]}
        return payload

    async def _record_accesses(self, hits: Sequence[ScoredMemory]) -> None:
        """Bump ``metadata.access_count``/``last_accessed`` on returned hits.

        Feeds the 0.05 usage term of the ranking score (and skill-pruning
        freshness via ``last_used``). The explicit ``updated_at`` self-assign
        suppresses the ORM ``onupdate`` — a read must never look like a
        content change, or recency ranking would inflate on every search.
        Fail-soft: recording usage must never break search.
        """
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            ids = [sm.memory.memory_id for sm in hits]
            skill_ids = [
                sm.memory.memory_id
                for sm in hits
                if sm.memory.memory_type == MemoryType.skill
            ]

            async def _attempt() -> None:
                async with self._session_factory() as session, session.begin():
                    base_meta = func.coalesce(Memory.metadata_, sa_text("'{}'"))
                    await session.execute(
                        sa_update(Memory)
                        .where(Memory.memory_id.in_(ids))
                        .values(
                            metadata_=func.json_set(
                                base_meta,
                                "$.access_count",
                                func.coalesce(
                                    func.json_extract(Memory.metadata_, "$.access_count"),
                                    0,
                                )
                                + 1,
                                "$.last_accessed",
                                now_iso,
                            ),
                            updated_at=Memory.updated_at,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if skill_ids:
                        await session.execute(
                            sa_update(Memory)
                            .where(Memory.memory_id.in_(skill_ids))
                            .values(
                                metadata_=func.json_set(
                                    func.coalesce(Memory.metadata_, sa_text("'{}'")),
                                    "$.last_used",
                                    now_iso,
                                ),
                                updated_at=Memory.updated_at,
                            )
                            .execution_options(synchronize_session=False)
                        )

            await with_sqlite_retry(_attempt, what="usage update")
        except Exception:  # noqa: BLE001 — usage accounting is best-effort
            logger.debug("access recording skipped", exc_info=True)

    async def _graph_augment(
        self, result: Any, *, depth: int, user_id: str | None
    ) -> dict[str, Any] | None:
        """Traverse from the top hits' entities and render a [GRAPH] section."""
        seed_names: list[str] = []
        for sm in result.all[:5]:
            content = sm.memory.content if isinstance(sm.memory.content, dict) else {}
            for key in ("subject", "object", "name"):
                value = content.get(key)
                if isinstance(value, str) and value and value not in seed_names:
                    seed_names.append(value)
        seed_names = [n for n in seed_names if n.lower() != "user"][:3]
        if not seed_names:
            return None
        async with self._session_factory() as session, session.begin():
            groups: list[tuple[str, list[dict[str, Any]]]] = []
            all_edges: list[dict[str, Any]] = []
            node_ids: list[str] = []
            for name in seed_names:
                node = await resolve_node(
                    session, name, user_id=user_id, create=False, bump_mention=False
                )
                if node is None:
                    continue
                edges = await graph_traverse.traverse(
                    session,
                    [node["node_id"]],
                    depth=depth,
                    user_id=user_id,
                    limit=12,
                )
                if not edges:
                    continue
                groups.append((node["display_name"], edges))
                all_edges.extend(edges)
                node_ids.extend(
                    [e["src_id"] for e in edges] + [e["dst_id"] for e in edges]
                )
            if not all_edges:
                return None
            nodes = await graph_traverse.fetch_nodes(session, node_ids)
            if not self._read_only:
                # bump_traversed is usage accounting: it raises edge weight and
                # node mention_count so traversed paths rank higher next time.
                # It is a write, and search() drops the whole [GRAPH] section if
                # anything in here raises -- so on a read-only store this one
                # bookkeeping call silently cost the user their graph context on
                # every search. The reading half works fine without it; only the
                # ranking feedback is lost, which a read-only store cannot
                # persist anyway.
                await graph_traverse.bump_traversed(session, all_edges)
            section = graph_render.render_graph_section(groups, nodes)
        if not section:
            return None
        edges_payload = [
            {
                "edge_id": e["edge_id"],
                "src": nodes.get(e["src_id"], {}).get("display_name", e["src_id"]),
                "relation": e["relation"],
                "dst": nodes.get(e["dst_id"], {}).get("display_name", e["dst_id"]),
                "confidence": e["confidence"],
                "depth": e["depth"],
                "score": round(float(e["score"]), 4),
            }
            for e in all_edges
        ]
        return {"section": section, "edges": edges_payload}

    async def recent(
        self,
        *,
        n: int = 10,
        types: Sequence[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Most relevant recent memories (recency + confidence ranked)."""
        return await self.search("", top_k=n, types=types, user_id=user_id)

    # ------------------------------------------------------------------
    # Mutate / lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_memory_id(memory_id: str) -> uuid.UUID:
        """A malformed id is "not found", not a stack trace.

        ``uuid.UUID`` raises "badly formed hexadecimal UUID string", which says
        nothing about what was passed or what to do about it, and leaked
        straight through the MCP tool to the model. A caller that invented an
        id should get the same answer as one that used a well-formed id which
        happens not to exist, because those are the same situation.
        """
        try:
            return uuid.UUID(str(memory_id))
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                f"Memory {memory_id!r} not found (not a valid memory id)."
            ) from None

    async def update(
        self,
        memory_id: str,
        *,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update confidence and/or tags on an existing memory."""
        self._require_writable()
        mid = self._parse_memory_id(memory_id)

        async def _attempt() -> str:
            async with self._session_factory() as session, session.begin():
                record = await session.get(Memory, mid)
                if record is None:
                    raise ValueError(f"Memory {memory_id} not found.")
                if confidence is not None:
                    record.confidence = max(0.0, min(1.0, float(confidence)))
                    await graph_project.project_confidence_changed(
                        session, str(mid), record.confidence
                    )
                if tags is not None:
                    meta = dict(record.metadata_) if record.metadata_ else {}
                    meta["tags"] = list(tags)
                    record.metadata_ = meta
                return record.memory_type.value

        rec_type = await with_sqlite_retry(_attempt, what="update")
        return {"memory_id": str(mid), "type": rec_type, "action": "updated"}

    async def forget(self, memory_id: str, *, archive: bool = False) -> dict[str, Any]:
        """Deprecate (default) or archive a memory. Never hard-deletes."""
        self._require_writable()
        mid = self._parse_memory_id(memory_id)
        new_status = MemoryStatus.archived if archive else MemoryStatus.deprecated

        async def _attempt() -> None:
            async with self._session_factory() as session, session.begin():
                record = await session.get(Memory, mid)
                if record is None:
                    raise ValueError(f"Memory {memory_id} not found.")
                record.status = new_status
                await graph_project.project_memory_invalidated(
                    session, str(mid), reason=new_status.value
                )

        await with_sqlite_retry(_attempt, what="forget")
        await self._refresh_stats_cache()
        return {"memory_id": str(mid), "action": new_status.value}

    async def stats(self) -> dict[str, Any]:
        """Counts of active memories per type, plus total rows."""
        async with self._session_factory() as session:
            by_type_stmt = (
                select(Memory.memory_type, func.count())
                .where(Memory.status == MemoryStatus.active)
                .group_by(Memory.memory_type)
            )
            by_type = {
                mtype.value: count
                for mtype, count in (await session.execute(by_type_stmt)).all()
            }
            total = (
                await session.execute(select(func.count()).select_from(Memory))
            ).scalar_one()
            graph_counts: dict[str, int] = {}
            try:
                for label, sql in (
                    ("nodes", "SELECT COUNT(*) FROM graph_nodes WHERE status = 'active'"),
                    ("edges", "SELECT COUNT(*) FROM graph_edges WHERE invalid_at IS NULL"),
                ):
                    graph_counts[label] = int(
                        (await session.execute(sa_text(sql))).scalar_one()
                    )
            except Exception:  # noqa: BLE001 — graph tables may be absent
                graph_counts = {}
        payload = {
            "db_path": self._db_path,
            "backend": "none (lexical + sqlite)",
            "active_by_type": by_type,
            "total_rows": int(total),
        }
        if graph_counts:
            payload["graph"] = graph_counts
        return payload

    # ------------------------------------------------------------------
    # Knowledge graph API
    # ------------------------------------------------------------------

    async def graph_entity(
        self, name: str, *, include_history: bool = False, user_id: str | None = None
    ) -> dict[str, Any]:
        """Entity card: node fields, aliases, possible_duplicates, top edges."""
        import json as _json

        if not graph_enabled():
            return {"found": False, "disabled": True, "error": GRAPH_DISABLED_HINT}

        async with self._session_factory() as session, session.begin():
            node = await resolve_node(
                session, name, user_id=user_id, create=False, bump_mention=False
            )
            if node is None:
                return {"found": False, "name": name}
            aliases = (
                await session.execute(
                    sa_text("SELECT alias_norm FROM graph_aliases WHERE node_id = :nid"),
                    {"nid": node["node_id"]},
                )
            ).scalars().all()
            edges = await graph_traverse.traverse(
                session, [node["node_id"]], depth=1, limit=10, user_id=user_id
            )
            history: list[dict[str, Any]] = []
            if include_history:
                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT * FROM graph_edges WHERE invalid_at IS NOT NULL "
                            "AND (src_id = :nid OR dst_id = :nid) "
                            "ORDER BY invalid_at DESC LIMIT 10"
                        ),
                        {"nid": node["node_id"]},
                    )
                ).mappings().all()
                history = [dict(r) for r in rows]
            node_ids = [node["node_id"]]
            for e in [*edges, *history]:
                node_ids.extend([e["src_id"], e["dst_id"]])
            nodes = await graph_traverse.fetch_nodes(session, node_ids)
        try:
            properties = _json.loads(node.get("properties") or "{}")
        except ValueError:
            properties = {}
        card_lines = [
            graph_render.format_edge_line(e, nodes) for e in [*edges, *history]
        ]
        return {
            "found": True,
            "node_id": node["node_id"],
            "name": node["display_name"],
            "canonical_name": node["canonical_name"],
            "entity_type": node["entity_type"],
            "mention_count": node["mention_count"],
            "expandable": bool(node["expandable"]),
            "world_model_id": node.get("world_model_id"),
            "aliases": list(aliases),
            "possible_duplicates": properties.get("possible_duplicates", []),
            "properties": properties,
            "edges": card_lines,
        }

    async def graph_neighbors(
        self,
        name: str,
        *,
        depth: int = 1,
        relation: str | None = None,
        as_of: str | None = None,
        limit: int = 20,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Neighborhood traversal from a named entity, rendered + structured."""
        if not graph_enabled():
            return {"found": False, "disabled": True, "error": GRAPH_DISABLED_HINT}
        async with self._session_factory() as session, session.begin():
            node = await resolve_node(
                session, name, user_id=user_id, create=False, bump_mention=False
            )
            if node is None:
                return {"found": False, "name": name, "edges": []}
            edges = await graph_traverse.traverse(
                session,
                [node["node_id"]],
                depth=depth,
                relation=relation,
                as_of=as_of,
                limit=limit,
                user_id=user_id,
            )
            node_ids = [node["node_id"]]
            for e in edges:
                node_ids.extend([e["src_id"], e["dst_id"]])
            nodes = await graph_traverse.fetch_nodes(session, node_ids)
            if edges and as_of is None:
                await graph_traverse.bump_traversed(session, edges)
        section = graph_render.render_graph_section(
            [(node["display_name"], edges)], nodes
        )
        return {
            "found": True,
            "name": node["display_name"],
            "context": section,
            "edges": [
                {
                    "edge_id": e["edge_id"],
                    "line": graph_render.format_edge_line(e, nodes),
                    "depth": e["depth"],
                    "score": round(float(e["score"]), 4),
                }
                for e in edges
            ],
        }

    async def graph_path(
        self,
        from_name: str,
        to_name: str,
        *,
        max_hops: int = 4,
        as_of: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Best path between two named entities, or found=False."""
        if not graph_enabled():
            return {"found": False, "disabled": True, "error": GRAPH_DISABLED_HINT}
        async with self._session_factory() as session, session.begin():
            src = await resolve_node(
                session, from_name, user_id=user_id, create=False, bump_mention=False
            )
            dst = await resolve_node(
                session, to_name, user_id=user_id, create=False, bump_mention=False
            )
            if src is None or dst is None:
                return {
                    "found": False,
                    "missing": [
                        n for n, r in ((from_name, src), (to_name, dst)) if r is None
                    ],
                }
            path = await graph_traverse.find_path(
                session,
                src["node_id"],
                dst["node_id"],
                max_hops=max_hops,
                as_of=as_of,
                user_id=user_id,
            )
            if path is None:
                return {"found": False, "hops": None}
            node_ids = [src["node_id"], dst["node_id"]]
            for e in path:
                node_ids.extend([e["src_id"], e["dst_id"]])
            nodes = await graph_traverse.fetch_nodes(session, node_ids)
        return {
            "found": True,
            "hops": len(path),
            "path": [graph_render.format_edge_line(e, nodes) for e in path],
        }

    async def memory_link(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        entity_types: list[str] | None = None,
        confidence: float | None = None,
        description: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Belief-save sugar: pre-resolve typed endpoints, save, return ids."""
        self._require_writable()
        if not graph_enabled():
            return {"disabled": True, "error": GRAPH_DISABLED_HINT,
                    "hint": "use memory_save for the belief without graph sugar"}
        src_type = entity_types[0] if entity_types and len(entity_types) > 0 else None
        dst_type = entity_types[1] if entity_types and len(entity_types) > 1 else None

        async def _attempt() -> dict[str, Any]:
            # One transaction for node resolution, the belief save, and the edge
            # lookup. Previously these were three separate transactions: if the
            # save raised (e.g. SecretRejected) the two nodes were left
            # orphaned, and the edge was guessed with ORDER BY rowid DESC, which
            # can return the wrong row under concurrent writers.
            async with self._session_factory() as session, session.begin():
                src_node = await resolve_node(
                    session, src, user_id=user_id, entity_type=src_type, create=True
                )
                dst_node = await resolve_node(
                    session, dst, user_id=user_id, entity_type=dst_type, create=True
                )
                record = await self._route_save(
                    session,
                    mem_type="belief",
                    subject=src,
                    relation=relation,
                    object=dst,
                    is_negation=False,
                    event_type=None,
                    name=None,
                    trigger_conditions=None,
                    steps=None,
                    entity_type=None,
                    properties=None,
                    description=description,
                    domain=None,
                    confidence=confidence,
                    source="user_direct",
                    user_id=user_id,
                )
                if confidence is not None:
                    record.confidence = max(0.0, min(1.0, float(confidence)))
                meta = dict(record.metadata_) if record.metadata_ else {}
                meta.setdefault("repo_id", _cached_repo_id())
                record.metadata_ = meta
                belief_id = canonical_id(record.memory_id)
                # Deterministic: the projection stamps belief_id = memory_id on
                # the edge it just created for this belief, in this transaction.
                edge_row = (
                    await session.execute(
                        sa_text(
                            "SELECT edge_id FROM graph_edges WHERE belief_id = :bid"
                        ),
                        {"bid": belief_id},
                    )
                ).first()
                return {
                    "belief_id": belief_id,
                    "edge_id": edge_row[0] if edge_row else None,
                    "src_node": src_node["display_name"] if src_node else src,
                    "dst_node": dst_node["display_name"] if dst_node else dst,
                    "action": "linked",
                }

        return await with_sqlite_retry(_attempt, what="memory_link")

    async def graph_rebuild(self, *, from_scratch: bool = False) -> dict[str, int]:
        """Regenerate graph tables from active memories (CLI entry point)."""
        async with self._session_factory() as session, session.begin():
            return await graph_project.rebuild(session, from_scratch=from_scratch)

    async def graph_merge(self, dup: str, canonical: str) -> dict[str, Any]:
        """Merge a duplicate node into its canonical node (auditable)."""
        from clara.graph.admin import merge_nodes

        if not graph_enabled():
            return {"merged": False, "disabled": True, "error": GRAPH_DISABLED_HINT}
        async with self._session_factory() as session, session.begin():
            return await merge_nodes(session, dup, canonical)

    async def graph_export(self, *, format: str = "mermaid") -> str:
        """Export valid graph edges as mermaid, dot, or json."""
        from clara.graph.admin import export_graph

        if not graph_enabled():
            raise RuntimeError(GRAPH_DISABLED_HINT)
        async with self._session_factory() as session:
            return await export_graph(session, format=format)

    # ------------------------------------------------------------------
    # Document verdict API (judgment layer — see clara/docs/verdicts.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _docs_disabled() -> dict[str, Any] | None:
        if docs_enabled():
            return None
        return {"found": False, "disabled": True, "error": DOCS_DISABLED_HINT}

    async def docs_classify(
        self, repo_root: str, *, path: str, doc_type: str,
        tier: str | None = None, rationale: str,
    ) -> dict[str, Any]:
        from clara.docs import verdicts

        if (disabled := self._docs_disabled()) is not None:
            return disabled
        return await verdicts.classify(
            self, repo_id(repo_root), path=path, doc_type=doc_type,
            tier=tier, rationale=rationale,
        )

    async def docs_supersede(
        self, repo_root: str, *, old_path: str, new_path: str, rationale: str,
    ) -> dict[str, Any]:
        from clara.docs import verdicts

        if (disabled := self._docs_disabled()) is not None:
            return disabled
        return await verdicts.supersede(
            self, repo_id(repo_root), old_path=old_path, new_path=new_path,
            rationale=rationale,
        )

    async def docs_fulfill(
        self, repo_root: str, *, path: str, distilled: list[dict[str, Any]],
        evidence: str | None = None, rationale: str = "plan completed",
    ) -> dict[str, Any]:
        from clara.docs import verdicts

        if (disabled := self._docs_disabled()) is not None:
            return disabled
        return await verdicts.fulfill(
            self, repo_id(repo_root), path=path, distilled=distilled,
            evidence=evidence, rationale=rationale,
        )

    async def docs_report(self, repo_root: str) -> dict[str, Any]:
        import sqlite3 as _sqlite3

        from clara.docs.report import build_report
        from clara.policy import load_policy

        if (disabled := self._docs_disabled()) is not None:
            return disabled

        def _build() -> dict[str, Any]:
            conn = _sqlite3.connect(self._db_path)
            conn.row_factory = _sqlite3.Row
            try:
                return build_report(conn, repo_id(repo_root), load_policy(repo_root))
            finally:
                conn.close()

        return await asyncio.to_thread(_build)

    async def docs_archive(
        self, repo_root: str, *, path: str, rationale: str = "archived by user",
    ) -> dict[str, Any]:
        from clara.docs import verdicts

        if (disabled := self._docs_disabled()) is not None:
            return disabled
        return await verdicts.archive(
            self, repo_id(repo_root), repo_root=repo_root, path=path,
            rationale=rationale,
        )

    async def docs_restore(
        self, repo_root: str, *, path: str, rationale: str = "restored by user",
    ) -> dict[str, Any]:
        from clara.docs import verdicts

        if (disabled := self._docs_disabled()) is not None:
            return disabled
        return await verdicts.restore(
            self, repo_id(repo_root), repo_root=repo_root, path=path,
            rationale=rationale,
        )
