"""Tests for the knowledge-graph layer (clara/graph/)."""

from __future__ import annotations

import json
import sys
import time
import uuid

import pytest
from sqlalchemy import text as sa_text

from clara.graph import project as graph_project
from clara.graph.normalize import (
    jaro_winkler,
    name_sim,
    normalize_name,
    normalize_relation,
    prefix_score,
)
from clara.graph.render import GRAPH_TOKEN_BUDGET, render_graph_section
from clara.graph.resolve import create_node, resolve_node
from clara.graph.traverse import traverse
from clara.integrations.local_memory import LocalMemory


async def _store(tmp_path) -> LocalMemory:
    return await LocalMemory.create(str(tmp_path / "graph.db"))


async def _edges(memory: LocalMemory, where: str = "1=1") -> list[dict]:
    async with memory._session_factory() as session:
        rows = (
            await session.execute(sa_text(f"SELECT * FROM graph_edges WHERE {where}"))
        ).mappings().all()
    return [dict(r) for r in rows]


async def _nodes(memory: LocalMemory, where: str = "status = 'active'") -> list[dict]:
    async with memory._session_factory() as session:
        rows = (
            await session.execute(sa_text(f"SELECT * FROM graph_nodes WHERE {where}"))
        ).mappings().all()
    return [dict(r) for r in rows]


class TestNormalize:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("martha", "marhta", 0.9611),
            ("dixon", "dicksonx", 0.8133),
            ("dwayne", "duane", 0.8400),
        ],
    )
    def test_jaro_winkler_known_pairs(self, a, b, expected):
        assert abs(jaro_winkler(a, b) - expected) < 0.001

    def test_jaro_empty(self):
        assert jaro_winkler("", "x") == 0.0

    def test_prefix_score(self):
        assert prefix_score("postgres", "postgresql") == 0.95
        assert prefix_score("post", "postgresql") == 0.0  # ratio < 0.6
        assert name_sim("postgres", "postgresql") >= 0.9

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Depends On", "depends_on"),
            ("using", "uses"),
            ("deployed to", "deployed_to"),
            ("Prefers", "prefers"),
            ("frobnicates", "frobnicates"),
        ],
    )
    def test_normalize_relation(self, raw, expected):
        assert normalize_relation(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("PostgreSQL!", "postgresql"),
            ("C++", "c++"),
            ("Node.js", "node js"),
            ("  spaced   out  ", "spaced out"),
        ],
    )
    def test_normalize_name(self, raw, expected):
        assert normalize_name(raw) == expected


class TestResolution:
    async def test_postgres_and_postgresql_one_node(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="postgres")
        await memory.save(mem_type="belief", subject="api", relation="depends on",
                          object="PostgreSQL")
        nodes = await _nodes(memory, "canonical_name = 'postgresql'")
        await memory.close()
        assert len(nodes) == 1

    async def test_user_singleton_not_expandable(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="pytest")
        nodes = await _nodes(memory, "entity_type = 'user'")
        await memory.close()
        assert len(nodes) == 1
        assert nodes[0]["expandable"] == 0

    async def test_multiple_fuzzy_candidates_create_new_with_duplicates(self, tmp_path):
        memory = await _store(tmp_path)
        async with memory._session_factory() as session, session.begin():
            a = await create_node(session, canonical="authentication",
                                  display="authentication")
            b = await create_node(session, canonical="authentification",
                                  display="authentification")
            node = await resolve_node(session, "authenticatio")
        await memory.close()
        assert node is not None and node.get("_created")
        dupes = set(json.loads(node["properties"])["possible_duplicates"])
        assert dupes == {a["node_id"], b["node_id"]}

    async def test_single_fuzzy_candidate_reused_and_aliased(self, tmp_path):
        memory = await _store(tmp_path)
        async with memory._session_factory() as session, session.begin():
            a = await create_node(session, canonical="sveltekit",
                                  display="SvelteKit")
            node = await resolve_node(session, "svelte kit")
            aliases = (
                await session.execute(
                    sa_text("SELECT alias_norm FROM graph_aliases WHERE node_id = :n"),
                    {"n": a["node_id"]},
                )
            ).scalars().all()
        await memory.close()
        assert node is not None and node["node_id"] == a["node_id"]
        assert "svelte kit" in aliases


class TestProjection:
    async def test_belief_creates_edge(self, tmp_path):
        memory = await _store(tmp_path)
        saved = await memory.save(mem_type="belief", subject="user",
                                  relation="prefers", object="ripgrep")
        edges = await _edges(memory)
        await memory.close()
        assert len(edges) == 1
        edge = edges[0]
        assert edge["belief_id"] == saved["memory_id"]
        assert edge["relation"] == "prefers"
        assert edge["invalid_at"] is None
        assert edge["temporal_precision"] == "exact"

    async def test_negation_invalidates_matching_edge(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="npm")
        neg = await memory.save(mem_type="belief", subject="user", relation="uses",
                                object="npm", is_negation=True)
        edges = await _edges(memory)
        await memory.close()
        assert len(edges) == 1  # negation creates no edge of its own
        assert edges[0]["invalid_at"] is not None
        assert json.loads(edges[0]["metadata"])["invalidated_by"] == neg["memory_id"]

    async def test_forget_invalidates_edge(self, tmp_path):
        memory = await _store(tmp_path)
        saved = await memory.save(mem_type="belief", subject="user",
                                  relation="uses", object="poetry")
        await memory.forget(saved["memory_id"])
        edges = await _edges(memory)
        await memory.close()
        assert edges[0]["invalid_at"] is not None

    async def test_confidence_update_mirrors_to_edge(self, tmp_path):
        memory = await _store(tmp_path)
        saved = await memory.save(mem_type="belief", subject="user",
                                  relation="uses", object="uv")
        await memory.update(saved["memory_id"], confidence=0.33)
        edges = await _edges(memory)
        await memory.close()
        assert abs(edges[0]["confidence"] - 0.33) < 1e-9

    async def test_world_model_links_node(self, tmp_path):
        memory = await _store(tmp_path)
        saved = await memory.save(mem_type="world_model", entity_type="service",
                                  name="api", properties={"host": "fly.io"})
        nodes = await _nodes(memory, "canonical_name = 'api'")
        await memory.close()
        assert len(nodes) == 1
        assert nodes[0]["world_model_id"] == saved["memory_id"]
        assert json.loads(nodes[0]["properties"])["host"] == "fly.io"

    async def test_supersede_sets_invalid_at(self, tmp_path):
        from clara.memory.belief import BeliefMemory

        memory = await _store(tmp_path)
        saved = await memory.save(mem_type="belief", subject="user",
                                  relation="uses", object="webpack")
        async with memory._session_factory() as session, session.begin():
            beliefs = BeliefMemory(session)
            await beliefs.supersede(
                uuid.UUID(saved["memory_id"]),
                subject="user", relation="uses", object_="vite",
            )
        old_edges = await _edges(memory, f"belief_id = '{saved['memory_id']}'")
        valid = await _edges(memory, "invalid_at IS NULL")
        await memory.close()
        assert old_edges[0]["invalid_at"] is not None
        assert len(valid) == 1  # the vite edge


class TestFaultInjection:
    async def test_projection_error_never_fails_memory_save(self, tmp_path, caplog):
        memory = await _store(tmp_path)

        async def boom(*args, **kwargs):
            raise RuntimeError("graph exploded")

        original = graph_project.resolve_node
        graph_project.resolve_node = boom  # type: ignore[assignment]
        try:
            saved = await memory.save(mem_type="belief", subject="user",
                                      relation="uses", object="doomed")
        finally:
            graph_project.resolve_node = original  # type: ignore[assignment]
        edges = await _edges(memory)
        await memory.close()
        assert saved["action"] == "saved"
        assert edges == []
        assert any("graph projection failed" in r.message for r in caplog.records)


class TestRebuild:
    async def test_rebuild_reproduces_counts(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="postgresql")
        await memory.save(mem_type="belief", subject="api", relation="depends_on",
                          object="postgresql")
        await memory.save(mem_type="world_model", entity_type="service",
                          name="api", properties={"host": "fly.io"})
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="postgresql", is_negation=True)

        before_nodes = len(await _nodes(memory))
        before_valid = len(await _edges(memory, "invalid_at IS NULL"))

        async with memory._session_factory() as session, session.begin():
            for table in ("graph_edges", "graph_aliases", "graph_nodes"):
                await session.execute(sa_text(f"DELETE FROM {table}"))

        counts = await memory.graph_rebuild()
        assert counts["nodes"] == before_nodes
        assert counts["edges"] == before_valid
        rebuilt = await _edges(memory)
        assert all(e["temporal_precision"] == "reconstructed" for e in rebuilt)

        again = await memory.graph_rebuild()
        await memory.close()
        assert again["nodes"] == counts["nodes"]
        assert again["edges"] == counts["edges"]
        assert again["edges_created"] == 0  # idempotent


class TestTraversal:
    # Chain entity names are deliberately dissimilar — near-identical names
    # (svc-a/svc-b) legitimately merge under the 0.90 name_sim rule.
    async def _chain(self, memory: LocalMemory) -> None:
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="gateway")
        await memory.save(mem_type="belief", subject="gateway", relation="depends_on",
                          object="redis")
        await memory.save(mem_type="belief", subject="redis", relation="runs_on",
                          object="linode")

    async def test_depth_and_scores(self, tmp_path):
        memory = await _store(tmp_path)
        await self._chain(memory)
        async with memory._session_factory() as session:
            node = await resolve_node(session, "user", create=False, bump_mention=False)
            rows = await traverse(session, [node["node_id"]], depth=2)
        await memory.close()
        assert {r["relation"] for r in rows} == {"uses", "depends_on"}
        by_rel = {r["relation"]: r for r in rows}
        assert by_rel["uses"]["depth"] == 1
        assert by_rel["depends_on"]["depth"] == 2
        assert by_rel["uses"]["score"] > by_rel["depends_on"]["score"]

    async def test_relation_filter_normalizes(self, tmp_path):
        memory = await _store(tmp_path)
        await self._chain(memory)
        async with memory._session_factory() as session:
            node = await resolve_node(session, "gateway", create=False, bump_mention=False)
            rows = await traverse(session, [node["node_id"]], depth=1,
                                  relation="Depends On")
        await memory.close()
        assert len(rows) == 1
        assert rows[0]["relation"] == "depends_on"

    async def test_non_expandable_node_stops_expansion(self, tmp_path):
        memory = await _store(tmp_path)
        await self._chain(memory)
        async with memory._session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_text("UPDATE graph_nodes SET expandable = 0 "
                            "WHERE canonical_name = 'gateway'")
                )
            node = await resolve_node(session, "user", create=False, bump_mention=False)
            rows = await traverse(session, [node["node_id"]], depth=2)
        await memory.close()
        assert {r["relation"] for r in rows} == {"uses"}

    async def _hub_with_edges(self, tmp_path, count: int = 5000):
        """Store with one hub node fanning out to *count* edges."""
        memory = await _store(tmp_path)
        async with memory._session_factory() as session, session.begin():
            hub = await create_node(session, canonical="hub", display="hub")
            params = []
            for i in range(count):
                params.append({
                    "eid": f"edge{i:05d}",
                    "src": hub["node_id"],
                    "dst": uuid.uuid4().hex,
                    "conf": 0.5 + (i % 100) / 250.0,
                    "vfrom": "2026-01-01 00:00:00",
                })
            await session.execute(
                sa_text(
                    "INSERT INTO graph_edges (edge_id, src_id, dst_id, relation, "
                    "confidence, weight, valid_from) VALUES (:eid, :src, :dst, "
                    "'uses', :conf, 1.0, :vfrom)"
                ),
                params,
            )
        return memory, hub

    async def test_hub_fanout_capped(self, tmp_path):
        # Correctness only — deterministic, so it belongs in the default tier.
        # The latency budget lives in the bench tier (see test_bench.py): a
        # wall-clock assertion here flakes under full-suite load on a busy or
        # shared runner, which is exactly what the bench marker exists for.
        memory, hub = await self._hub_with_edges(tmp_path)
        async with memory._session_factory() as session:
            rows = await traverse(session, [hub["node_id"]], depth=1, fanout=6)
        await memory.close()
        assert len(rows) == 6

    @pytest.mark.bench
    async def test_hub_fanout_fast(self, tmp_path):
        # Order-of-magnitude guard against a scan sneaking into the hub path.
        # Windows spawn/filesystem overhead runs 2-4x POSIX, matching the
        # budget scaling convention in test_bench.py.
        budget = 0.20 if sys.platform == "win32" else 0.05
        memory, hub = await self._hub_with_edges(tmp_path)
        async with memory._session_factory() as session:
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                rows = await traverse(session, [hub["node_id"]], depth=1, fanout=6)
                timings.append(time.perf_counter() - start)
        await memory.close()
        assert len(rows) == 6
        assert min(timings) < budget, f"hub traversal too slow: {min(timings):.3f}s"

    async def test_as_of_returns_pre_supersede_edge(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="api", relation="uses",
                          object="mysql")
        async with memory._session_factory() as session, session.begin():
            await session.execute(
                sa_text("UPDATE graph_edges SET valid_from = '2026-01-01 00:00:00'")
            )
        await memory.save(mem_type="belief", subject="api", relation="uses",
                          object="mysql", is_negation=True)
        async with memory._session_factory() as session:
            node = await resolve_node(session, "api", create=False, bump_mention=False)
            now_rows = await traverse(session, [node["node_id"]], depth=1)
            past_rows = await traverse(session, [node["node_id"]], depth=1,
                                       as_of="2026-03-01 00:00:00")
        await memory.close()
        assert now_rows == []
        assert len(past_rows) == 1
        assert past_rows[0]["relation"] == "uses"


class TestGraphApi:
    async def test_memory_link_returns_ids(self, tmp_path):
        memory = await _store(tmp_path)
        result = await memory.memory_link(
            "src/api.py", "depends_on", "postgresql",
            entity_types=["file", "tool"], confidence=0.9,
        )
        await memory.close()
        assert result["belief_id"]
        assert result["edge_id"]
        assert result["src_node"] == "src/api.py"

    async def test_search_graph_depth_appends_section(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="api", relation="depends_on",
                          object="postgresql")
        await memory.save(mem_type="belief", subject="postgresql",
                          relation="runs_on", object="rds")
        result = await memory.search("postgresql", graph_depth=1)
        plain = await memory.search("postgresql", graph_depth=0)
        await memory.close()
        assert "[GRAPH]" in result["context"]
        assert result["graph"]["edges"]
        assert "[GRAPH]" not in plain["context"]

    async def test_stats_include_graph_counts(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="sqlite")
        stats = await memory.stats()
        await memory.close()
        assert stats["graph"]["edges"] == 1
        assert stats["graph"]["nodes"] >= 2

    async def test_graph_entity_card(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="user", relation="uses",
                          object="postgres")
        card = await memory.graph_entity("pg")
        missing = await memory.graph_entity("nonexistent-thing")
        await memory.close()
        assert card["found"]
        assert card["canonical_name"] == "postgresql"
        assert card["edges"]
        assert missing["found"] is False

    async def test_graph_path_two_hops(self, tmp_path):
        memory = await _store(tmp_path)
        await memory.save(mem_type="belief", subject="frontend", relation="uses",
                          object="api-service")
        await memory.save(mem_type="belief", subject="api-service",
                          relation="depends_on", object="postgresql")
        result = await memory.graph_path("frontend", "postgresql")
        await memory.close()
        assert result["found"]
        assert result["hops"] == 2


class TestRender:
    def test_budget_enforced(self):
        nodes = {}
        edges = []
        for i in range(200):
            src, dst = f"n{i}a", f"n{i}b"
            nodes[src] = {"display_name": f"very-long-entity-name-number-{i:03d}"}
            nodes[dst] = {"display_name": f"another-long-entity-name-{i:03d}"}
            edges.append({
                "edge_id": f"e{i}", "src_id": src, "dst_id": dst,
                "relation": "depends_on", "confidence": 0.8,
                "invalid_at": None, "valid_from": "2026-01-01",
                "temporal_precision": "exact",
            })
        block = render_graph_section([("seed", edges)], nodes)
        assert block.startswith("[GRAPH]")
        assert (len(block) + 3) // 4 <= GRAPH_TOKEN_BUDGET

    def test_invalidated_and_reconstructed_markers(self):
        nodes = {
            "a": {"display_name": "api"},
            "b": {"display_name": "mysql"},
        }
        dead = {
            "edge_id": "e1", "src_id": "a", "dst_id": "b", "relation": "uses",
            "confidence": 0.8, "invalid_at": "2026-02-11 00:00:00",
            "valid_from": "2025-01-01 00:00:00", "temporal_precision": "exact",
        }
        rebuilt = {
            "edge_id": "e2", "src_id": "a", "dst_id": "b", "relation": "uses",
            "confidence": 0.8, "invalid_at": None,
            "valid_from": "2026-06-01 00:00:00",
            "temporal_precision": "reconstructed",
        }
        block = render_graph_section([("api", [dead, rebuilt])], nodes)
        assert "✗ api uses mysql (2025-01-01 → 2026-02-11)" in block
        assert "since ~2026-06-01" in block
