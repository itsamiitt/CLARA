# Implementation log

One entry per milestone of the CLARA → Claude Code plugin conversion.

## 2026-07-23 — Milestone 04: curator doc ledger (Prompt 04)

Branch `feat/plugin-sprint0`, one commit
(`feat: curator doc ledger — signals, knowledge map, docs_status`).

**Built.**
- Migration 3: `doc_registry` (unique on repo_id+rel_path while valid),
  `doc_refs`, `doc_attestations`. Ledger keyed on `repo_id` in the GLOBAL
  store — worktrees/clones share one ledger (worktree test proves it).
- `clara/docs/`: `simhash.py` (in-tree 64-bit simhash, 3-word shingles,
  cluster at Hamming ≤ 6), `refs.py` (md links, inline-code repo paths,
  `path::symbol`, urls, issue ids; code fences stripped), `signals.py`
  (one-sentence docstring per signal: age_days, last_author,
  churn_divergence "targets:doc", dead_refs incl. capped `git grep -w`
  symbol probes, checkbox_state, merge_evidence, path_convention from
  clara.yml + built-ins, frontmatter status/type/tier/supersedes — one
  batched `git log --name-only` parse feeds all git signals), `scan.py`
  (hash-gated incremental scan; rename keeps doc_id via content hash +
  `moved` attestation; vanish invalidates; frontmatter `supersedes:` marks
  targets superseded; repo-wide link_indegree + duplicate clusters;
  quarantine manifest `$CLARA_HOME/quarantine/<repo_id>.tsv`; graph
  projection of document nodes + describes→file edges feature-detected),
  `report.py` (status sentence + rot report — proposals only, T0/T1 never
  archive candidates).
- Fastpath: `docs_map.py` renders the fenced `[KNOWLEDGE MAP]` after
  `[MEMORY CONTEXT]` (raw SQL, ≤300 tokens, 120-char lines, control chars
  stripped, paths verbatim-quoted, never content excerpts; cold-start
  notice line), runs the `git status --porcelain -- '*.md'` dirty-check
  into a sidecar consumed by `clara docs scan --changed`, and refreshes
  the quarantine manifest when the ledger stamp changed. Import purity
  extended to docs_map (+ yaml guard).
- Tier weighting: memories with `metadata.doc_tier` get multipliers
  (T0 1.2 / T1 1.1 / T2 1.0 / T3 0.8) and TX is excluded — implemented in
  `LexicalRetriever` (rich path) and fastpath `rank()` (mirror; parity
  test), constants in `clara.docs.TIER_MULTIPLIER`.
- CLI `clara docs scan|status|report` (scan warns when clara.yml is
  git-ignored; `--changed` with no paths consumes the dirty sidecar); MCP
  tool `docs_status(path_or_query, repo=None)` resolving repo via MCP
  roots then cwd; `/clara:docs`; SKILL.md "Document trust" section.
- Tests: +21 (`test_docs.py`) — rotting-repo fixture (30 docs) with 12/12
  planted rot found (≥90% required), zero archive proposals on T1,
  frontmatter typing, rename keeps doc_id, incremental rescan <100 ms
  (min-of-3, measured ~60 ms), scan idempotence, worktree shared ledger,
  knowledge-map sanitization (adversarial `IGNORE PREVIOUS
  INSTRUCTIONS.md` renders quoted inside the fence), budget, cold-start
  notice, dirty sidecar round-trip, lexical + fastpath tier weighting.

**Deviations from the prompt.**
1. "Mark dirty docs in a lightweight fastpath table or sidecar" — sidecar
   file chosen (`$CLARA_HOME/dirty/<repo_id>.list`); no new table.
2. Tier weighting implemented in the lexical path (the zero-backend
   retriever the plugin uses) and fastpath; the LanceDB vector path is
   untouched this milestone.
3. `_discover` additionally skips hidden directories (except `.github`) —
   `.pytest_cache/README.md` etc. are never documentation.
4. Dead-ref heuristics have known benign false positives (branch names,
   npm package ids that look path-like); deterministic and explainable,
   left as-is.

## 2026-07-22 — Milestone 03: knowledge graph layer (Prompt 03)

Branch `feat/plugin-sprint0`, one commit
(`feat: knowledge graph layer — projection, traversal, MCP tools`).

**Built.**
- Migration 2 (`clara/db/migrations.py`): `graph_nodes` (unique active
  identity on coalesce(user_id,'')/entity_type/canonical_name),
  `graph_aliases`, `graph_edges` (partial valid-edge indexes + belief index),
  `graph_nodes_fts` (trigram, created fail-soft).
- `clara/graph/normalize.py`: name normalization (keeps `+`/`#` so c++/c#
  survive), ~40 seed dev aliases, ~30-lemma relation normalizer, in-tree
  Jaro-Winkler + prefix_score (unit-tested against known pairs).
- `clara/graph/resolve.py`: normalize → exact canonical → alias table →
  seed+clara.yml aliases → trigram-FTS fuzzy (OR-of-sampled-trigrams for
  recall; precision from name_sim ≥ 0.90). Multi-candidate ⇒ NEW node with
  `properties.possible_duplicates`; `user`/`skill` singletons expandable=0.
- `clara/graph/project.py`: fail-soft projection (decorator swallows + logs);
  belief→edge, negation→invalidate matching, supersede→invalid_at +
  provenance, confidence mirror, forget→invalidate, world_model→node upsert
  linking world_model_id (+entity_type promotion from 'concept'); loud
  `rebuild(from_scratch)` replaying active memories in insertion order with
  `temporal_precision='reconstructed'` (idempotent, tested).
- `clara/graph/traverse.py`: recursive CTE, fan-out capped in SQL via
  ROW_NUMBER per src/dst (hub with 5,000 edges returns exactly `fanout`
  edges in <50 ms — timed test), expandable gating, edge-id path cycle
  guard, `as_of` temporal filter, score = decay^depth × confidence,
  traversed-and-kept bumps (weight +0.05, mention_count +1), `find_path`.
- `clara/graph/render.py`: `[GRAPH]` section, ✗ + date range for
  invalidated, `~` for reconstructed dates, ≤400-token budget.
- `clara/graph/maintain.py`: hub flags, decay mirror, belief-inactive +
  below-threshold archiving, stale degree-0 prune (status change — never
  delete); wired into the opportunistic maintenance pass in mcp_server.
- Hooks: `BeliefMemory.store/update/supersede`, `WorldModelStore.upsert`,
  `MemoryUpdateEngine._store_fact` (world_model branch), `LocalMemory`
  forget/update; `LocalMemory.create` now runs versioned migrations
  (fail-soft, via thread) — the milestone-02 wiring promise.
- MCP: `graph_entity`, `graph_neighbors`, `graph_path`, `memory_link`;
  `memory_search(graph_depth=1)` appends [GRAPH]; `memory_stats` graph
  counts; `memory_forget` invalidates edges.
- CLI `clara graph rebuild|stats|show|path|doctor`; `commands/graph.md`
  (`/clara:graph`); SKILL.md gained the Linking section.
- Tests: +42 (test_graph.py 38, TestGraphCli 3 in test_cli.py, migration
  updates). All spec fixtures pass: postgres/postgresql one node, hub
  stress, fault injection (projection exception never fails memory_save),
  rebuild count parity, supersede invalid_at, as_of pre-supersede edge.

**Deviations from the prompt.**
1. Projection hooks live at the `BeliefMemory`/`WorldModelStore` layer (plus
   the engine's direct world_model branch) instead of literally inside
   `LocalMemory.save`/`MemoryUpdateEngine.process` — both named paths route
   through these stores, so one hook surface covers CLI, MCP, and LLM
   pipelines uniformly.
2. Fuzzy FTS uses OR-of-sampled-trigrams rather than a phrase MATCH — a
   phrase demands a contiguous substring and missed e.g.
   authentification/authenticatio; precision still comes from the
   name_sim ≥ 0.90 gate.
3. Known sharp edge (spec-conformant): near-identical short names
   (svc-a/svc-b) legitimately exceed 0.90 via the prefix rule and merge as
   single-candidate matches. Graph tests document this; the merge-review
   flow (possible_duplicates) is the escape hatch.
4. `/clara:graph` full E2E (live MCP tools) needs a POSIX shell for the
   plugin's .sh launcher; on this Windows dev box the DoD render was proven
   by driving `graph_entity` + `graph_neighbors` directly (card + [GRAPH]
   neighborhood for svelte) and the plugin load smoke stays green.

## 2026-07-22 — Milestone 02: base plugin (Prompt 02)

Branch `feat/plugin-sprint0`, one commit
(`feat: base Claude Code plugin — manifest, bootstrap, hooks, fastpath`).

**Built.**
- `.claude-plugin/plugin.json` + `marketplace.json`: plugin `clara`, inline
  MCP server `memory` → `scripts/clara-mcp-launch.sh`, hooks pointer.
  `claude plugin validate --strict .` passes (CLI 2.1.218).
- `hooks/hooks.json`: one SessionStart hook (`startup|resume|compact`) →
  `session-start.sh`.
- `scripts/bootstrap.sh`: POSIX sh, stderr-only; Python gate (>=3.11),
  pyproject-hash-versioned venvs under `$CLAUDE_PLUGIN_DATA`, detached
  nohup install (uv preferred, pip fallback), mkdir lock, 15-min stale flag,
  atomic `ln -sfn` swap, keep-2 GC. Exit 0/1/3.
- `scripts/clara-mcp-launch.sh`: bootstrap → poll ≤20 s on exit 3 → exec
  clara-mcp; stdout carries only the MCP protocol.
- `scripts/session-start.sh`: bootstrap → install notice on exit 3 → one
  fastpath invocation; always exits 0.
- `clara/fastpath/` (stdlib-only): store resolution (repo `.clara/clara.db`
  → `$CLARA_DB_PATH`/`$CLARA_HOME`/`~/.clara`), schema check via
  `ensure_schema` (too-new ⇒ silent + stderr), raw-SQL recall mirroring
  `LocalMemory.recent` (weights 0.20/0.10/0.05, λ=0.01, top 12), formatting
  identical to `format_context` (subprocess parity test proves byte-equality),
  ≤900-token budget dropping oldest first.
- `LocalMemory.save` stamps `metadata.repo_id` (cached per cwd) on new writes.
- Skill `using-clara-memory`, commands `/remember /recall /memories /forget`,
  `MANIFEST.in` pruning plugin dirs from dists, `.gitattributes` (`*.sh` LF),
  plugin.yml: validate + shellcheck + exec-bit + fastpath purity + tests.
- Tests: +35 (`test_fastpath.py` 22, `test_plugin_layout.py` 13). E2E verified
  via Git Bash: first session prints the install notice, background install
  completes, second session emits `=== MEMORY CONTEXT ===`, launcher execs
  clara-mcp cleanly; `claude --plugin-dir .` loads the plugin.

**Deviations from the prompt.**
1. `plugin.json` carries `"version": "0.1.0"` — the spec wanted commit-SHA
   versioning (no field), but `claude plugin validate --strict` fails on a
   missing version, and the DoD requires strict validation to pass.
2. Python gate additionally tries plain `python` after `python3` (Windows/
   Git-Bash setups expose only `python`; the >=3.11 check still gates).
3. Scripts probe `bin/` then `Scripts/` venv layouts (`find_bin`) so native
   Windows works best-effort; on MSYS `ln -sfn` degrades to a copy, noted in
   bootstrap comments. POSIX behavior is unchanged.
4. Fastpath imports the stdlib-safe CLARA modules `clara.repoid` and
   `clara.db.migrations` (the spec's "imported from a stdlib-safe module"
   option); `urllib.request` in migrations became a lazy import inside
   `open_db` to keep the fastpath import graph lean. No `math`/`datetime`/
   `re` in fastpath: `E ** x` for exp, Newton-on-exp for the usage log ratio,
   SQLite `strftime('%s', …)` for timestamps, and a scanner port of
   `sanitize_memory_text` locked by a parity test.

## 2026-07-22 — Sprint 0: conventions locked + de-risking spikes (Prompt 01)

Branch `feat/plugin-sprint0`, single commit
(`feat: plugin sprint 0 — policy loader, repo identity, schema versioning`).

**Built.**
- `clara/policy.py` + `clara.yml.example`: repo-root `clara.yml` policy loader
  (frozen dataclass mirroring `ClaraConfig`; defaults when absent; top-level
  replace, unknown keys ignored, malformed input fails loud). `pyyaml` joined
  core dependencies; `types-PyYAML` joined dev extras.
- `clara/repoid.py`: 16-hex repo identity — sha256 of root commit, falling
  back to normalized `remote.origin.url`, then normalized realpath. Worktree
  equality verified by test.
- `clara/db/migrations.py`: forward-only SQLite schema versioning
  (`schema_info` table, per-migration `BEGIN IMMEDIATE` transactions,
  idempotent `ensure_schema`); newer-than-code DBs raise `SchemaTooNew` before
  any write, and `open_db` reopens them read-only (URI `mode=ro` +
  `PRAGMA query_only`) with one stderr warning.
- Gitignore fix in `clara/cli.py::_resolve_db_path`: project `.clara/.gitignore`
  now `clara.db*` / `.maintenance` / `quarantine/` instead of `*`; an existing
  file containing exactly `*` is rewritten with a one-line stderr warning.
- `.github/workflows/plugin.yml`: `claude plugin validate --strict .` stub,
  gated on the milestone-02 manifest.
- `README.md`: "Supported platforms" section (macOS/Linux/WSL at launch).
- `docs/plans/sprint0-findings.md`: S1–S5 answered with evidence (live
  PostToolUse `additionalContext` proof, FileChanged docs, latency numbers,
  validate output).
- Tests: +37 (`test_policy.py` 14, `test_repoid.py` 12,
  `test_db_migrations.py` 8, `TestProjectGitignore` 3 in `test_cli.py`).

**Deviations from the prompt.**
1. The two dead PostgreSQL migrations were relocated
   (`clara/db/migrations/*.sql` → `docs/history/legacy-postgres-migrations/`)
   to make room for `clara/db/migrations.py`. The prompt did not anticipate
   the name collision; relocation (not deletion) preserves the artifacts
   in-tree. Note: the directory had no `__init__.py`, so a real module would
   have won import resolution anyway — the move is hygiene, not necessity.
2. S2's premise ("if unsupported...") is outdated: a `FileChanged` hook event
   exists (filename regex matcher, not glob). Git dirty-check remains the
   curator default by choice; findings record the real capability and cost.
3. `ensure_schema` is intentionally NOT wired into the existing runtime paths
   (SQLAlchemy `create_all` remains the live schema path) — the milestone's
   non-negotiable forbids runtime behavior changes beyond the gitignore fix.
   Wiring lands with the milestone-02 plugin bootstrap.
