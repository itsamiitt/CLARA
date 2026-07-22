# Implementation log

One entry per milestone of the CLARA → Claude Code plugin conversion.

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
