"""
CLARA — bidirectional bridge to Claude Code's native memory files.

CLARA is the primary structured store; the native side (CLAUDE.md and the
auto-memory directory ``~/.claude/projects/<project>/memory/``) is a
projection plus an import source:

- **export**: a marker-fenced section inside ``MEMORY.md`` (budgeted, so
  Claude's own notes keep the native 200-line window) and one fully-owned
  topic file ``clara-memory.md``.
- **import**: bullets from CLAUDE.md / MEMORY.md-outside-the-fence / topic
  files run through the heuristic extractor into the store, with provenance
  (``origin_file``, ``origin_hash``) so re-imports dedup and exports never
  echo a file's own lines back at it.

Loop prevention: fenced lines and ``<!--clara:...-->``-tagged lines are
never imported; memories originating from a file are excluded from exports
targeting that file; an edited fence is imported before it is regenerated.
"""
