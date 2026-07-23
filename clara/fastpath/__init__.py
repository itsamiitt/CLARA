"""
CLARA — fastpath: dependency-free session-start context injection.

This package is executed by the Claude Code plugin's SessionStart hook via
``python -m clara.fastpath.context`` and must start in tens of milliseconds.

HARD RULE: modules here may import only the cheap stdlib set
(``sqlite3, os, sys, json, hashlib, subprocess, pathlib, time``) plus the
stdlib-safe CLARA modules ``clara.repoid`` and ``clara.db.migrations``.
Never import SQLAlchemy, asyncio, or anything that pulls them — a test
enforces this (``tests/test_fastpath.py::test_import_purity``).
"""
