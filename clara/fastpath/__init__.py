"""
CLARA — fastpath: dependency-free session-start context injection.

This package is executed by the Claude Code plugin's SessionStart hook via
``python -m clara.fastpath.context`` and must start in tens of milliseconds.

HARD RULE: modules here may import only the cheap stdlib set
(``sqlite3, os, sys, json, hashlib, subprocess, pathlib, time, re``) plus the
stdlib-safe CLARA modules ``clara.repoid``, ``clara.db.migrations`` and
``clara.project.detect``. Never import SQLAlchemy, asyncio, or anything that
pulls them — a test enforces this (``tests/test_fastpath.py::test_import_purity``).

``re`` is on the list because ``subprocess`` already imports it, so it costs
nothing extra here; ``clara.project.detect`` is stdlib-only by the same rule
and measured at ~2 ms on a ~160 ms cold start. Anything new belongs on this
list only with the same kind of measurement behind it.
"""
