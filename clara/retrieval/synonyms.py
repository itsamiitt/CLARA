"""
CLARA — query-side synonym expansion (developer vocabulary).

Porter stemming already handles inflection ("deploying" ~ "deployed"); the
real recall gap is abbreviation — "postgres" vs "postgresql", "k8s" vs
"kubernetes". A tiny curated map fixes that with zero model downloads.

Expansion is *query-side only* (the index stores original text), applied as
extra OR terms: BM25's IDF keeps a synonym from dominating rows that match
the literal term. Kill switch: ``CLARA_QUERY_EXPANSION=0``.
"""

from __future__ import annotations

import os

_FALSY = {"0", "false", "no", "off"}

# Each tuple is one equivalence group; membership is bidirectional.
_GROUPS: list[tuple[str, ...]] = [
    ("postgres", "postgresql", "pg"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("py", "python"),
    ("k8s", "kubernetes"),
    ("db", "database"),
    ("auth", "authentication"),
    ("authz", "authorization"),
    ("config", "configuration"),
    ("env", "environment"),
    ("repo", "repository"),
    ("dir", "directory"),
    ("docs", "documentation"),
    ("dev", "development"),
    ("prod", "production"),
    ("perf", "performance"),
    ("api", "endpoint"),
    ("cli", "commandline"),
    ("ui", "frontend"),
    ("backend", "server"),
    ("npm", "node"),
    ("venv", "virtualenv"),
    ("regex", "regexp"),
    ("param", "parameter"),
    ("arg", "argument"),
    ("func", "function"),
    ("var", "variable"),
    ("msg", "message"),
    ("err", "error"),
    ("deps", "dependencies"),
    ("pkg", "package"),
    ("cfg", "configuration"),
    ("mem", "memory"),
    ("infra", "infrastructure"),
    ("vm", "virtualmachine"),
    ("ci", "pipeline"),
    ("pr", "pullrequest"),
    ("wsl", "windowssubsystem"),
    ("cwd", "workingdirectory"),
    ("sqlite", "sqlite3"),
]

_SYNONYMS: dict[str, frozenset[str]] = {}
for _group in _GROUPS:
    members = frozenset(_group)
    for _term in _group:
        _SYNONYMS[_term] = _SYNONYMS.get(_term, frozenset()) | members


def expansion_enabled() -> bool:
    raw = os.environ.get("CLARA_QUERY_EXPANSION", "").strip().lower()
    return not raw or raw not in _FALSY


def expand_groups(tokens: list[str]) -> list[set[str]]:
    """One set per query token: the token plus its synonyms (if any).

    A memory "matches" a group when it contains *any* member — so the
    similarity denominator stays the number of concepts the user typed, not
    the number of expansion terms.
    """
    if not expansion_enabled():
        return [{tok} for tok in tokens]
    return [{tok, *(_SYNONYMS.get(tok, ()))} for tok in tokens]


def flatten(groups: list[set[str]]) -> list[str]:
    """Deduplicated, order-preserving flat token list for FTS/scan filters."""
    seen: dict[str, None] = {}
    for group in groups:
        for term in sorted(group):
            seen.setdefault(term, None)
    return list(seen)
