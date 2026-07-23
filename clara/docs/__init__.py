"""
CLARA — curator document ledger (visibility tier).

Deterministic, explainable signals about repository documents: what exists,
what it references, what rotted. No LLM calls anywhere in this package —
judgment (verdicts, promotion) is a later layer. `clara docs scan` is
idempotent and is the universal repair path; nothing here mutates repo files.

The ledger lives in the GLOBAL store and is keyed on ``repo_id`` so every
clone and worktree of a repository shares one ledger.
"""

# Retrieval weighting for memories carrying doc-derived provenance
# (metadata.doc_tier). TX is excluded from default context entirely.
# Keep in sync with the mirrors in clara/retrieval/lexical.py and
# clara/fastpath/context.py (a parity test enforces the fastpath one).
TIER_MULTIPLIER: dict[str, float] = {
    "T0": 1.2,
    "T1": 1.1,
    "T2": 1.0,
    "T3": 0.8,
}
