"""
CLARA graph — name/relation normalization and in-tree string similarity.

Pure, synchronous helpers: no database access, no new dependencies. The
Jaro-Winkler implementation lives here so entity resolution needs nothing
outside the stdlib (non-negotiable for the zero-backend profile).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# '+' and '#' stay word characters so "c++" and "c#" do not collapse into "c".
_KEEP = {"+", "#"}


def normalize_name(name: str) -> str:
    """Casefold, punctuation → space (keeping ``+``/``#``), collapse whitespace."""
    folded = str(name).casefold()
    chars = [ch if (ch.isalnum() or ch in _KEEP) else " " for ch in folded]
    return " ".join("".join(chars).split())


# Built-in developer-vocabulary aliases, keyed and valued in *normalized* form.
# Merged with (and overridden by) the `aliases:` map from clara.yml.
SEED_ALIASES: dict[str, str] = {
    "postgres": "postgresql",
    "pg": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "py3": "python",
    "rb": "ruby",
    "golang": "go",
    "rs": "rust",
    "cpp": "c++",
    "csharp": "c#",
    "node": "nodejs",
    "node js": "nodejs",
    "reactjs": "react",
    "react js": "react",
    "vuejs": "vue",
    "vue js": "vue",
    "next js": "nextjs",
    "gh": "github",
    "gl": "gitlab",
    "tf": "terraform",
    "np": "numpy",
    "pd": "pandas",
    "sklearn": "scikit learn",
    "tailwindcss": "tailwind",
    "es": "elasticsearch",
    "elastic": "elasticsearch",
    "ddb": "dynamodb",
    "gcp": "google cloud",
    "vs code": "vscode",
    "vsc": "vscode",
    "sqlite3": "sqlite",
    "yml": "yaml",
    "md": "markdown",
    "dockerfile": "docker",
    "psql": "postgresql",
}


# ---------------------------------------------------------------------------
# Relation normalization
# ---------------------------------------------------------------------------

# Lemma groups: every listed variant maps to the canonical snake_case form.
#
# A group may only contain variants that mean the SAME thing in the SAME
# direction. Converse pairs ("A hosts B" vs "A hosted_on B") must never share a
# lemma: collapsing them keeps src/dst untouched and so silently reverses the
# claim. Leaving an unmatched relation un-normalised is always safer than
# recording the opposite of what the caller said.
_RELATION_GROUPS: dict[str, list[str]] = {
    "uses": ["use", "uses", "using", "used", "uses_the"],
    "depends_on": ["depend", "depends", "depend_on", "depends_on", "depending_on"],
    "deployed_to": ["deploy", "deploys", "deployed", "deploy_to", "deployed_to", "deploying_to"],
    "prefers": ["prefer", "prefers", "preferred"],
    "runs_on": ["run_on", "runs_on", "running_on", "ran_on"],
    "written_in": ["written_in", "write_in", "writes_in", "wrote_in"],
    # "A hosts B" (A is the host) is the converse of "A hosted_on B" (B is).
    "hosts": ["host", "hosts", "hosting", "hosted"],
    "hosted_on": ["hosted_on", "hosting_on"],
    "migrated_to": ["migrate", "migrated", "migrates", "migrate_to", "migrated_to", "migrating_to"],
    "owns": ["own", "owns", "owned"],
    "maintains": ["maintain", "maintains", "maintained"],
    "created": ["create", "creates", "created"],
    "works_on": ["work_on", "works_on", "working_on", "worked_on"],
    "likes": ["like", "likes", "liked"],
    "dislikes": ["dislike", "dislikes", "disliked"],
    "requires": ["need", "needs", "require", "requires", "required"],
    "connects_to": ["connect_to", "connects_to", "connected_to"],
    "belongs_to": ["belong_to", "belongs_to"],
    "part_of": ["part_of", "is_part_of"],
    "is_a": ["is", "is_a", "are", "was", "be"],
    "has": ["has", "have", "had"],
    # switched_from is the opposite of switched_to, not a spelling of it.
    "switched_to": ["switch_to", "switched_to", "switches_to"],
    "switched_from": ["switch_from", "switched_from", "switches_from"],
    "avoids": ["avoid", "avoids", "avoided"],
    "recommends": ["recommend", "recommends", "recommended"],
    "tests_with": ["test_with", "tests_with", "tested_with"],
    "built_with": ["build_with", "builds_with", "built_with"],
    "integrates_with": ["integrate_with", "integrates_with", "integrated_with"],
    "replaces": ["replace", "replaces", "replaced"],
    "stored_in": ["store_in", "stores_in", "stored_in"],
    "serves": ["serve", "serves", "served"],
    "manages": ["manage", "manages", "managed"],
}

_RELATION_LEMMAS: dict[str, str] = {
    variant: canonical
    for canonical, variants in _RELATION_GROUPS.items()
    for variant in variants
}


def normalize_relation(relation: str) -> str:
    """lowercase → snake_case → lemma map. Applied at edge write and when
    parsing a traversal ``relation`` filter."""
    snake = "_".join(str(relation).strip().lower().replace("-", " ").split())
    return _RELATION_LEMMAS.get(snake, snake)


# ---------------------------------------------------------------------------
# String similarity (in-tree, stdlib-only)
# ---------------------------------------------------------------------------


def jaro(a: str, b: str) -> float:
    """Jaro similarity in [0, 1]."""
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0
    window = max(len_a, len_b) // 2 - 1
    window = max(window, 0)

    match_a = [False] * len_a
    match_b = [False] * len_b
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - window)
        hi = min(len_b, i + window + 1)
        for j in range(lo, hi):
            if not match_b[j] and b[j] == ch:
                match_a[i] = True
                match_b[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    transpositions = 0
    j = 0
    for i in range(len_a):
        if match_a[i]:
            while not match_b[j]:
                j += 1
            if a[i] != b[j]:
                transpositions += 1
            j += 1
    transpositions //= 2

    return (
        matches / len_a + matches / len_b + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a: str, b: str, *, prefix_scale: float = 0.1) -> float:
    """Jaro-Winkler similarity: Jaro boosted by a shared prefix (max 4 chars)."""
    base = jaro(a, b)
    prefix = 0
    for ch_a, ch_b in zip(a[:4], b[:4], strict=False):
        if ch_a != ch_b:
            break
        prefix += 1
    return base + prefix * prefix_scale * (1.0 - base)


def prefix_score(a: str, b: str) -> float:
    """0.95 when the shorter string is a prefix of the longer and the length
    ratio is >= 0.6; else 0. Catches postgres/postgresql-style truncations."""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
        return 0.95
    return 0.0


def name_sim(a: str, b: str) -> float:
    """Similarity used by entity resolution: max(jaro_winkler, prefix_score)."""
    return max(jaro_winkler(a, b), prefix_score(a, b))
