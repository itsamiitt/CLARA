"""
CLARA docs — 64-bit simhash for near-duplicate detection (in-tree, stdlib).

Documents are normalized (lowercase, whitespace collapsed), shingled into
3-word windows, and each shingle's 64-bit hash votes bit-wise; the sign of
each bit's vote total forms the fingerprint. Two documents are near-dupes
when the Hamming distance between fingerprints is small (scan clusters at
<= 6).
"""

from __future__ import annotations

import hashlib

SHINGLE_WORDS = 3
HAMMING_THRESHOLD = 6
_BITS = 64


def _normalize(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


def shingles(text: str, size: int = SHINGLE_WORDS) -> list[str]:
    """Overlapping *size*-word windows of the normalized text."""
    words = _normalize(text)
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(text: str) -> int:
    """64-bit simhash fingerprint of *text* (0 for empty/degenerate input)."""
    grams = shingles(text)
    if not grams:
        return 0
    votes = [0] * _BITS
    for gram in grams:
        h = _hash64(gram)
        for bit in range(_BITS):
            votes[bit] += 1 if (h >> bit) & 1 else -1
    fingerprint = 0
    for bit in range(_BITS):
        if votes[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit fingerprints."""
    return bin(a ^ b).count("1")


def cluster(fingerprints: dict[str, int], threshold: int = HAMMING_THRESHOLD) -> list[set[str]]:
    """Group keys whose fingerprints are within *threshold* Hamming distance.

    O(n²) pairwise — fine at repo-doc scale (hundreds). Zero fingerprints
    (empty docs) never cluster.
    """
    keys = [k for k, fp in fingerprints.items() if fp != 0]
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if hamming(fingerprints[a], fingerprints[b]) <= threshold:
                parent[find(a)] = find(b)

    groups: dict[str, set[str]] = {}
    for k in keys:
        groups.setdefault(find(k), set()).add(k)
    return [group for group in groups.values() if len(group) > 1]
