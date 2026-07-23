"""
CLARA — knowledge graph layer (projection of the memory store).

Beliefs become edges, world-model rows become nodes. The graph is entirely
derived: every table is rebuildable from active memories (`clara graph
rebuild`), rows are invalidated rather than deleted, and projection failures
never fail a memory write (fail-soft by contract).
"""

from clara.graph.normalize import normalize_name, normalize_relation
from clara.graph.resolve import resolve_node

__all__ = ["normalize_name", "normalize_relation", "resolve_node"]
