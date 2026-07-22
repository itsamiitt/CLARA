---
description: Explore the CLARA knowledge graph around an entity
argument-hint: [entity]
---

Explore the knowledge graph for: $ARGUMENTS

If an entity is given: call `graph_entity` for its card (type, aliases,
possible_duplicates, linked world model), then `graph_neighbors` (depth 1;
depth 2 if the first hop is sparse) and present the relations as a compact
text neighborhood. If the card lists possible_duplicates, surface them and
ask whether to merge — never merge silently. Offer `graph_path` when the
user asks how two things are connected.

If no entity is given: call `memory_stats` and report the graph node/edge
counts, then suggest a few well-connected entities to explore.
