---
description: Save a fact, preference, or decision to CLARA long-term memory
argument-hint: <fact to remember>
---

Store this in CLARA memory: $ARGUMENTS

Decide the right memory type and call `memory_save` once per atomic fact:
belief (subject/relation/object) for preferences and stable facts, event
(subject/event_type/description) for things that happened, skill (name/
trigger_conditions/steps) for reusable procedures, world_model (entity_type/
name/properties) for current state of a system. If this corrects an earlier
fact, also save a negation belief (`is_negation: true`) for the old one.
Never store secrets or credentials. Confirm in one line what was saved and
its memory_id.
