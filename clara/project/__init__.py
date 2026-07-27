"""
CLARA project memory — what this codebase *is*, learned from its manifests.

The point is that nobody has to teach CLARA the obvious: opening a repo should
be enough for it to know the language, package manager, frameworks, build and
test tooling, and whether it is a monorepo.

Detection is deliberately evidence-based rather than clever. Every fact records
the file (and key) that proved it, and anything a manifest does not actually
state is simply not claimed — a memory system that guesses its way to a
plausible-but-wrong project profile is worse than one that stays quiet.
"""

from clara.project.detect import (
    ProjectFact,
    ProjectProfile,
    detect_project,
)

__all__ = ["ProjectFact", "ProjectProfile", "detect_project"]
