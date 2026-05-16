"""Sim-agnostic canonical joint ordering.

The action manager's joint order *must* be the same across simulators so a
policy trained on one backend transfers to the others. Each sim's internal
body/joint enumeration is different (Genesis BFS by tree depth, Newton/mjlab
DFS-but-flavoured), so we cannot trust whatever order the parser happened to
land on.

Instead, every sim's scene manager constructs its canonical joint list by
**walking the in-memory kinematic body tree depth-first from the root**,
**sorting siblings alphabetically by bare body name** at each node, and
collecting each body's inbound joint name in visit order. The result is a
function of the (parent → children) relations and joint/body names only —
identical for the same robot regardless of file format (MJCF / URDF / USD)
and regardless of how the importer happened to flatten the tree internally.

This module hosts the small bits that don't need sim-specific APIs:

* :func:`filter_canonical_to_actuated` — apply the user's
  ``actuated_dof_patterns`` regex list against an already-canonical joint
  name list, returning the matched names IN CANONICAL ORDER (not query
  order) together with their canonical positions.

Per-sim canonical walkers live in each ``managers/{genesis,newton,mujoco}/
scene.py`` module since they need that sim's body-tree access API.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def filter_canonical_to_actuated(
    canonical_names: Sequence[str], actuated_dof_names: Sequence[str]
) -> tuple[list[str], list[int]]:
    """Filter ``canonical_names`` by ``actuated_dof_names`` regexes.

    Each canonical name is included iff at least one pattern :func:`re.fullmatch`
    matches it (first matching pattern wins; each name is included at most
    once). The output preserves CANONICAL order, *not* query order, so the
    resulting list is the same across simulators when ``canonical_names`` is.

    Args:
        canonical_names: Joint names in canonical (kinematic-DFS) order.
        actuated_dof_names: Sequence of regex patterns (or literal names).

    Returns:
        ``(matched_names, matched_canonical_indices)`` — both lists have the
        same length; ``matched_canonical_indices[k]`` is the index of
        ``matched_names[k]`` in ``canonical_names``.
    """
    compiled = [re.compile(p) for p in actuated_dof_names]
    matched_names: list[str] = []
    matched_canonical_indices: list[int] = []
    for ci, name in enumerate(canonical_names):
        for pat in compiled:
            if pat.fullmatch(name):
                matched_names.append(name)
                matched_canonical_indices.append(ci)
                break
    return matched_names, matched_canonical_indices
