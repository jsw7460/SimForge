"""Sim-agnostic helpers shared by Newton/Genesis/MuJoCo scene managers.

These helpers extract small pieces of cross-sim duplication into a
single place. They are intentionally **standalone functions**, not a
base class — each scene manager keeps full control of its own lifecycle
and only delegates the unifiable bits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Literal, Tuple, Union

import mujoco

from jaxrlworld.rl.configs.robots.kinematic_tree import KinematicTree

# Spec returned by a scene manager's per-entity resolver. The first
# element selects the source format; the second is the payload that
# format expects:
#
#   ("urdf", "/abs/path/to/robot.urdf")
#       → KinematicTree(urdf_path=...)
#
#   ("mjcf_path", "/abs/path/to/robot.xml")
#       → KinematicTree(mjcf_path=...)
#
#   ("mjcf_spec", <mujoco.MjSpec>)
#       → KinematicTree(mjcf_spec=...)
#         (used by mjlab where the model lives in a standalone MjSpec
#         whose asset paths are relative to the file it was loaded from,
#         so its XML cannot be compiled from anywhere else)
KinematicSourceSpec = Tuple[Literal["urdf", "mjcf_path", "mjcf_spec"], Union[str, Path, mujoco.MjSpec]]


def build_kinematic_trees(
    entity_names: Iterable[str],
    spec_resolver: Callable[[str], KinematicSourceSpec | None],
) -> dict[str, KinematicTree]:
    """Build a ``KinematicTree`` per entity by querying ``spec_resolver``.

    Each scene manager passes a small lambda that converts its own
    entity dict / config layout into the unified
    ``KinematicSourceSpec`` shape, and this helper handles the rest:
    format dispatch. A tree that fails
    to build raises.

    Args:
        entity_names: Iterable of entity names to consider. The scene
            manager controls iteration order.
        spec_resolver: Callable mapping ``entity_name`` to either a
            ``KinematicSourceSpec`` tuple or ``None`` (skip — e.g. for
            ground-plane / non-articulated entities).

    Returns:
        ``dict[str, KinematicTree]`` keyed by entity name. Entities for
        which ``spec_resolver`` returned ``None`` are absent from the dict.
    """
    trees: dict[str, KinematicTree] = {}

    for name in entity_names:
        spec = spec_resolver(name)
        if spec is None:
            continue
        trees[name] = _build_one(spec)

    return trees


def _build_one(spec: KinematicSourceSpec) -> KinematicTree:
    kind, payload = spec
    if kind == "urdf":
        return KinematicTree(urdf_path=str(payload))
    if kind == "mjcf_path":
        return KinematicTree(mjcf_path=str(payload))
    if kind == "mjcf_spec":
        return KinematicTree(mjcf_spec=payload)
    raise ValueError(f"Unknown KinematicSourceSpec kind: {kind!r}. Expected one of 'urdf', 'mjcf_path', 'mjcf_spec'.")
