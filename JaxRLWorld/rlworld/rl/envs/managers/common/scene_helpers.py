"""Sim-agnostic helpers shared by Newton/Genesis/MuJoCo scene managers.

These helpers extract small pieces of cross-sim duplication into a
single place. They are intentionally **standalone functions**, not a
base class — each scene manager keeps full control of its own lifecycle
and only delegates the unifiable bits.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Callable, Iterable, Literal, Tuple, Union

from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

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
#   ("mjcf_xml", "<mujoco>...</mujoco>")
#       → write to a temp file, then KinematicTree(mjcf_path=tmp)
#         (used by mjlab where the XML lives in entity.spec.to_xml()
#         instead of on disk)
KinematicSourceSpec = Tuple[Literal["urdf", "mjcf_path", "mjcf_xml"], Union[str, Path]]


def build_kinematic_trees(
    entity_names: Iterable[str],
    spec_resolver: Callable[[str], KinematicSourceSpec | None],
    *,
    warn_on_failure: bool = True,
) -> dict[str, KinematicTree]:
    """Build a ``KinematicTree`` per entity by querying ``spec_resolver``.

    Each scene manager passes a small lambda that converts its own
    entity dict / config layout into the unified
    ``KinematicSourceSpec`` shape, and this helper handles the rest:
    file I/O for inline MJCF XML, format dispatch, and (optionally)
    swallowing per-entity build failures with a warning so a single
    bad entity does not abort the whole scene.

    Args:
        entity_names: Iterable of entity names to consider. The scene
            manager controls iteration order.
        spec_resolver: Callable mapping ``entity_name`` to either a
            ``KinematicSourceSpec`` tuple or ``None`` (skip — e.g. for
            ground-plane / non-articulated entities).
        warn_on_failure: When ``True`` (default), KinematicTree build
            errors are logged via ``warnings.warn`` and the entity is
            skipped. Set to ``False`` to re-raise — useful in tests.

    Returns:
        ``dict[str, KinematicTree]`` keyed by entity name. Entities for
        which ``spec_resolver`` returned ``None`` (or that failed in
        warn-mode) are absent from the dict.
    """
    trees: dict[str, KinematicTree] = {}

    for name in entity_names:
        spec = spec_resolver(name)
        if spec is None:
            continue

        try:
            tree = _build_one(spec)
        except Exception as e:
            if not warn_on_failure:
                raise
            warnings.warn(f"Could not build kinematic tree for entity {name!r}: {e}")
            continue

        trees[name] = tree

    return trees


def _build_one(spec: KinematicSourceSpec) -> KinematicTree:
    kind, payload = spec
    if kind == "urdf":
        return KinematicTree(urdf_path=str(payload))
    if kind == "mjcf_path":
        return KinematicTree(mjcf_path=str(payload))
    if kind == "mjcf_xml":
        # mjlab path: ``entity.spec.to_xml()`` returns the XML as a
        # string, but ``KinematicTree`` only knows how to parse files.
        # Write to a temp file, build, then clean up.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp_file:
            tmp_file.write(payload)
            tmp_path = Path(tmp_file.name)
        try:
            return KinematicTree(mjcf_path=str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    raise ValueError(f"Unknown KinematicSourceSpec kind: {kind!r}. Expected one of 'urdf', 'mjcf_path', 'mjcf_xml'.")
