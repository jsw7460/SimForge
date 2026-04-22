"""Newton body/joint label canonicalization — leaf-name extraction.

Newton's MJCF loader stores labels as XPath hierarchies
(``g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_pitch_joint``);
URDF loader stores ``{entity}/{name}`` (``g1_29dof/left_hip_pitch_joint``).
We canonicalize both down to the bare leaf name
(``left_hip_pitch_joint``) so every downstream consumer — builder-time
site / PD-gain application, model-time ArticulationIndexing, body
cache, DR / reset / contact lookups — sees the same name format that
Newton's ``ArticulationView.link_names`` / ``joint_dof_names``
already expose. This matches IsaacLab's Newton integration, which
feeds bare names from ArticulationView directly into user-facing
config.

Prefix-based multi-robot namespacing is replaced by per-entity
``ArticulationView`` instances that filter by ``body_label_prefix``
at view-construction time, so joint / body name collisions between
robots are resolved by which view you query, not by mangling the
names themselves.
"""
from __future__ import annotations


def leaf_name(label: str) -> str:
    """Return the leaf segment of a slash-delimited Newton label.

    Mirrors Newton ``ArticulationView.get_name_from_label`` (selection.py:374).

    Examples:
        ``g1_29dof/worldbody/pelvis/.../left_hip_pitch_joint`` → ``left_hip_pitch_joint``
        ``g1_29dof/left_hip_pitch_joint`` → ``left_hip_pitch_joint``
        ``floating_base`` → ``floating_base``
    """
    return label.rsplit("/", maxsplit=1)[-1]
