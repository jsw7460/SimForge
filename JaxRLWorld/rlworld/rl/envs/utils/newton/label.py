"""Newton body/joint label canonicalization.

Newton's MJCF loader stores labels as full XPath hierarchies
(``g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_pitch_joint``)
while its URDF loader stores flat labels
(``g1_29dof/left_hip_pitch_joint``). Downstream code (regex matching
in :mod:`rlworld.rl.utils.string`, user-side joint / body pattern
configs, DR body_patterns, site lookups, motion command body
resolution) is dramatically simpler when both paths produce the same
shape, so scene-manager-level canonicalization pays back many times
over. Same insight IsaacLab's Newton integration relies on — their
``ArticulationView.joint_dof_names`` / ``link_names`` only ever
present flat names to user code.

We canonicalize to ``"{entity_prefix}/{leaf_segment}"``: keep the
first slash-delimited segment (the entity prefix such as
``g1_29dof/``) and the last (the joint or body leaf name), dropping
anything in between. This lets user-facing regex patterns stay as
simple as they were under the URDF layout — ``left_.*`` instead of
``(?:.*/)?left_(?!...).*$`` — and keeps the URDF flat path working
unchanged (flattening a label that is already flat is a no-op).
"""
from __future__ import annotations


def flatten_xpath_label(label: str) -> str:
    """Collapse Newton MJCF XPath labels to ``{prefix}/{leaf}``.

    Single-segment labels (e.g. the Newton-internal ``floating_base``)
    are returned unchanged. Labels with two or more segments keep only
    the first and the last, e.g.

        ``g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_pitch_joint``
        → ``g1_29dof/left_hip_pitch_joint``
        ``g1_29dof/left_hip_pitch_joint`` → unchanged (already flat)
    """
    parts = label.split("/")
    if len(parts) <= 1:
        return label
    return f"{parts[0]}/{parts[-1]}"
