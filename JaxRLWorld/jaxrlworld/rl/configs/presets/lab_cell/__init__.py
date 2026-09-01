"""A workcell: a bench-mounted arm, a workpiece, and a quadruped beside it.

The two-robot case with two DIFFERENT robots — a fixed-base 7-DOF arm and
a floating-base 12-DOF quadruped — sharing one scene, one action vector
and one observation. ``yam_dual`` covers two copies of the same arm, which
cannot distinguish "each entity is addressed separately" from "both
entities happen to be identical"; this one can.

Not a training preset: no reward, no command, no locomotion terminations.
It exists to be looked at and measured::

    python -m jaxrlworld.scripts.view_scene --preset lab_cell --sim newton
"""

__all__ = ["base"]
