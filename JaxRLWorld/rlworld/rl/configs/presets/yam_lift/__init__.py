"""Lift a cube off the table and bring it to a commanded point.

mjlab's ``lift_cube`` task, ported to run on all three simulators. Same
robot (the I2RT YAM), same 40 mm 50 g cube, same reward shape — the
reaching kernel multiplying one plus the bringing kernel, so the lift
bonus is unreachable until the arm is at the object.

One thing differs on purpose: mjlab picks the cube off the FLOOR, and
here the arm and the cube share a table. The MDP terms are unchanged;
only the sampling ranges move up by the table's height, and those were
measured rather than shifted arithmetically — see
``scripts/diag/yam_reach_envelope.py``, which put the holdable grasp
point at 0.41 to 1.18 m over the cube and confirmed every corner of the
goal box has a reachable pose within 2 cm.
"""

__all__ = ["base"]
