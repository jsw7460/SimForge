"""Two YAM arms on one bench — the multi-robot reference preset.

Extends the single-arm preset with a second arm and splits the action
space across three terms:

* ``left_arm``     — the six arm joints of ``robot``
* ``left_gripper`` — the gripper joint of ``robot``
* ``right_arm``    — every actuated joint of ``robot_right``

which is the smallest configuration that exercises both directions the
action layer has to get right: several terms driving one robot, and
terms driving different robots. Each is a case where a single shared
joint list would put a term's output on the wrong joints.
"""

__all__ = ["base"]
