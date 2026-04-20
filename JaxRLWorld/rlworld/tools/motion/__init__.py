"""Motion-tracking preprocessing tools.

Uses only the ``mujoco`` Python package for forward-kinematics replay —
no mjlab, Genesis, or Newton imports. The produced NPZ is sim-agnostic
and can be consumed by JaxRLWorld's MotionCommand on any of the three
supported simulators.
"""
