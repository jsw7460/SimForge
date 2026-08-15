"""YAM arm — the minimal fixed-base preset.

A bench-mounted 6-DOF arm on a ground plane, with nothing to do: joint
observations, a joints-only reset, a time-out termination and no reward.
It exists to exercise the fixed-base path end to end before any
manipulation task is layered on it, so that when the task misbehaves the
arm itself is already known good.
"""

__all__ = ["base"]
