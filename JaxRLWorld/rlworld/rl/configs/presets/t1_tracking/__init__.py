"""T1 motion tracking preset.

Entry point: :class:`T1TrackingConfig` — same pattern as
``t1_getup``. Dispatches to ``_{sim}_builders`` modules at build time.
"""

from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig

__all__ = ["T1TrackingConfig"]
