"""Domain randomization terms for JaxRLWorld.

Cross-sim DR terms live in :mod:`.unified` and dispatch on
``env.sim_type`` — preset configs should target those.

Newton keeps a small set of *non-randomised* fixed-value setters
(``set_joint_friction`` / ``set_foot_friction``) in :mod:`.newton`;
these write a fixed configured value (optionally with a narrow DR band)
and have no cross-sim counterpart.

Shared utilities (``DefaultCache``, ``sample``, etc.) are in ``_utils``.

Usage in preset configs::

    from rlworld.rl.envs.mdp.events.dr import unified as unified_dr

    randomize_friction = EventTermConfig(
        func=unified_dr.randomize_friction,
        mode="reset_dr",
        params={
            "asset_cfg": SceneEntitySelector(name="robot", body_names=(...)),
            "friction_range": (0.8, 1.2),
            "operation": "scale",
        },
    )
"""

from . import unified
from ._utils import DefaultCache, apply_operation, resolve_patterns, sample

# NOTE: the ``newton`` submodule (``set_joint_friction`` / ``set_foot_friction``)
# is *not* re-exported here -- it does ``import warp`` / ``from newton...`` at
# module load, so importing it eagerly would drag the Newton stack into
# Genesis-/MuJoCo-only processes.  Call sites that need it import the submodule
# directly: ``from rlworld.rl.envs.mdp.events.dr import newton as newton_dr``.

__all__ = [
    "unified",
    "DefaultCache",
    "apply_operation",
    "resolve_patterns",
    "sample",
]
