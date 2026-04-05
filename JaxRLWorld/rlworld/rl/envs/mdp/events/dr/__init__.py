"""Domain randomization terms for JaxRLWorld.

Simulator-specific DR functions live in sub-modules:

- :mod:`.newton` — Newton simulator DR terms.
- :mod:`.genesis` — Genesis simulator DR terms.

Shared utilities (``DefaultCache``, ``sample``, etc.) are in ``_utils``.

Usage in preset configs::

    from rlworld.rl.envs.mdp.events.dr import newton as newton_dr

    randomize_friction = EventTermConfig(
        func=newton_dr.randomize_friction,
        mode="reset",
        params={"friction_range": (0.8, 1.2), "operation": "scale"},
    )
"""

from . import genesis, newton
from ._utils import DefaultCache, apply_operation, resolve_patterns, sample

__all__ = [
    "genesis",
    "newton",
    "DefaultCache",
    "apply_operation",
    "resolve_patterns",
    "sample",
]
