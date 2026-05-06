"""Newton-specific observation functions.

These functions are designed to work with Newton environments and extract
observations from Newton's state representation (warp arrays).

Usage:
    from rlworld.rl.envs.mdp.observations.newton import (
        base_lin_vel, base_ang_vel, projected_gravity,
        dof_pos, dof_vel,
    )
"""

from .proprioception import (
    dof_pos,
    dof_pos_nominal_difference,
    dof_vel,
    prev_processed_actions,
    projected_gravity,
    raw_actions,
)
from .state import (
    base_ang_vel,
    base_height,
    base_lin_vel,
    base_pos,
    base_quat,
)

__all__ = [
    # State
    "base_pos",
    "base_quat",
    "base_height",
    "base_lin_vel",
    "base_ang_vel",
    # Proprioception
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "raw_actions",
    "prev_processed_actions",
    "dof_pos_nominal_difference",
]
