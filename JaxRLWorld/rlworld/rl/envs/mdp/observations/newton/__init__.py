"""Newton-specific observation functions.

These functions are designed to work with Newton environments and extract
observations from Newton's state representation (warp arrays).

Usage:
    from rlworld.rl.envs.mdp.observations.newton import (
        base_lin_vel, base_ang_vel, projected_gravity,
        dof_pos, dof_vel,
    )
"""
from .state import (
    base_pos,
    base_quat,
    base_height,
    base_lin_vel,
    base_ang_vel,
)
from .proprioception import (
    projected_gravity,
    dof_pos,
    dof_vel,
    raw_actions,
    prev_processed_actions,
    dof_pos_nominal_difference
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
    "dof_pos_nominal_difference"
]