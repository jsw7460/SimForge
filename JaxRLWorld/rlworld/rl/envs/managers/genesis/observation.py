"""Genesis Observation Manager.

This module re-exports the common ObservationManager for Genesis environments.
The observation functions themselves are Genesis-specific and located in
rlworld/rl/envs/mdp/observations/ (proprioception.py, state.py, etc.).

For backward compatibility, this module exports:
- ObservationManager: The common simulator-agnostic observation manager
- ObsManagerConfig: The common config dataclass

Usage:
    from rlworld.rl.envs.managers.genesis import ObservationManager, ObsManagerConfig
    from rlworld.rl.envs.mdp.observations.proprioception import dof_pos, dof_vel
    from rlworld.rl.envs.mdp.observations.state import base_lin_vel

    config = ObsManagerConfig(
        num_envs=4096,
        obs_group={
            "actor": [
                ObservationTermConfig(func=base_lin_vel, scale=1.0),
                ObservationTermConfig(func=dof_pos, scale=1.0),
                ObservationTermConfig(func=dof_vel, scale=0.05),
            ],
        },
    )
"""

from rlworld.rl.envs.managers.common.observation import (
    ObservationManager,
    ObsManagerConfig,
)

__all__ = ["ObservationManager", "ObsManagerConfig"]
