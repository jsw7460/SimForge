"""Newton Observation Manager.

This module re-exports the common ObservationManager for Newton environments.
The observation functions themselves are Newton-specific and located in
rlworld/rl/envs/mdp/observations/newton/ (proprioception.py, state.py).

For backward compatibility with existing Newton code that uses NewtonObservationManager,
we also provide NewtonObservationManager and NewtonObsManagerConfig aliases.

Usage:
    from rlworld.rl.envs.managers.newton import (
        NewtonObservationManager,
        NewtonObsManagerConfig,
    )
    from rlworld.rl.envs.mdp.observations.newton import (
        base_lin_vel, base_ang_vel, projected_gravity,
        dof_pos, dof_vel,
    )
    from rlworld.rl.configs.observations import ObservationTermConfig

    config = NewtonObsManagerConfig(
        num_envs=4096,
        obs_group={
            "actor": [
                ObservationTermConfig(func=base_lin_vel, scale=1.0),
                ObservationTermConfig(func=base_ang_vel, scale=1.0),
                ObservationTermConfig(func=projected_gravity, scale=1.0),
                ObservationTermConfig(func=dof_pos, scale=1.0),
                ObservationTermConfig(func=dof_vel, scale=0.05),
            ],
            "critic": [
                ObservationTermConfig(func=base_lin_vel, scale=1.0),
                ObservationTermConfig(func=base_ang_vel, scale=1.0),
                ObservationTermConfig(func=projected_gravity, scale=1.0),
                ObservationTermConfig(func=dof_pos, scale=1.0),
                ObservationTermConfig(func=dof_vel, scale=0.05),
            ],
        },
    )
"""
from rlworld.rl.envs.managers.common.observation_jax import (
    JaxObservationManager,
    ObsManagerConfig,
)

# Aliases for Newton-specific naming convention
NewtonObservationManager = JaxObservationManager
NewtonObsManagerConfig = ObsManagerConfig

__all__ = [
    "JaxObservationManager",
    "ObsManagerConfig",
    "NewtonObservationManager",
    "NewtonObsManagerConfig",
]