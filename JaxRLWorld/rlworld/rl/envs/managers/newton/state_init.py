from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import warp as wp

import newton
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import StateInitializationTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class NewtonStateInitConfig:
    """Configuration for Newton state initialization.

    Similar to Genesis, this config accepts a list of initialization term functions
    that are called during reset.

    Example:
        from rlworld.rl.envs.mdp.reset.newton_reset_terms import (
            initialize_dof_pos,
            initialize_base_pose,
        )

        config = NewtonStateInitConfig(
            initialization_terms=[
                StateInitializationTermConfig(
                    func=initialize_base_pose,
                    params={"height": 0.42}
                ),
                StateInitializationTermConfig(
                    func=initialize_dof_pos,
                    params={"noise_range": (-0.1, 0.1)}
                ),
            ],
            base_init_pos=[0.0, 0.0, 0.42],
            base_init_quat=[0.0, 0.0, 0.0, 1.0],
        )
    """
    # Initialization term functions (called in order during reset)
    initialization_terms: list[StateInitializationTermConfig] = field(default_factory=list)

    # Default base position and orientation
    base_init_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.42])
    base_init_quat: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw


class NewtonStateInitManager(BaseManager):
    """Manages state initialization for Newton environments.

    This manager handles resetting robot states during training. It supports:
    1. Function-based initialization terms (same pattern as Genesis)
    2. Default base position/orientation
    3. Per-environment reset

    The initialization terms are called in order during reset, allowing for
    flexible composition of initialization behaviors (e.g., set base pose,
    then randomize joint positions, then add noise to velocities).
    """

    def __init__(self, env: "World", config: NewtonStateInitConfig):
        super().__init__(env)
        self.config = config

        # Initialize default positions
        self.base_init_pos = torch.tensor(
            self.config.base_init_pos,
            device=self.device
        ).unsqueeze(0).expand(self.env.num_envs, -1).clone()

        self.base_init_quat = torch.tensor(
            self.config.base_init_quat,
            device=self.device
        ).unsqueeze(0).expand(self.env.num_envs, -1).clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset specified environments to initial state.

        If initialization_terms are provided, they are called in order.
        Otherwise, falls back to default reset behavior.

        Args:
            env_ids: Tensor of environment indices to reset. If None or empty, returns.
        """
        if env_ids is None or len(env_ids) == 0:
            return

        # If initialization terms are defined, use them (like Genesis)
        if self.config.initialization_terms:
            for init_term in self.config.initialization_terms:
                init_term.func(self.env, env_ids, **init_term.params)
        else:
            # Default behavior: reset to initial state
            self._default_reset(env_ids)

    def _default_reset(self, env_ids: torch.Tensor) -> None:
        """Default reset behavior when no initialization terms are provided.

        Resets joint positions and velocities to the model's initial values
        and re-evaluates forward kinematics.
        """
        scene_manager = self.env.scene_manager
        model = scene_manager.model
        state = scene_manager.state_0

        num_worlds = model.num_worlds

        # Get current state as torch tensors
        joint_q_flat = wp.to_torch(state.joint_q)
        joint_qd_flat = wp.to_torch(state.joint_qd)

        # Calculate coords/dofs per world from actual tensor sizes
        coords_per_world = joint_q_flat.numel() // num_worlds
        dofs_per_world = joint_qd_flat.numel() // num_worlds

        joint_q = joint_q_flat.reshape(num_worlds, coords_per_world)
        joint_qd = joint_qd_flat.reshape(num_worlds, dofs_per_world)

        # Get model defaults as torch tensors
        default_q = wp.to_torch(model.joint_q).reshape(num_worlds, coords_per_world)
        default_qd = wp.to_torch(model.joint_qd).reshape(num_worlds, dofs_per_world)

        # Reset only specified environments to default values
        joint_q[env_ids] = default_q[env_ids]
        joint_qd[env_ids] = default_qd[env_ids]

        # Override base position/orientation with config values
        joint_q[env_ids, 0:3] = self.base_init_pos[env_ids]
        joint_q[env_ids, 3:7] = self.base_init_quat[env_ids]

        # Copy back to warp arrays
        wp.copy(state.joint_q, wp.from_torch(joint_q.flatten(), dtype=wp.float32))
        wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))

        # Re-evaluate FK for only the reset environments
        indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices)