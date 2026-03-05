from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import newton
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import StateInitializationTermConfig
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax, wp_from_jax

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class NewtonStateInitConfig:
    """Configuration for Newton state initialization."""
    initialization_terms: list[StateInitializationTermConfig] = field(default_factory=list)
    base_init_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.42])
    base_init_quat: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw


class NewtonStateInitManager(BaseManager):
    """Manages state initialization for Newton environments (JAX-native)."""

    def __init__(self, env: "World", config: NewtonStateInitConfig):
        super().__init__(env)
        self.config = config

        self.base_init_pos = jnp.broadcast_to(
            jnp.array(self.config.base_init_pos),
            (self.env.num_envs, 3),
        )

        self.base_init_quat = jnp.broadcast_to(
            jnp.array(self.config.base_init_quat),
            (self.env.num_envs, 4),
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None or len(env_ids) == 0:
            return

        if self.config.initialization_terms:
            for init_term in self.config.initialization_terms:
                init_term.func(self.env, env_ids, **init_term.params)
        else:
            self._default_reset(env_ids)

    def _default_reset(self, env_ids) -> None:
        scene_manager = self.env.scene_manager
        model = scene_manager.model
        state = scene_manager.state_0

        num_worlds = model.num_worlds

        joint_q_flat = wp_to_jax(state.joint_q)
        joint_qd_flat = wp_to_jax(state.joint_qd)

        coords_per_world = joint_q_flat.size // num_worlds
        dofs_per_world = joint_qd_flat.size // num_worlds

        joint_q = joint_q_flat.reshape(num_worlds, coords_per_world)
        joint_qd = joint_qd_flat.reshape(num_worlds, dofs_per_world)

        default_q = wp_to_jax(model.joint_q).reshape(num_worlds, coords_per_world)
        default_qd = wp_to_jax(model.joint_qd).reshape(num_worlds, dofs_per_world)

        joint_q = joint_q.at[env_ids].set(default_q[env_ids])
        joint_qd = joint_qd.at[env_ids].set(default_qd[env_ids])

        joint_q = joint_q.at[env_ids, 0:3].set(self.base_init_pos[env_ids])
        joint_q = joint_q.at[env_ids, 3:7].set(self.base_init_quat[env_ids])

        wp.copy(state.joint_q, wp_from_jax(joint_q.flatten(), dtype=wp.float32))
        wp.copy(state.joint_qd, wp_from_jax(joint_qd.flatten(), dtype=wp.float32))

        # Convert env_ids to int32 numpy for warp
        indices_np = np.array(env_ids, dtype=np.int32)
        indices_wp = wp.array(indices_np, dtype=wp.int32)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices_wp)
