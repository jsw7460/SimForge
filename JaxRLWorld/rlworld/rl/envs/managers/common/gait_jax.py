"""JAX-native gait manager for Newton environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class GaitManagerConfig:
    num_envs: int
    gait_period: float
    foot_names: tuple[str, ...] | list[str]


class JaxGaitManager(BaseManager):
    """JAX-native gait pattern manager for legged locomotion."""

    def __init__(self, env: "World", config: GaitManagerConfig):
        super().__init__(env)
        self.config = config

        self.foot_names = tuple(self.env.scene_manager.find_body_names(body_names=config.foot_names))

        self.num_feet = len(self.foot_names)
        self.gait_period = config.gait_period
        self.phase_width = 2 * jnp.pi / self.num_feet

        self.phase_offsets = jnp.array(
            [-2 * jnp.pi * i / self.num_feet for i in range(self.num_feet)]
        )

        self.gait_timer = jnp.zeros(config.num_envs)

    def advance(self) -> None:
        self.gait_timer = self.gait_timer + self.env.control_dt

    def get_swing_mask(self) -> jax.Array:
        """Get boolean mask indicating which feet are in swing phase."""
        timer_expanded = jnp.expand_dims(self.gait_timer, 1)
        offsets_expanded = jnp.expand_dims(self.phase_offsets, 0)

        phi = 2 * jnp.pi * (timer_expanded / self.gait_period) + offsets_expanded
        phi = phi % (2 * jnp.pi)

        return (0 <= phi) & (phi < self.phase_width)

    def get_phase_encoding(self) -> jax.Array:
        """Get sin/cos encoding of each foot's phase for observation."""
        timer_expanded = jnp.expand_dims(self.gait_timer, 1)
        offsets_expanded = jnp.expand_dims(self.phase_offsets, 0)

        phi = 2 * jnp.pi * (timer_expanded / self.gait_period) + offsets_expanded

        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)

        encoding = jnp.stack([cos_phi, sin_phi], axis=-1)
        return encoding.reshape(self.env.num_envs, -1)

    def reset(self, env_ids) -> None:
        self.gait_timer = self.gait_timer.at[env_ids].set(0.0)

    def get_swing_progress(self) -> jax.Array:
        """Get normalized progress within swing phase for each foot."""
        timer_expanded = jnp.expand_dims(self.gait_timer, 1)
        offsets_expanded = jnp.expand_dims(self.phase_offsets, 0)

        phi = 2 * jnp.pi * (timer_expanded / self.gait_period) + offsets_expanded
        phi = phi % (2 * jnp.pi)

        is_swing = (0 <= phi) & (phi < self.phase_width)

        progress = phi / self.phase_width

        progress = jnp.where(is_swing, progress, jnp.full_like(progress, -1.0))

        return progress

    def get_target_foot_height(
        self,
        max_height: float,
        profile: str = "sine",
    ) -> jax.Array:
        """Get target foot height based on swing phase progress."""
        progress = self.get_swing_progress()
        is_swing = progress >= 0

        phi = jnp.clip(progress, 0.0, 1.0)

        if profile == "sine":
            height_ratio = jnp.sin(jnp.pi * phi)
        elif profile == "cosine":
            height_ratio = 0.5 * (1 - jnp.cos(2 * jnp.pi * phi))
        else:
            raise ValueError(f"Unknown profile: {profile}")

        target_height = max_height * height_ratio

        target_height = jnp.where(is_swing, target_height, jnp.zeros_like(target_height))

        return target_height