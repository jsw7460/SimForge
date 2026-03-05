"""JAX-native reward manager for Newton environments."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.configs.rewards import RewardTermConfig, get_weight_value
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World

from dataclasses import dataclass


@dataclass
class RewardManagerConfig:
    reward_terms: list[RewardTermConfig] = None


class JaxRewardManager(BaseManager):
    """JAX-native reward manager."""

    def __init__(self, env: "World", config: RewardManagerConfig):
        super().__init__(env=env)
        self.config = config
        self.reward_terms = config.reward_terms

        self._instances: dict[int, object] = {}
        if self.reward_terms:
            for idx, reward_term in enumerate(self.reward_terms):
                func = reward_term.func
                if isinstance(func, type):
                    self._instances[idx] = func(env=self.env, **reward_term.params)

    def set_rewards(
        self,
        reward_buffer: jax.Array,
        episode_sums: dict[str, jax.Array],
        reward_buffer_per_type: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute rewards and return updated buffer + episode_sums.

        JAX arrays are immutable, so we return new values instead of in-place update.
        """
        for idx, reward_term in enumerate(self.reward_terms):
            reward_value = self._compute_weighted_reward(idx, reward_term)

            if idx in self._instances:
                reward_name = self._instances[idx].__name__
            else:
                reward_name = reward_term.func.__name__

            reward_buffer_per_type[reward_name] = reward_value
            if reward_name in episode_sums:
                episode_sums[reward_name] = episode_sums[reward_name] + reward_value
            else:
                episode_sums[reward_name] = reward_value
            reward_buffer = reward_buffer + reward_value

        reward_buffer_per_type["total_reward"] = reward_buffer
        return reward_buffer, episode_sums

    def _compute_weighted_reward(self, idx: int, reward_term: RewardTermConfig) -> jax.Array:
        if idx in self._instances:
            raw_reward = self._instances[idx](self.env)
        else:
            raw_reward = reward_term.func(self.env, **reward_term.params)

        weight = get_weight_value(reward_term.weight, self.env_step_calls)
        return raw_reward * weight * self.env.control_dt

    def reset(self, env_ids) -> None:
        for instance in self._instances.values():
            if hasattr(instance, 'reset'):
                instance.reset(env_ids)

    def advance(self) -> None:
        pass

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string, format_weight

        if not self.reward_terms:
            return ""

        rows = []
        for idx, term in enumerate(self.reward_terms):
            if idx in self._instances:
                func_name = self._instances[idx].__name__
            else:
                func_name = getattr(term.func, '__name__', f"term_{idx}")

            weight_str = format_weight(term.weight)

            params_str = "-"
            if term.params and idx not in self._instances:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([idx, func_name, weight_str, params_str])

        table = create_manager_table(
            title="Reward Terms",
            columns=["Idx", "Name", "Weight", "Params"],
            rows=rows,
            footer=f"{len(self.reward_terms)} terms"
        )
        return table_to_string(table)
