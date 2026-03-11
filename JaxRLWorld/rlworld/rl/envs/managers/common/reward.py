from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.rewards import RewardTermConfig, get_weight_value
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class RewardManagerConfig:
    """Configuration for reward management."""
    reward_terms: dict[str, RewardTermConfig] = None


class RewardManager(BaseManager):
    """Manages reward computation from configurable reward terms."""

    def __init__(self, env: "World", config: RewardManagerConfig):
        super().__init__(env=env)
        self.config = config
        self.reward_terms = config.reward_terms

        # Initialize stateful reward instances
        self._instances: dict[str, object] = {}
        if self.reward_terms:
            for name, reward_term in self.reward_terms.items():
                func = reward_term.func
                # Check if func is a class (not an instance, not a function)
                if isinstance(func, type):
                    self._instances[name] = func(env=self.env, **reward_term.params)

    def set_rewards(
        self,
        reward_buffer: torch.Tensor,
        episode_sums: dict[str, torch.Tensor],
        reward_buffer_per_type: dict[str, torch.Tensor]
    ) -> None:

        for name, reward_term in self.reward_terms.items():
            reward_value = self._compute_weighted_reward(name, reward_term)

            reward_buffer_per_type[name] = reward_value
            episode_sums[name] += reward_value
            reward_buffer += reward_value

        reward_buffer_per_type["total_reward"] = reward_buffer

    def _compute_weighted_reward(self, name: str, reward_term: RewardTermConfig) -> torch.Tensor:
        # Use instance if available (stateful class), otherwise call function
        if name in self._instances:
            raw_reward = self._instances[name](self.env)
        else:
            raw_reward = reward_term.func(self.env, **reward_term.params)

        weight = get_weight_value(reward_term.weight, self.env_step_calls)
        return raw_reward * weight * self.env.control_dt

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset stateful reward terms for specified envs."""
        for instance in self._instances.values():
            if hasattr(instance, 'reset'):
                instance.reset(env_ids)

    def advance(self) -> None:
        pass

    def __str__(self) -> str:
        """Pretty print reward manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string, format_weight

        if not self.reward_terms:
            return ""

        rows = []
        for name, term in self.reward_terms.items():
            weight_str = format_weight(term.weight)

            # Format params if present
            params_str = "-"
            if term.params and name not in self._instances:
                # Only show params for non-class rewards (class params used in __init__)
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([name, weight_str, params_str])

        table = create_manager_table(
            title="Reward Terms",
            columns=["Name", "Weight", "Params"],
            rows=rows,
            footer=f"{len(self.reward_terms)} terms"
        )
        return table_to_string(table)
