from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.common_config_classes import RewardConfig
from rlworld.rl.configs.rewards import RewardTermConfig, get_weight_value
from rlworld.rl.envs.managers.base import BaseManager

# Backward-compatible alias (used by ManagerRegistry and imports)
RewardManagerConfig = RewardConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class RewardManager(BaseManager):
    """Manages reward computation from configurable reward terms.

    Terms are discovered via :func:`iter_terms` on the ``RewardConfig`` instance.

    Setup-time selector resolution: any
    :class:`~rlworld.rl.configs.scene.entity_selector.SceneEntitySelector`
    found inside a term's ``params`` dict is replaced **once** with a
    pre-resolved :class:`ResolvedEntity` before the term function is
    ever called.  This mirrors mjlab's ``manager_base._resolve_common_term_cfg``
    pattern and means reward terms pay zero per-step resolution cost
    even when the selector targets specific bodies/joints.
    """

    def __init__(self, env: World, config: RewardConfig):
        super().__init__(env=env)
        self.config = config

        # Discover named terms from config attributes
        self.reward_terms: dict[str, RewardTermConfig] = iter_terms(config, RewardTermConfig)

        # Resolve func (callable or string) → actual callable, cached at init
        self._resolved_fns: dict[str, object] = {}
        self._instances: dict[str, object] = {}
        for name, reward_term in self.reward_terms.items():
            func = reward_term.resolved_func
            self._resolved_fns[name] = func
            # Replace any SceneEntitySelector in params with its resolved
            # ResolvedEntity, before class instantiation / function binding.
            self._resolve_term_selectors(func, reward_term.params)
            # Check if func is a class (stateful reward)
            if isinstance(func, type):
                self._instances[name] = func(env=self.env, **reward_term.params)

    def get_term_cfg(self, name: str) -> RewardTermConfig:
        """Return the live RewardTermConfig for a registered term.

        Used by the curriculum manager to mutate a reward term's
        ``weight`` or ``params`` based on training progress. The
        returned object is the same instance that
        :meth:`_compute_weighted_reward` reads from, so in-place
        modifications take effect on the next reward computation.
        """
        if name not in self.reward_terms:
            raise KeyError(f"Reward term {name!r} not found. Available: {list(self.reward_terms)}")
        return self.reward_terms[name]

    def set_rewards(
        self,
        reward_buffer: torch.Tensor,
        episode_sums: dict[str, torch.Tensor],
        reward_buffer_per_type: dict[str, torch.Tensor],
    ) -> None:
        mode = self.config.reward_mode

        if mode == "sum":
            for name, reward_term in self.reward_terms.items():
                reward_value = self._compute_weighted_reward(name, reward_term)
                reward_buffer_per_type[name] = reward_value
                episode_sums[name] += reward_value
                reward_buffer += reward_value
        elif mode == "exponential":
            self._set_rewards_exponential_fixed(reward_buffer, episode_sums, reward_buffer_per_type)
        elif mode == "exponential_auto":
            self._set_rewards_exponential_auto(reward_buffer, episode_sums, reward_buffer_per_type)
        else:
            raise ValueError(f"Unknown reward_mode: {mode!r}")

        reward_buffer_per_type["total_reward"] = reward_buffer

    def _set_rewards_exponential_fixed(
        self,
        reward_buffer: torch.Tensor,
        episode_sums: dict[str, torch.Tensor],
        reward_buffer_per_type: dict[str, torch.Tensor],
    ) -> None:
        """Exponential shaping with fixed classification via exp_shaping flag.

        total = (sum of exp_shaping=False terms) * exp((sum of exp_shaping=True terms) / sigma)
        """
        rew_task = torch.zeros_like(reward_buffer)
        rew_shaped = torch.zeros_like(reward_buffer)

        for name, reward_term in self.reward_terms.items():
            reward_value = self._compute_weighted_reward(name, reward_term)
            reward_buffer_per_type[name] = reward_value
            episode_sums[name] += reward_value

            if reward_term.exp_shaping:
                rew_shaped += reward_value
            else:
                rew_task += reward_value

        reward_buffer += rew_task * torch.exp(rew_shaped / self.config.shaping_sigma)

    def _set_rewards_exponential_auto(
        self,
        reward_buffer: torch.Tensor,
        episode_sums: dict[str, torch.Tensor],
        reward_buffer_per_type: dict[str, torch.Tensor],
    ) -> None:
        """Exponential shaping with dynamic classification by global sum sign.

        Each step, terms with negative global sum go inside exp().
        """
        rew_pos = torch.zeros_like(reward_buffer)
        rew_neg = torch.zeros_like(reward_buffer)

        for name, reward_term in self.reward_terms.items():
            reward_value = self._compute_weighted_reward(name, reward_term)
            reward_buffer_per_type[name] = reward_value
            episode_sums[name] += reward_value

            if torch.sum(reward_value) >= 0:
                rew_pos += reward_value
            else:
                rew_neg += reward_value

        reward_buffer += rew_pos * torch.exp(rew_neg / self.config.shaping_sigma)

    def _compute_weighted_reward(self, name: str, reward_term: RewardTermConfig) -> torch.Tensor:
        if name in self._instances:
            raw_reward = self._instances[name](self.env)
        else:
            raw_reward = self._resolved_fns[name](self.env, **reward_term.params)

        weight = get_weight_value(reward_term.weight, self.env_step_calls)
        return raw_reward * weight * self.env.control_dt

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset stateful reward terms for specified envs."""
        for instance in self._instances.values():
            if hasattr(instance, "reset"):
                instance.reset(env_ids)

    def advance(self) -> None:
        pass

    def __str__(self) -> str:
        """Pretty print reward manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, format_weight, table_to_string

        if not self.reward_terms:
            return ""

        rows = []
        for name, term in self.reward_terms.items():
            weight_str = format_weight(term.weight)

            params_str = "-"
            if term.params and name not in self._instances:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([name, weight_str, params_str])

        table = create_manager_table(
            title="Reward Terms",
            columns=["Name", "Weight", "Params"],
            rows=rows,
            footer=f"{len(self.reward_terms)} terms",
        )
        return table_to_string(table)
