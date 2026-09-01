from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import RewardConfig
from jaxrlworld.rl.configs.rewards import RewardTermConfig, get_weight_value
from jaxrlworld.rl.envs.managers.base import BaseManager

# Backward-compatible alias (used by ManagerRegistry and imports)
RewardManagerConfig = RewardConfig

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World


class RewardManager(BaseManager):
    """Manages reward computation from configurable reward terms.

    Terms are discovered via :func:`iter_terms` on the ``RewardConfig`` instance.

    Setup-time selector resolution: any
    :class:`~jaxrlworld.rl.configs.scene.entity_selector.SceneEntitySelector`
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

        # Lazily-built caches for set_rewards (see the methods below).
        self._exp_shaped_mask_cached: torch.Tensor | None = None
        self._zero_reward_cached: torch.Tensor | None = None

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
        reward_buffer_per_type: dict[str, torch.Tensor],
    ) -> None:
        """Compute every term once and combine them with batched reductions.

        The per-term bookkeeping used to launch 2-4 small kernels per term
        per step (accumulator adds, masked ``torch.where`` pairs in the
        exponential_auto mode); stacking the weighted terms into one
        ``(n_terms, num_envs)`` tensor turns all of it into a handful of
        batched ops. The reduction order over terms therefore changes
        from sequential adds to ``sum(dim=0)`` — same math, float rounding
        differs in the last bits. Per-term values themselves
        (``_compute_weighted_reward``) are computed exactly as before.
        """
        mode = self.config.reward_mode

        if not self.reward_terms:
            if self.config.total_clip is not None:
                reward_buffer.clamp_(*self.config.total_clip)
            reward_buffer_per_type["total_reward"] = reward_buffer
            return

        values = []
        for name, reward_term in self.reward_terms.items():
            reward_value = self._compute_weighted_reward(name, reward_term)
            reward_buffer_per_type[name] = reward_value
            values.append(reward_value)
        stacked = torch.stack(values, dim=0)

        if mode == "sum":
            reward_buffer += stacked.sum(dim=0)
        elif mode == "exponential":
            # total = (sum of exp_shaping=False terms) * exp((sum of exp_shaping=True terms) / sigma)
            shaped_mask = self._exp_shaped_mask(stacked)
            rew_shaped = (stacked * shaped_mask).sum(dim=0)
            rew_task = (stacked * (1.0 - shaped_mask)).sum(dim=0)
            reward_buffer += rew_task * torch.exp(rew_shaped / self.config.shaping_sigma)
        elif mode == "exponential_auto":
            # Terms whose global sum is negative go inside exp(). The sign
            # classification stays a DEVICE decision — a Python ``if`` on
            # the sums would drain the CUDA queue every step.
            is_pos = (stacked.sum(dim=1, keepdim=True) >= 0).to(stacked.dtype)
            rew_pos = (stacked * is_pos).sum(dim=0)
            rew_neg = (stacked * (1.0 - is_pos)).sum(dim=0)
            reward_buffer += rew_pos * torch.exp(rew_neg / self.config.shaping_sigma)
        else:
            raise ValueError(f"Unknown reward_mode: {mode!r}")

        if self.config.total_clip is not None:
            reward_buffer.clamp_(*self.config.total_clip)

        reward_buffer_per_type["total_reward"] = reward_buffer

    def _exp_shaped_mask(self, stacked: torch.Tensor) -> torch.Tensor:
        """Static ``(n_terms, 1)`` float mask of ``exp_shaping`` flags."""
        if self._exp_shaped_mask_cached is None:
            flags = [float(term.exp_shaping) for term in self.reward_terms.values()]
            self._exp_shaped_mask_cached = torch.tensor(flags, device=stacked.device, dtype=stacked.dtype).unsqueeze(1)
        return self._exp_shaped_mask_cached

    def _compute_weighted_reward(self, name: str, reward_term: RewardTermConfig) -> torch.Tensor:
        weight = get_weight_value(reward_term.weight, self.env_step_calls)
        # A statically-zero pure-function term contributes nothing and has
        # no state to advance — skip its kernels entirely. Stateful terms
        # (``_instances``) always run so their internal state stays live.
        if weight == 0.0 and name not in self._instances:
            return self._zero_reward()

        if name in self._instances:
            raw_reward = self._instances[name](self.env)
        else:
            raw_reward = self._resolved_fns[name](self.env, **reward_term.params)

        return raw_reward * weight * self.env.control_dt

    def _zero_reward(self) -> torch.Tensor:
        """Shared all-zeros ``(num_envs,)`` reward — treat as read-only."""
        if self._zero_reward_cached is None:
            self._zero_reward_cached = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.float32)
        return self._zero_reward_cached

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset stateful reward terms for specified envs."""
        for instance in self._instances.values():
            if hasattr(instance, "reset"):
                instance.reset(env_ids)

    def advance(self) -> None:
        pass

    def __str__(self) -> str:
        """Pretty print reward manager configuration."""
        from jaxrlworld.rl.utils.pretty import create_manager_table, format_weight, table_to_string

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
