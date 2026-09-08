from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import RewardConfig
from jaxrlworld.rl.configs.rewards import RewardTermConfig, get_weight_value
from jaxrlworld.rl.envs.managers.base import BaseManager
from jaxrlworld.rl.envs.managers.common.reward_view import RecordingEnvView, RewardEnvView, RewardReadRecord

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
        # Previous-step foot positions of the finite-difference slip
        # terms (``rewards.common.reward_terms._fd_foot_velocity``),
        # keyed by term key; plain tensor state the terms update in place.
        self._fd_prev_foot_pos: dict[str, torch.Tensor] = {}
        # The compiled chain (``config.compile_terms``). Built after the
        # first call, which runs eagerly through the recorders of
        # ``reward_view`` to learn what the terms read.
        self._compile_terms = config.compile_terms
        self._view: RewardEnvView | None = None
        self._compiled_chain = None
        self._weights_cache: tuple[tuple[float, ...], torch.Tensor] | None = None

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
        if not self.reward_terms:
            if self.config.total_clip is not None:
                reward_buffer.clamp_(*self.config.total_clip)
            reward_buffer_per_type["total_reward"] = reward_buffer
            return

        if not self._compile_terms:
            stacked = self._compute_stacked(self.env)
            self._combine(stacked, reward_buffer)
        elif self._view is None:
            # First call: eager, through the recorders, so the snapshot
            # knows what to read from now on. Same values as the eager
            # path; also warms every lazily-built cache the terms keep.
            record = RewardReadRecord()
            stacked = self._compute_stacked(RecordingEnvView(self.env, record))
            self._combine(stacked, reward_buffer)
            self._view = RewardEnvView(self.env, record)
            self._compiled_chain = torch.compile(self._chain, fullgraph=True, dynamic=False)
        else:
            self._view.refresh()
            stacked = self._compiled_chain(self._view, self._weights(), self._active(), reward_buffer)

        for i, name in enumerate(self.reward_terms):
            reward_buffer_per_type[name] = stacked[i]
        reward_buffer_per_type["total_reward"] = reward_buffer

    def _compute_stacked(self, env) -> torch.Tensor:
        """Every weighted term, eagerly, as ``(n_terms, num_envs)``.

        On the real env this goes through :meth:`_compute_weighted_reward`
        — the seam the reward diagnostics patch to intercept term values —
        and on a recording view through :meth:`_weighted_term` directly.
        """
        if env is self.env:
            values = [self._compute_weighted_reward(name, term) for name, term in self.reward_terms.items()]
        else:
            values = [self._weighted_term(name, term, env) for name, term in self.reward_terms.items()]
        return torch.stack(values, dim=0)

    def _chain(self, env, weights: torch.Tensor, active: tuple[bool, ...], reward_buffer: torch.Tensor) -> torch.Tensor:
        """The compiled program: every active term on the snapshot view,
        the weighting, and the mode combination into ``reward_buffer``.

        ``active`` is the per-term "runs this step" flag (a zero-weight
        pure term is skipped, as the eager path skips it); as a tuple of
        Python bools it is part of the compiled program's signature, so a
        weight schedule crossing zero recompiles once. ``weights`` already
        carries ``control_dt``.
        """
        raws = []
        for i, (name, term) in enumerate(self.reward_terms.items()):
            if not active[i]:
                raws.append(self._zero_reward())
            elif name in self._instances:
                raws.append(self._instances[name](env))
            else:
                raws.append(self._resolved_fns[name](env, **term.params))
        stacked = torch.stack(raws, dim=0) * weights[:, None]
        self._combine(stacked, reward_buffer)
        return stacked

    def _weights(self) -> torch.Tensor:
        """``(n_terms,)`` of ``weight * control_dt``, re-uploaded only when a
        weight changes (a schedule or the curriculum manager)."""
        step = self.env_step_calls
        dt = self.env.control_dt
        values = tuple(get_weight_value(term.weight, step) * dt for term in self.reward_terms.values())
        if self._weights_cache is None or self._weights_cache[0] != values:
            self._weights_cache = (values, torch.tensor(values, device=self.device, dtype=torch.float32))
        return self._weights_cache[1]

    def _active(self) -> tuple[bool, ...]:
        step = self.env_step_calls
        return tuple(
            get_weight_value(term.weight, step) != 0.0 or name in self._instances
            for name, term in self.reward_terms.items()
        )

    def _combine(self, stacked: torch.Tensor, reward_buffer: torch.Tensor) -> None:
        """Fold the weighted terms into ``reward_buffer`` per ``reward_mode``."""
        mode = self.config.reward_mode
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

    def _exp_shaped_mask(self, stacked: torch.Tensor) -> torch.Tensor:
        """Static ``(n_terms, 1)`` float mask of ``exp_shaping`` flags."""
        if self._exp_shaped_mask_cached is None:
            flags = [float(term.exp_shaping) for term in self.reward_terms.values()]
            self._exp_shaped_mask_cached = torch.tensor(flags, device=stacked.device, dtype=stacked.dtype).unsqueeze(1)
        return self._exp_shaped_mask_cached

    def _compute_weighted_reward(self, name: str, reward_term: RewardTermConfig) -> torch.Tensor:
        return self._weighted_term(name, reward_term, self.env)

    def _weighted_term(self, name: str, reward_term: RewardTermConfig, env) -> torch.Tensor:
        weight = get_weight_value(reward_term.weight, self.env_step_calls)
        # A statically-zero pure-function term contributes nothing and has
        # no state to advance — skip its kernels entirely. Stateful terms
        # (``_instances``) always run so their internal state stays live.
        if weight == 0.0 and name not in self._instances:
            return self._zero_reward()

        if name in self._instances:
            raw_reward = self._instances[name](env)
        else:
            raw_reward = self._resolved_fns[name](env, **reward_term.params)

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
