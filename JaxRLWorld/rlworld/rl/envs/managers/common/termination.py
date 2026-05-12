from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs import TerminationResult, TerminationTermConfig
from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.configs.common_config_classes import TerminationsConfig
    from rlworld.rl.envs import World

# Backward-compatible alias
TerminationConfig = None  # will be cleaned up later


class TerminationManager(BaseManager):
    """Manages termination conditions for the environment.

    Terms are discovered via :func:`iter_terms` on the config instance.
    """

    def __init__(self, env: World, config: TerminationsConfig, episode_length_s: float):
        super().__init__(env=env)
        self.config = config
        self._episode_length_s = episode_length_s

        # Discover named terms
        self._all_terms: dict[str, TerminationTermConfig] = iter_terms(config, TerminationTermConfig)
        self._resolved_fns: dict[str, callable] = {name: term.resolved_func for name, term in self._all_terms.items()}
        # Per-term {param_name: ResolvedEntity} overrides, merged over the
        # (config-owned, untouched) term params at call time.
        self._term_overrides: dict[str, dict] = {
            name: self._selector_overrides(term.resolved_func, term.params) for name, term in self._all_terms.items()
        }

        self.reset_buf = torch.ones(env.num_envs, device=self.device, dtype=torch.bool)
        self.episode_count = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)

        # Per-term fire state. ``_term_dones`` is the last-step mask (public
        # via the ``term_dones`` property so curriculum / reward / observation
        # terms can query whether a specific termination fired without
        # poking private internals). ``_episode_fires`` accumulates per-env
        # fire counts over the current episode and is cleared on reset;
        # :meth:`reset` reads it to emit ``Episode_Termination/<name>`` —
        # the fraction of just-reset envs for which ``<name>`` fired at
        # least once during the ending episode. Initialized eagerly so
        # downstream code can read the dict structure before the first
        # ``check_termination`` call.
        self._term_dones: dict[str, torch.Tensor] = {
            name: torch.zeros(env.num_envs, dtype=torch.bool, device=self.device) for name in self._all_terms
        }
        self._episode_fires: dict[str, torch.Tensor] = {
            name: torch.zeros(env.num_envs, dtype=torch.long, device=self.device) for name in self._all_terms
        }

        # Iteration-window accumulators consumed by
        # :meth:`consume_episode_stats`. ``_iter_reset_count`` counts every
        # env that reset since the last consume; ``_iter_fire_counts`` counts,
        # per term, how many of those resets had the term fire at least once
        # during their ending episode. The ratio is logged as
        # ``Episode_Termination/<name>`` by the runner once per training
        # iteration. Cleared on each consume so each call covers a fresh
        # window.
        self._iter_reset_count: int = 0
        self._iter_fire_counts: dict[str, int] = {name: 0 for name in self._all_terms}

        # Last-step union of non-timeout termination fires; consumed by
        # MotionCommand's adaptive sampling to weight motion bins by
        # episode-failure frequency. Refreshed on every
        # :meth:`check_termination` call.
        self._terminated_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        self.extras = {}

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self._episode_length_s / self.env.control_dt)

    def consume_episode_stats(self) -> dict[str, float]:
        """Snapshot + clear per-term reset-cause ratios over the current window.

        For each registered term, returns the share of envs that reset
        since the last consume call for which the term fired at least
        once during the ending episode. Multiple terms can each
        contribute on the same env, so the ratios are not mutually
        exclusive and generally do not sum to 1.

        The internal counters are cleared on each call, so successive
        calls cover disjoint windows. Returns an empty dict when no
        resets have occurred in the window — callers should treat an
        empty return as "nothing to log" rather than "all zeros".

        Keys follow the convention ``"Episode_Termination/<term_name>"``
        so wandb auto-groups them in a single UI folder.
        """
        if self._iter_reset_count == 0:
            return {}
        n = self._iter_reset_count
        out = {f"Episode_Termination/{name}": count / n for name, count in self._iter_fire_counts.items()}
        self._iter_reset_count = 0
        for name in self._iter_fire_counts:
            self._iter_fire_counts[name] = 0
        return out

    @property
    def term_dones(self) -> dict[str, torch.Tensor]:
        """Last-step per-term fire masks, keyed by term name.

        Each value is a ``(num_envs,)`` bool tensor indicating whether
        that term's :meth:`check_termination` result was True for each
        env during the most recent call. Read-only from consumers'
        perspective; do not mutate — rewrite happens every step.

        Intended consumers: curriculum (e.g. "fall rate > 0.5 → ramp
        down difficulty"), reward shaping (e.g. zero a bonus when a
        soft-failure term fires), diagnostic observations.
        """
        return self._term_dones

    def get_term_cfg(self, name: str) -> TerminationTermConfig:
        """Return the live TerminationTermConfig for a registered term.

        Used by the curriculum manager to mutate a termination term's
        ``params`` dict based on training progress. The returned object
        is the same instance that :meth:`check_termination` reads from,
        so in-place modifications take effect on the next check.
        """
        if name not in self._all_terms:
            raise KeyError(f"Termination term {name!r} not found. Available: {list(self._all_terms)}")
        return self._all_terms[name]

    def advance(self) -> None:
        self.episode_length_buf += 1

    def check_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for name, term_config in self._all_terms.items():
            result: TerminationResult = self._resolved_fns[name](
                self.env, **{**term_config.params, **self._term_overrides[name]}
            )

            self._term_dones[name] = result.reset
            self._episode_fires[name] += result.reset.long()

            if result.is_timeout:
                truncated |= result.reset
            else:
                terminated |= result.reset

            if result.extras:
                self.extras.update(result.extras)

        self._terminated_mask = terminated
        self.reset_buf = terminated | truncated
        return terminated, truncated

    @property
    def terminated(self) -> torch.Tensor:
        """Last-step union of non-timeout termination fires (``(num_envs,) bool``).

        Complement of truncations (timeouts). Refreshed on every
        :meth:`check_termination` call. Consumed by motion-tracking
        adaptive sampling to measure per-bin episode failure rates.
        """
        return self._terminated_mask

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return

        # Fold the just-reset envs into the iteration-window accumulator.
        # Raw counts — not fractions — so
        # :meth:`consume_episode_stats` can compute an unbiased
        # iteration-wide ratio as (total term fires / total resets).
        # Per-env fire tallies are cleared after read so the next
        # episode starts with a fresh counter for this env.
        n_reset = env_ids.numel() if hasattr(env_ids, "numel") else len(env_ids)
        if n_reset > 0:
            self._iter_reset_count += int(n_reset)
            for name, fires in self._episode_fires.items():
                self._iter_fire_counts[name] += int((fires[env_ids] > 0).long().sum().item())
                fires[env_ids] = 0

        self.episode_count[env_ids] += 1
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

    def __str__(self) -> str:
        """Pretty print termination manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._all_terms:
            return ""

        rows = []
        for idx, (name, term) in enumerate(self._all_terms.items()):
            func_name = getattr(self._resolved_fns[name], "__name__", name)

            if "timeout" in func_name.lower() or "time_out" in func_name.lower():
                type_str = "Truncation (timeout)"
            else:
                type_str = "Termination"

            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)

            rows.append([idx, func_name, type_str, params_str])

        table = create_manager_table(
            title="Termination Criteria",
            columns=["Idx", "Name", "Type", "Params"],
            rows=rows,
            footer=f"Max Episode: {self.max_episode_length} steps ({self._episode_length_s}s)",
        )
        return table_to_string(table)
