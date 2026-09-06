from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs import TerminationResult, TerminationTermConfig
from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
    from jaxrlworld.rl.envs import World

# Backward-compatible alias
TerminationConfig = None  # will be cleaned up later


class TerminationManager(BaseManager):
    """Manages termination conditions for the environment.

    Terms are discovered via :func:`iter_terms` on the config instance.

    A term's ``func`` may be a plain function, called as
    ``func(env, **params)``, or a class, instantiated once at setup with
    ``func(env=env, **params)`` and called as ``instance(env)`` thereafter.
    The class form is how a condition gets to depend on history rather than
    on the current step alone -- "this has been true for N steps in a row"
    is not readable off a single state. Such a term may define ``reset``,
    which :meth:`reset` calls with the environments that just ended, and
    the same convention the reward manager uses for stateful reward terms.
    """

    def __init__(self, env: World, config: TerminationsConfig, episode_length_s: float):
        super().__init__(env=env)
        self.config = config
        self._episode_length_s = episode_length_s

        # Discover named terms
        self._all_terms: dict[str, TerminationTermConfig] = iter_terms(config, TerminationTermConfig)
        self._resolved_fns: dict[str, callable] = {name: term.resolved_func for name, term in self._all_terms.items()}
        # Replace SceneEntitySelector params with their resolved ResolvedEntity.
        for term in self._all_terms.values():
            self._resolve_term_selectors(term.resolved_func, term.params)

        # Stateful terms. A term whose ``func`` is a class is instantiated
        # once here and called as ``instance(env)`` thereafter, mirroring the
        # reward manager. This is what lets a condition depend on history —
        # "the tool has been gripped for N consecutive steps" cannot be read
        # off a single step's state. Per-env state is cleared in :meth:`reset`.
        self._instances: dict[str, object] = {
            name: fn(env=self.env, **self._all_terms[name].params)
            for name, fn in self._resolved_fns.items()
            if isinstance(fn, type)
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
        # One (n_terms, num_envs) tensor, and the per-name dict holds VIEWS
        # into its rows. Readers and the per-step ``+=`` see the dict they
        # always did; :meth:`reset` works on the whole tensor at once, so a
        # reset step costs one gather, one reduction and one scatter instead
        # of three launches per term. The dict must never be rebound
        # (``self._episode_fires[name] = ...``) or the row stops being one.
        self._episode_fires_all: torch.Tensor = torch.zeros(
            (len(self._all_terms), env.num_envs), dtype=torch.long, device=self.device
        )
        self._episode_fires: dict[str, torch.Tensor] = {
            name: self._episode_fires_all[i] for i, name in enumerate(self._all_terms)
        }

        # Iteration-window accumulators consumed by
        # :meth:`consume_episode_stats`. ``_iter_reset_count`` counts every
        # env that reset since the last consume; ``_iter_fire_counts`` counts,
        # per term, how many of those resets had the term fire at least once
        # during their ending episode. The ratio is logged as
        # ``Episode_Termination/<name>`` by the runner once per training
        # iteration. Cleared on each consume so each call covers a fresh
        # window.
        # Accumulated ON THE DEVICE. Both counters are read once per
        # training iteration by :meth:`consume_episode_stats`, but they are
        # written on every step that anything resets — which, with episodes
        # ending at staggered times, is essentially every step. Keeping
        # them as Python ints meant an ``item()`` per term per step: a
        # device-to-host copy that waits for every kernel queued so far,
        # paid so a ratio could be logged once an iteration.
        self._iter_reset_count: torch.Tensor = torch.zeros((), dtype=torch.long, device=self.device)
        # Same layout as ``_episode_fires``: a (n_terms,) tensor with 0-d
        # views per name, so the reset-step accumulation is one vector add.
        self._iter_fire_counts_all: torch.Tensor = torch.zeros(
            len(self._all_terms), dtype=torch.long, device=self.device
        )
        self._iter_fire_counts: dict[str, torch.Tensor] = {
            name: self._iter_fire_counts_all[i] for i, name in enumerate(self._all_terms)
        }

        # Last-step union of non-timeout termination fires; consumed by
        # MotionCommand's adaptive sampling to weight motion bins by
        # episode-failure frequency. Refreshed on every
        # :meth:`check_termination` call.
        self._terminated_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        # Per-env mask of "bootstrap the terminal value this step" — union of
        # truncations and non-absorbing terminations, minus absorbing failures
        # (see :meth:`check_termination`). With no term setting
        # ``bootstrap_value=True`` this reduces exactly to ``truncated &
        # ~terminated`` (the historical PPO behaviour). Refreshed every step.
        self._bootstrap_buf = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

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
        # The one place these are read back to the host, once per training
        # iteration rather than once per term per step.
        n = int(self._iter_reset_count)
        if n == 0:
            return {}
        out = {f"Episode_Termination/{name}": int(count) / n for name, count in self._iter_fire_counts.items()}
        self._iter_reset_count.zero_()
        for count in self._iter_fire_counts.values():
            count.zero_()
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
        # Split non-timeout terminations by whether their terminal value should
        # be bootstrapped (non-absorbing, e.g. goal reached) or not (absorbing
        # failure). Used to build the bootstrap mask below.
        boot_term = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        absorb_term = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for name, term_config in self._all_terms.items():
            result: TerminationResult
            if name in self._instances:
                result = self._instances[name](self.env)
            else:
                result = self._resolved_fns[name](self.env, **term_config.params)

            self._term_dones[name] = result.reset
            self._episode_fires[name] += result.reset.long()

            if result.is_timeout:
                truncated |= result.reset
            else:
                terminated |= result.reset
                if term_config.bootstrap_value:
                    boot_term |= result.reset
                else:
                    absorb_term |= result.reset

            if result.extras:
                self.extras.update(result.extras)

        self._terminated_mask = terminated
        # Bootstrap iff (timeout OR non-absorbing termination) AND no absorbing
        # failure fired on the same env this step. With no bootstrap_value term
        # (boot_term all-False) this is ``truncated & ~terminated`` — identical
        # to the prior behaviour.
        self._bootstrap_buf = (truncated | boot_term) & ~absorb_term
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

    @property
    def bootstrap_buf(self) -> torch.Tensor:
        """Per-env mask of "bootstrap the terminal value this step".

        ``(truncated | non_absorbing_termination) & ~absorbing_failure``.
        Consumed by the on-policy bootstrap; with no ``bootstrap_value=True``
        term it equals ``truncated & ~terminated`` (historical behaviour).
        """
        return self._bootstrap_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return

        # Clear per-env state held by stateful terms, so a condition that
        # counts steps starts the new episode from zero rather than
        # inheriting the run the previous one ended on.
        for instance in self._instances.values():
            if hasattr(instance, "reset"):
                instance.reset(env_ids)

        # Fold the just-reset envs into the iteration-window accumulator.
        # Raw counts — not fractions — so
        # :meth:`consume_episode_stats` can compute an unbiased
        # iteration-wide ratio as (total term fires / total resets).
        # Per-env fire tallies are cleared after read so the next
        # episode starts with a fresh counter for this env.
        n_reset = env_ids.numel() if hasattr(env_ids, "numel") else len(env_ids)
        if n_reset > 0:
            self._iter_reset_count += n_reset
            # Every term at once. Integer counts, so the result is exactly
            # the per-term loop's.
            fires = self._episode_fires_all[:, env_ids]
            self._iter_fire_counts_all += (fires > 0).sum(dim=1)
            self._episode_fires_all[:, env_ids] = 0

        self.episode_count[env_ids] += 1
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

    def __str__(self) -> str:
        """Pretty print termination manager configuration."""
        from jaxrlworld.rl.utils.pretty import create_manager_table, table_to_string

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
