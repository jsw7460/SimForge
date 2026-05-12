from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.common_config_classes import EventConfig
from rlworld.rl.configs.events.event_term_config import EventTermConfig
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World

# Backward-compatible alias
EventManagerConfig = EventConfig


class EventManager(BaseManager):
    """Manages event execution for domain randomization and disturbances.

    Terms are discovered via :func:`iter_terms` on the ``EventConfig`` instance.
    """

    def __init__(self, env: World, config: EventConfig):
        super().__init__(env=env)
        self.config = config
        self.num_envs = env.num_envs

        # Discover named terms and resolve callables
        self._all_terms: dict[str, EventTermConfig] = iter_terms(config, EventTermConfig)
        self._resolved_fns: dict[str, callable] = {name: term.resolved_func for name, term in self._all_terms.items()}
        # Pre-resolve SceneEntitySelector params (and selector-typed
        # defaults) so event functions receive a ResolvedEntity directly.
        for term in self._all_terms.values():
            self._resolve_term_selectors(term.resolved_func, term.params)

        # Group by mode
        self._terms_by_mode: dict[str, list[tuple[str, EventTermConfig]]] = defaultdict(list)
        for name, term in self._all_terms.items():
            self._terms_by_mode[term.mode].append((name, term))

        # Set up interval timers
        self._interval_timers: dict[int, torch.Tensor] = {}
        self._interval_ranges: dict[int, tuple[float, float]] = {}

        for local_idx, (name, term) in enumerate(self._terms_by_mode.get("interval", [])):
            if term.interval_range_s is None:
                raise ValueError(f"Interval event term '{name}' must have interval_range_s specified.")
            self._interval_ranges[local_idx] = term.interval_range_s
            self._interval_timers[local_idx] = self._sample_interval(local_idx)

    @property
    def available_modes(self) -> list[str]:
        return list(self._terms_by_mode.keys())

    def apply(self, mode: str, env_ids: torch.Tensor | None = None, dt: float | None = None) -> None:
        if mode not in self._terms_by_mode:
            return

        if mode == "startup":
            self._apply_startup()
        elif mode == "reset":
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._apply_reset(env_ids)
        elif mode == "reset_dr":
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._apply_reset_dr(env_ids)
        elif mode == "interval":
            if dt is None:
                raise ValueError("dt must be provided for interval mode")
            self._apply_interval(dt)

    def reset(self, env_ids: torch.Tensor) -> None:
        for idx in self._interval_timers:
            new_intervals = self._sample_interval(idx, batch_size=len(env_ids))
            self._interval_timers[idx][env_ids] = new_intervals

    def _call_event_fn(self, name: str, term: EventTermConfig, env_ids: torch.Tensor) -> None:
        self._resolved_fns[name](env=self.env, env_ids=env_ids, **term.params)

    def _apply_startup(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        for name, term in self._terms_by_mode["startup"]:
            self._call_event_fn(name, term, env_ids=env_ids)

    def _apply_reset(self, env_ids: torch.Tensor) -> None:
        for name, term in self._terms_by_mode["reset"]:
            self._call_event_fn(name, term, env_ids=env_ids)

    def _apply_reset_dr(self, env_ids: torch.Tensor) -> None:
        for name, term in self._terms_by_mode["reset_dr"]:
            self._call_event_fn(name, term, env_ids=env_ids)

    def _apply_interval(self, dt: float) -> None:
        for local_idx, (name, term) in enumerate(self._terms_by_mode["interval"]):
            self._interval_timers[local_idx] -= dt
            triggered_mask = self._interval_timers[local_idx] <= 0
            triggered_env_ids = triggered_mask.nonzero(as_tuple=False).flatten()

            if len(triggered_env_ids) > 0:
                self._call_event_fn(name, term, env_ids=triggered_env_ids)
                new_intervals = self._sample_interval(local_idx, batch_size=len(triggered_env_ids))
                self._interval_timers[local_idx][triggered_env_ids] = new_intervals

    def _sample_interval(self, term_idx: int, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.num_envs
        min_interval, max_interval = self._interval_ranges[term_idx]
        return torch.empty(batch_size, device=self.device).uniform_(min_interval, max_interval)

    def __str__(self) -> str:
        """Pretty print event manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._all_terms:
            return ""

        rows = []
        for idx, (name, term) in enumerate(self._all_terms.items()):
            func_name = getattr(self._resolved_fns[name], "__name__", name)
            mode_str = term.mode.capitalize()

            interval_str = "-"
            if term.interval_range_s is not None:
                min_t, max_t = term.interval_range_s
                interval_str = f"{min_t}-{max_t}s"

            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([idx, func_name, mode_str, interval_str, params_str])

        mode_counts = {mode: len(terms) for mode, terms in self._terms_by_mode.items()}
        footer = ", ".join(f"{mode}: {count}" for mode, count in mode_counts.items())

        table = create_manager_table(
            title="Event Terms", columns=["Idx", "Name", "Mode", "Interval", "Params"], rows=rows, footer=footer
        )
        return table_to_string(table)
