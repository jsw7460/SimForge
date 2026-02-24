from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.events.event_term_config import EventTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class EventManagerConfig:
    """Configuration for the event manager."""
    event_terms: list[EventTermConfig]


class EventManager:
    """Manages event execution for domain randomization and disturbances."""

    def __init__(self, env: "World", config: EventManagerConfig):
        self.env = env
        self.config = config
        self.device = env.device
        self.num_envs = env.num_envs

        self._terms_by_mode: dict[str, list[EventTermConfig]] = defaultdict(list)
        for term in config.event_terms:
            self._terms_by_mode[term.mode].append(term)

        self._interval_timers: dict[int, torch.Tensor] = {}
        self._interval_ranges: dict[int, tuple[float, float]] = {}

        for idx, term in enumerate(self._terms_by_mode.get("interval", [])):
            if term.interval_range_s is None:
                raise ValueError(
                    f"Interval event term must have interval_range_s specified. "
                    f"Got None for term with func: {term.func.__name__}"
                )
            self._interval_ranges[idx] = term.interval_range_s
            self._interval_timers[idx] = self._sample_interval(idx)

    @property
    def available_modes(self) -> list[str]:
        return list(self._terms_by_mode.keys())

    def apply(
        self,
        mode: str,
        env_ids: torch.Tensor | None = None,
        dt: float | None = None
    ) -> None:
        if mode not in self._terms_by_mode:
            return

        if mode == "startup":
            self._apply_startup()
        elif mode == "reset":
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._apply_reset(env_ids)
        elif mode == "interval":
            if dt is None:
                raise ValueError("dt must be provided for interval mode")
            self._apply_interval(dt)

    def reset(self, env_ids: torch.Tensor) -> None:
        for idx in self._interval_timers:
            new_intervals = self._sample_interval(idx, batch_size=len(env_ids))
            self._interval_timers[idx][env_ids] = new_intervals

    def _apply_startup(self) -> None:
        for term in self._terms_by_mode["startup"]:
            term.func(self.env, **term.params)

    def _apply_reset(self, env_ids: torch.Tensor) -> None:
        for term in self._terms_by_mode["reset"]:
            term.func(self.env, env_ids, **term.params)

    def _apply_interval(self, dt: float) -> None:
        for idx, term in enumerate(self._terms_by_mode["interval"]):
            self._interval_timers[idx] -= dt
            triggered_mask = self._interval_timers[idx] <= 0
            triggered_env_ids = triggered_mask.nonzero(as_tuple=False).flatten()

            if len(triggered_env_ids) > 0:
                term.func(self.env, triggered_env_ids, **term.params)
                new_intervals = self._sample_interval(idx, batch_size=len(triggered_env_ids))
                self._interval_timers[idx][triggered_env_ids] = new_intervals

    def _sample_interval(self, term_idx: int, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.num_envs
        min_interval, max_interval = self._interval_ranges[term_idx]
        return torch.empty(batch_size, device=self.device).uniform_(min_interval, max_interval)

    def __str__(self) -> str:
        """Pretty print event manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self.config.event_terms:
            return ""

        rows = []
        for idx, term in enumerate(self.config.event_terms):
            func_name = getattr(term.func, '__name__', f"term_{idx}")
            mode_str = term.mode.capitalize()

            # Format interval if present
            interval_str = "-"
            if term.interval_range_s is not None:
                min_t, max_t = term.interval_range_s
                interval_str = f"{min_t}-{max_t}s"

            # Format key params
            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([idx, func_name, mode_str, interval_str, params_str])

        # Count events by mode
        mode_counts = {}
        for mode in self.available_modes:
            count = len(self._terms_by_mode.get(mode, []))
            mode_counts[mode] = count
        footer = ", ".join(f"{mode}: {count}" for mode, count in mode_counts.items())

        table = create_manager_table(
            title="Event Terms",
            columns=["Idx", "Name", "Mode", "Interval", "Params"],
            rows=rows,
            footer=footer
        )
        return table_to_string(table)