"""JAX-native event manager for Newton environments."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from rlworld.rl.configs.events.event_term_config import EventTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class EventManagerConfig:
    """Configuration for the event manager."""
    event_terms: list[EventTermConfig]


class JaxEventManager:
    """JAX-native event manager for domain randomization and disturbances."""

    def __init__(self, env: "World", config: EventManagerConfig):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self._terms_by_mode: dict[str, list[EventTermConfig]] = defaultdict(list)
        for term in config.event_terms:
            self._terms_by_mode[term.mode].append(term)

        self._interval_timers: dict[int, jax.Array] = {}
        self._interval_ranges: dict[int, tuple[float, float]] = {}
        self._rng_key = jax.random.PRNGKey(0)

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
        env_ids=None,
        dt: float | None = None
    ) -> None:
        if mode not in self._terms_by_mode:
            return

        if mode == "startup":
            self._apply_startup()
        elif mode == "reset":
            if env_ids is None:
                env_ids = jnp.arange(self.num_envs)
            self._apply_reset(env_ids)
        elif mode == "interval":
            if dt is None:
                raise ValueError("dt must be provided for interval mode")
            self._apply_interval(dt)

    def reset(self, env_ids) -> None:
        for idx in self._interval_timers:
            new_intervals = self._sample_interval(idx, batch_size=len(env_ids))
            self._interval_timers[idx] = self._interval_timers[idx].at[env_ids].set(new_intervals)

    def _apply_startup(self) -> None:
        for term in self._terms_by_mode["startup"]:
            term.func(self.env, **term.params)

    def _apply_reset(self, env_ids) -> None:
        for term in self._terms_by_mode["reset"]:
            term.func(self.env, env_ids, **term.params)

    def _apply_interval(self, dt: float) -> None:
        for idx, term in enumerate(self._terms_by_mode["interval"]):
            self._interval_timers[idx] = self._interval_timers[idx] - dt
            triggered_mask = self._interval_timers[idx] <= 0
            triggered_env_ids = jnp.where(triggered_mask)[0]

            if len(triggered_env_ids) > 0:
                term.func(self.env, triggered_env_ids, **term.params)
                new_intervals = self._sample_interval(idx, batch_size=len(triggered_env_ids))
                self._interval_timers[idx] = self._interval_timers[idx].at[triggered_env_ids].set(new_intervals)

    def _sample_interval(self, term_idx: int, batch_size: int | None = None) -> jax.Array:
        if batch_size is None:
            batch_size = self.num_envs
        min_interval, max_interval = self._interval_ranges[term_idx]
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return jax.random.uniform(subkey, shape=(batch_size,), minval=min_interval, maxval=max_interval)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not hasattr(self, '_terms_by_mode') or not any(self._terms_by_mode.values()):
            return ""

        rows = []
        all_terms = []
        for mode_terms in self._terms_by_mode.values():
            all_terms.extend(mode_terms)

        for idx, term in enumerate(all_terms):
            func_name = getattr(term.func, '__name__', f"term_{idx}")
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
