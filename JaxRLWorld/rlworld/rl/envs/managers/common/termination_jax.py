"""JAX-native termination manager for Newton environments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import TerminationTermConfig, TerminationResult

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class TerminationConfig:
    num_envs: int
    termination_criteria: list[TerminationTermConfig]
    episode_length_s: float


class JaxTerminationManager(BaseManager):
    """JAX-native termination manager."""

    def __init__(self, env: "World", config: TerminationConfig):
        super().__init__(env=env)
        self.config = config
        self.termination_criteria = config.termination_criteria

        self.reset_buf = jnp.ones(config.num_envs, dtype=jnp.bool_)
        self.episode_count = jnp.zeros(config.num_envs, dtype=jnp.int32)
        self.episode_length_buf = jnp.zeros(config.num_envs, dtype=jnp.int32)

        self.extras = {}

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.config.episode_length_s / self.env.control_dt)

    def advance(self) -> None:
        self.episode_length_buf = self.episode_length_buf + 1

    def check_termination(self) -> tuple[jax.Array, jax.Array]:
        terminated = jnp.zeros(self.env.num_envs, dtype=jnp.bool_)
        truncated = jnp.zeros(self.env.num_envs, dtype=jnp.bool_)

        for term_config in self.termination_criteria:
            result: TerminationResult = term_config.func(self.env, **term_config.params)

            if result.is_timeout:
                truncated = truncated | result.reset
            else:
                terminated = terminated | result.reset

            if result.extras:
                self.extras.update(result.extras)

        self.reset_buf = terminated | truncated
        return terminated, truncated

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            return
        self.episode_count = self.episode_count.at[env_ids].add(1)
        self.episode_length_buf = self.episode_length_buf.at[env_ids].set(0)
        self.reset_buf = self.reset_buf.at[env_ids].set(True)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self.termination_criteria:
            return ""

        rows = []
        for idx, term in enumerate(self.termination_criteria):
            func_name = getattr(term.func, '__name__', f"term_{idx}")
            if "timeout" in func_name.lower() or "time_out" in func_name.lower():
                type_str = "Truncation (timeout)"
            elif any(key in func_name.lower() for key in ["contact", "collision"]):
                type_str = "Termination"
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
            footer=f"Max Episode: {self.max_episode_length} steps ({self.config.episode_length_s}s)"
        )
        return table_to_string(table)
