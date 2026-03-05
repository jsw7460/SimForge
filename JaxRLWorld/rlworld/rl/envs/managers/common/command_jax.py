"""JAX-native command manager for Newton environments."""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import CommandTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _wrap_to_pi(angles: jax.Array) -> jax.Array:
    return (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi


@dataclass
class CommandManagerConfig:
    command_terms: list[CommandTermConfig] = None
    resampling_time_s: tuple[float, float] | float | None = None
    rel_standing_envs: float = 0.0
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: tuple[float, float] = (-3.14, 3.14)
    rel_heading_envs: float = 1.0


class JaxCommandManager(BaseManager):
    """JAX-native command manager."""

    def __init__(self, env: "World", config: CommandManagerConfig):
        super().__init__(env)
        self.config = config
        self.num_envs = env.num_envs
        self.control_dt = env.control_dt

        self.command_names = [term.func.__name__ for term in config.command_terms]
        self.num_commands = len(self.command_names)
        self._command_indices = {name: idx for idx, name in enumerate(self.command_names)}

        self._commands_tensor = jnp.zeros((self.num_envs, self.num_commands))
        self._next_resample_time = jnp.zeros(self.num_envs, dtype=jnp.int32)

        self.is_standing_env = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        self.is_heading_env = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        self.heading_target = jnp.zeros(self.num_envs)

        self._ang_vel_z_idx: int | None = self._command_indices.get("ang_vel", None)
        self._rng_key = jax.random.PRNGKey(0)

    def __getattr__(self, name: str):
        if name.startswith('_') or name in [
            'config', 'num_envs', 'device', 'control_dt',
            'command_names', 'num_commands', 'env',
            'is_standing_env', 'is_heading_env', 'heading_target',
        ]:
            return object.__getattribute__(self, name)

        _command_indices = object.__getattribute__(self, '_command_indices')
        if name in _command_indices:
            idx = _command_indices[name]
            _commands_tensor = object.__getattribute__(self, '_commands_tensor')
            return _commands_tensor[:, idx]

        return object.__getattribute__(self, name)

    def resample_commands(self, env_ids) -> None:
        for cmd_idx, term in enumerate(self.config.command_terms):
            raw_command = term.func(self.env, env_ids, **term.params)
            self._commands_tensor = self._commands_tensor.at[env_ids, cmd_idx].set(
                raw_command * term.scale
            )

        if self.config.rel_standing_envs > 0.0:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            r = jax.random.uniform(subkey, shape=(len(env_ids),))
            self.is_standing_env = self.is_standing_env.at[env_ids].set(
                r <= self.config.rel_standing_envs
            )

        if self.config.heading_command:
            self._rng_key, subkey1, subkey2 = jax.random.split(self._rng_key, 3)
            r = jax.random.uniform(subkey1, shape=(len(env_ids),))
            self.is_heading_env = self.is_heading_env.at[env_ids].set(
                r <= self.config.rel_heading_envs
            )
            heading = jax.random.uniform(
                subkey2, shape=(len(env_ids),),
                minval=self.config.heading_range[0],
                maxval=self.config.heading_range[1],
            )
            self.heading_target = self.heading_target.at[env_ids].set(heading)

        self._schedule_next_resample(env_ids)

    def update_commands(self, episode_length) -> None:
        if self.config.resampling_time_s is not None:
            envs_to_resample = jnp.where(
                episode_length == self._next_resample_time,
                True, False
            )
            env_ids = jnp.where(envs_to_resample)[0]

            if len(env_ids) > 0:
                self.resample_commands(env_ids)

        if self.config.heading_command and self._ang_vel_z_idx is not None:
            heading_w = self.env.heading_w
            heading_error = _wrap_to_pi(self.heading_target - heading_w)

            heading_env_ids = jnp.where(self.is_heading_env)[0]
            if len(heading_env_ids) > 0:
                ang_vel_term = self.config.command_terms[self._ang_vel_z_idx]
                ang_vel_range = ang_vel_term.params.get("range", (-1.0, 1.0))

                new_ang_vel = jnp.clip(
                    self.config.heading_control_stiffness * heading_error[heading_env_ids],
                    ang_vel_range[0],
                    ang_vel_range[1],
                )
                self._commands_tensor = self._commands_tensor.at[
                    heading_env_ids, self._ang_vel_z_idx
                ].set(new_ang_vel)

        if self.config.rel_standing_envs > 0.0:
            standing_env_ids = jnp.where(self.is_standing_env)[0]
            if len(standing_env_ids) > 0:
                self._commands_tensor = self._commands_tensor.at[standing_env_ids, :].set(0.0)

    def get_commands_tensor(self) -> jax.Array:
        return self._commands_tensor

    def _schedule_next_resample(self, env_ids) -> None:
        if isinstance(self.config.resampling_time_s, (tuple, list)):
            min_time, max_time = self.config.resampling_time_s
            min_steps = int(min_time / self.control_dt)
            max_steps = int(max_time / self.control_dt)
            self._rng_key, subkey = jax.random.split(self._rng_key)
            random_intervals = jax.random.randint(
                subkey, shape=(len(env_ids),),
                minval=min_steps, maxval=max_steps + 1,
            )
            self._next_resample_time = self._next_resample_time.at[env_ids].set(
                self.env.episode_length_buf[env_ids] + random_intervals
            )
        elif self.config.resampling_time_s is not None:
            fixed_interval = int(self.config.resampling_time_s / self.control_dt)
            self._next_resample_time = self._next_resample_time.at[env_ids].set(
                self.env.episode_length_buf[env_ids] + fixed_interval
            )

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self.config.command_terms:
            return ""

        rows = []
        for idx, term in enumerate(self.config.command_terms):
            func_name = getattr(term.func, '__name__', f"term_{idx}")
            scale_str = f"{term.scale}" if term.scale != 1.0 else "1.0"
            range_str = "-"
            if term.params:
                for key in ["range", "ranges", "min_max"]:
                    if key in term.params:
                        val = term.params[key]
                        if isinstance(val, (list, tuple)) and len(val) == 2:
                            range_str = f"[{val[0]}, {val[1]}]"
                        break
            rows.append([idx, func_name, scale_str, range_str])

        resample_str = "-"
        if self.config.resampling_time_s is not None:
            if isinstance(self.config.resampling_time_s, (tuple, list)):
                resample_str = f"{self.config.resampling_time_s[0]}-{self.config.resampling_time_s[1]}s"
            else:
                resample_str = f"{self.config.resampling_time_s}s"

        footer_parts = [f"Resample: {resample_str}"]
        if self.config.rel_standing_envs > 0:
            footer_parts.append(f"Standing: {self.config.rel_standing_envs:.0%}")
        if self.config.heading_command:
            footer_parts.append(
                f"Heading: {self.config.rel_heading_envs:.0%}"
                f" (K={self.config.heading_control_stiffness})"
            )

        table = create_manager_table(
            title="Command Terms",
            columns=["Idx", "Name", "Scale", "Range"],
            rows=rows,
            footer=" | ".join(footer_parts),
        )
        return table_to_string(table)
