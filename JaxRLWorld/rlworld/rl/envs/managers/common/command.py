from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import CommandTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return (angles + math.pi) % (2 * math.pi) - math.pi


@dataclass
class CommandManagerConfig:
    """Configuration for command management."""
    command_terms: list[CommandTermConfig] = None
    resampling_time_s: tuple[float, float] | float | None = None

    # Standing/heading (matching mjlab UniformVelocityCommandCfg)
    rel_standing_envs: float = 0.0
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: tuple[float, float] = (-3.14, 3.14)
    rel_heading_envs: float = 1.0


class CommandManager(BaseManager):
    """Manages command generation with optional heading control and standing envs.

    Compatible with Genesis, Newton, and MuJoCo simulators.
    When heading_command or rel_standing_envs are left at defaults (off/0.0),
    behavior is identical to the original CommandManager.
    """

    def __init__(self, env: "World", config: CommandManagerConfig):
        super().__init__(env)
        self.config = config
        self.num_envs = env.num_envs
        self.control_dt = env.control_dt

        self.command_names = [term.func.__name__ for term in config.command_terms]
        self.num_commands = len(self.command_names)
        self._command_indices = {name: idx for idx, name in enumerate(self.command_names)}

        self._commands_tensor = torch.zeros(
            (self.num_envs, self.num_commands),
            device=self.device
        )
        self._next_resample_time = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Standing/heading state
        self.is_standing_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.heading_target = torch.zeros(
            self.num_envs, device=self.device
        )

        # Cache ang_vel_z index for heading control
        self._ang_vel_z_idx: int | None = self._command_indices.get("ang_vel", None)

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

    def resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample commands for given envs, then apply standing/heading masks."""
        # Sample raw commands from each term
        for cmd_idx, term in enumerate(self.config.command_terms):
            raw_command = term.func(self.env, env_ids, **term.params)
            self._commands_tensor[env_ids, cmd_idx] = raw_command * term.scale

        # Determine standing envs
        if self.config.rel_standing_envs > 0.0:
            r = torch.rand(len(env_ids), device=self.device)
            self.is_standing_env[env_ids] = r <= self.config.rel_standing_envs

        # Determine heading envs and sample heading target
        if self.config.heading_command:
            r = torch.rand(len(env_ids), device=self.device)
            self.is_heading_env[env_ids] = r <= self.config.rel_heading_envs
            self.heading_target[env_ids] = torch.empty(
                len(env_ids), device=self.device
            ).uniform_(*self.config.heading_range)

        # Schedule next resample
        self._schedule_next_resample(env_ids)

    def update_commands(self, episode_length: torch.Tensor) -> None:
        """Update commands: resample if needed, then apply heading/standing."""
        if self.config.resampling_time_s is not None:
            envs_to_resample = (
                episode_length == self._next_resample_time
            ).nonzero(as_tuple=False).flatten()

            if len(envs_to_resample) > 0:
                self.resample_commands(envs_to_resample)

        # Apply heading P-control (overwrite ang_vel_z for heading envs)
        if self.config.heading_command and self._ang_vel_z_idx is not None:
            heading_w = self.env.heading_w  # [num_envs]
            heading_error = _wrap_to_pi(self.heading_target - heading_w)

            heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(heading_env_ids) > 0:
                # Find ang_vel_z range for clamping
                ang_vel_term = self.config.command_terms[self._ang_vel_z_idx]
                ang_vel_range = ang_vel_term.params.get("range", (-1.0, 1.0))

                self._commands_tensor[heading_env_ids, self._ang_vel_z_idx] = torch.clamp(
                    self.config.heading_control_stiffness * heading_error[heading_env_ids],
                    min=ang_vel_range[0],
                    max=ang_vel_range[1],
                )

        # Zero out standing envs (after heading, so standing takes priority)
        if self.config.rel_standing_envs > 0.0:
            standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            if len(standing_env_ids) > 0:
                self._commands_tensor[standing_env_ids, :] = 0.0

    def get_commands_tensor(self) -> torch.Tensor:
        return self._commands_tensor

    def _schedule_next_resample(self, env_ids: torch.Tensor) -> None:
        """Schedule next resample time for given envs."""
        if isinstance(self.config.resampling_time_s, (tuple, list)):
            min_time, max_time = self.config.resampling_time_s
            min_steps = int(min_time / self.control_dt)
            max_steps = int(max_time / self.control_dt)
            random_intervals = torch.randint(
                min_steps, max_steps + 1, (len(env_ids),), device=self.device
            )
            self._next_resample_time[env_ids] = (
                self.env.episode_length_buf[env_ids] + random_intervals
            )
        elif self.config.resampling_time_s is not None:
            fixed_interval = int(self.config.resampling_time_s / self.control_dt)
            self._next_resample_time[env_ids] = (
                self.env.episode_length_buf[env_ids] + fixed_interval
            )

    def __str__(self) -> str:
        """Pretty print command manager configuration."""
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
                resample_str = (
                    f"{self.config.resampling_time_s[0]}"
                    f"-{self.config.resampling_time_s[1]}s"
                )
            else:
                resample_str = f"{self.config.resampling_time_s}s"

        footer_parts = [f"Resample: {resample_str}"]
        if self.config.rel_standing_envs > 0:
            footer_parts.append(
                f"Standing: {self.config.rel_standing_envs:.0%}"
            )
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