"""CommandTerm base class and built-in implementations.

A CommandTerm encapsulates a group of related commands with their own
sampling logic, resampling timer, and per-step post-processing.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return (angles + math.pi) % (2 * math.pi) - math.pi


# ──────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────

@dataclass
class CommandTermCfg(ABC):
    """Configuration for a CommandTerm.

    Each subclass defines its own fields (ranges, flags, etc.)
    and implements ``build()`` to construct the corresponding CommandTerm.
    """
    resampling_time_range: tuple[float, float] = (5.0, 10.0)

    @abstractmethod
    def build(self, env: "World") -> "CommandTerm":
        ...


class CommandTerm(ABC):
    """Abstract base for a group of related commands.

    Subclasses must implement:
        ``command`` (property):      Return the command tensor [num_envs, dim].
        ``_resample_command(ids)``:  Sample new commands for given env ids.

    Optionally override:
        ``_update_command()``:       Per-step post-processing (e.g., heading control).
        ``column_names``:            Tuple of names for each column (for attribute access).
    """

    column_names: tuple[str, ...] = ()

    def __init__(self, env: "World", cfg: CommandTermCfg):
        self._env = env
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.time_left = torch.zeros(self.num_envs, device=self.device)

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """Command tensor of shape [num_envs, command_dim]."""
        ...

    @abstractmethod
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new commands for the given environment indices."""
        ...

    def _update_command(self) -> None:
        """Per-step post-processing. Override if needed."""
        pass

    def compute(self, dt: float) -> None:
        """Advance timer, resample if expired, then post-process."""
        self.time_left -= dt
        resample_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self.time_left[resample_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(resample_ids)
        self._update_command()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Force resample for the given environments."""
        self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
        self._resample_command(env_ids)


# ──────────────────────────────────────────────
# VelocityCommandTerm
# ──────────────────────────────────────────────

@dataclass
class VelocityCommandTermCfg(CommandTermCfg):
    """Configuration for uniform velocity command sampling.

    Samples (lin_vel_x, lin_vel_y, ang_vel) uniformly from configured ranges.
    Optionally applies heading P-control and standing-env zeroing.
    """
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)

    rel_standing_envs: float = 0.0
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: tuple[float, float] = (-3.14, 3.14)
    rel_heading_envs: float = 1.0

    def build(self, env: "World") -> "VelocityCommandTerm":
        return VelocityCommandTerm(env, self)


class VelocityCommandTerm(CommandTerm):
    """3-dim velocity command: [lin_vel_x, lin_vel_y, ang_vel]."""

    column_names = ("lin_vel_x", "lin_vel_y", "ang_vel")

    cfg: VelocityCommandTermCfg

    def __init__(self, env: "World", cfg: VelocityCommandTermCfg):
        super().__init__(env, cfg)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def lin_vel_x(self) -> torch.Tensor:
        return self._command[:, 0]

    @property
    def lin_vel_y(self) -> torch.Tensor:
        return self._command[:, 1]

    @property
    def ang_vel(self) -> torch.Tensor:
        return self._command[:, 2]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._command[env_ids, 0].uniform_(*self.cfg.lin_vel_x_range)
        self._command[env_ids, 1].uniform_(*self.cfg.lin_vel_y_range)
        self._command[env_ids, 2].uniform_(*self.cfg.ang_vel_range)

        if self.cfg.rel_standing_envs > 0.0:
            r = torch.rand(n, device=self.device)
            self.is_standing_env[env_ids] = r < self.cfg.rel_standing_envs

        if self.cfg.heading_command:
            r = torch.rand(n, device=self.device)
            self.is_heading_env[env_ids] = r < self.cfg.rel_heading_envs
            self.heading_target[env_ids].uniform_(*self.cfg.heading_range)

    def _update_command(self) -> None:
        # Heading P-control: overwrite ang_vel for heading envs
        if self.cfg.heading_command:
            heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(heading_ids) > 0:
                heading_w = self._env.heading_w
                heading_error = _wrap_to_pi(self.heading_target - heading_w)
                self._command[heading_ids, 2] = torch.clamp(
                    self.cfg.heading_control_stiffness * heading_error[heading_ids],
                    self.cfg.ang_vel_range[0],
                    self.cfg.ang_vel_range[1],
                )

        # Standing envs: zero all commands
        if self.cfg.rel_standing_envs > 0.0:
            standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            if len(standing_ids) > 0:
                self._command[standing_ids] = 0.0
