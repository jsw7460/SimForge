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
            self.time_left[resample_ids] = torch.empty(len(resample_ids), device=self.device).uniform_(*self.cfg.resampling_time_range)
            self._resample_command(resample_ids)
        self._update_command()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Force resample for the given environments."""
        self.time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.resampling_time_range)
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
        self._command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_x_range)
        self._command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_y_range)
        self._command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.ang_vel_range)
        if self.cfg.rel_standing_envs > 0.0:
            r = torch.rand(n, device=self.device)
            self.is_standing_env[env_ids] = r < self.cfg.rel_standing_envs

        if self.cfg.heading_command:
            r = torch.rand(n, device=self.device)
            self.is_heading_env[env_ids] = r < self.cfg.rel_heading_envs
            self.heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.heading_range)

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


# ──────────────────────────────────────────────
# GaitCommandTerm
# ──────────────────────────────────────────────

# Column indices for the gait command tensor.
_GAIT_FREQ = 0
_GAIT_PHASE = 1
_GAIT_OFFSET = 2
_GAIT_BOUND = 3
_GAIT_DURATION = 4
_FOOTSWING_HEIGHT = 5
_BODY_HEIGHT = 6
_BODY_PITCH = 7
_BODY_ROLL = 8
_STANCE_WIDTH = 9
_STANCE_LENGTH = 10
_GAIT_DIM = 11


@dataclass
class GaitCommandTermCfg(CommandTermCfg):
    """Configuration for gait behavior command sampling.

    Produces an 11-dim behavior command matching Walk-These-Ways:
        [gait_freq, gait_phase, gait_offset, gait_bound, gait_duration,
         footswing_height, body_height, body_pitch, body_roll,
         stance_width, stance_length]

    Gait phase/offset/bound are sampled uniformly then post-processed
    according to the selected ``gait_category_mode``.
    """
    # Sampling ranges for each parameter.
    freq_range: tuple[float, float] = (2.0, 4.0)
    phase_range: tuple[float, float] = (0.0, 1.0)
    offset_range: tuple[float, float] = (0.0, 1.0)
    bound_range: tuple[float, float] = (0.0, 1.0)
    duration_range: tuple[float, float] = (0.5, 0.5)
    footswing_height_range: tuple[float, float] = (0.03, 0.35)
    body_height_range: tuple[float, float] = (-0.25, 0.15)
    body_pitch_range: tuple[float, float] = (-0.4, 0.4)
    body_roll_range: tuple[float, float] = (0.0, 0.0)
    stance_width_range: tuple[float, float] = (0.10, 0.45)
    stance_length_range: tuple[float, float] = (0.35, 0.45)

    # Gait category post-processing mode.
    #   "gaitwise":    Sample category (pronk/trot/pace/bound), constrain
    #                  phase/offset/bound to that category. Matches WTW
    #                  ``gaitwise_curricula`` mode.
    #   "exclusive":   Randomly zero out two of three phase offsets.
    #                  Matches WTW ``exclusive_phase_offset`` mode.
    #   "balanced":    25% each for pronk/trot/pace/bound.
    #                  Matches WTW ``balance_gait_distribution`` mode.
    #   "none":        No post-processing; phase/offset/bound are independent.
    gait_category_mode: str = "gaitwise"

    # Category names and their equal probability.
    # Used by "gaitwise" mode.
    categories: tuple[str, ...] = ("pronk", "trot", "pace", "bound")

    # Quantize phases to {0, 0.5} after category post-processing.
    # Matches WTW ``binary_phases``.
    binary_phases: bool = True

    def build(self, env: "World") -> "GaitCommandTerm":
        return GaitCommandTerm(env, self)


class GaitCommandTerm(CommandTerm):
    """11-dim gait behavior command.

    Sampling logic follows Walk-These-Ways (Margolis & Agrawal, CoRL 2022).
    Each resample:
        1. Sample all 11 dims uniformly from configured ranges.
        2. Apply gait category post-processing on phase/offset/bound.
        3. Optionally quantize phases to binary {0, 0.5}.
    """

    column_names = (
        "gait_freq", "gait_phase", "gait_offset", "gait_bound", "gait_duration",
        "footswing_height", "body_height", "body_pitch", "body_roll",
        "stance_width", "stance_length",
    )

    cfg: GaitCommandTermCfg

    def __init__(self, env: "World", cfg: GaitCommandTermCfg):
        super().__init__(env, cfg)
        self._command = torch.zeros(self.num_envs, _GAIT_DIM, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        cfg = self.cfg

        # 1. Uniform sampling for all parameters.
        self._command[env_ids, _GAIT_FREQ] = torch.empty(n, device=self.device).uniform_(*cfg.freq_range)
        self._command[env_ids, _GAIT_PHASE] = torch.empty(n, device=self.device).uniform_(*cfg.phase_range)
        self._command[env_ids, _GAIT_OFFSET] = torch.empty(n, device=self.device).uniform_(*cfg.offset_range)
        self._command[env_ids, _GAIT_BOUND] = torch.empty(n, device=self.device).uniform_(*cfg.bound_range)
        self._command[env_ids, _GAIT_DURATION] = torch.empty(n, device=self.device).uniform_(*cfg.duration_range)
        self._command[env_ids, _FOOTSWING_HEIGHT] = torch.empty(n, device=self.device).uniform_(*cfg.footswing_height_range)
        self._command[env_ids, _BODY_HEIGHT] = torch.empty(n, device=self.device).uniform_(*cfg.body_height_range)
        self._command[env_ids, _BODY_PITCH] = torch.empty(n, device=self.device).uniform_(*cfg.body_pitch_range)
        self._command[env_ids, _BODY_ROLL] = torch.empty(n, device=self.device).uniform_(*cfg.body_roll_range)
        self._command[env_ids, _STANCE_WIDTH] = torch.empty(n, device=self.device).uniform_(*cfg.stance_width_range)
        self._command[env_ids, _STANCE_LENGTH] = torch.empty(n, device=self.device).uniform_(*cfg.stance_length_range)

        # 2. Gait category post-processing on phase/offset/bound.
        self._apply_gait_categories(env_ids)

        # 3. Binary phase quantization.
        if cfg.binary_phases:
            for col in (_GAIT_PHASE, _GAIT_OFFSET, _GAIT_BOUND):
                raw = self._command[env_ids, col]
                self._command[env_ids, col] = (torch.round(2 * raw) / 2.0) % 1.0

    def _apply_gait_categories(self, env_ids: torch.Tensor) -> None:
        """Post-process phase/offset/bound based on gait category mode.

        Exactly replicates the Walk-These-Ways ``_resample_commands``
        gait category logic.
        """
        mode = self.cfg.gait_category_mode
        if mode == "none":
            return

        n = len(env_ids)
        rand = torch.rand(n, device=self.device)
        cmd = self._command

        if mode == "gaitwise":
            # Equal probability per category.
            cats = self.cfg.categories
            num_cats = len(cats)
            prob = 1.0 / num_cats

            for i, cat in enumerate(cats):
                mask = (prob * i <= rand) & (rand < prob * (i + 1))
                ids = env_ids[mask]
                if len(ids) == 0:
                    continue

                if cat == "pronk":
                    cmd[ids, _GAIT_PHASE] = (cmd[ids, _GAIT_PHASE] / 2 - 0.25) % 1
                    cmd[ids, _GAIT_OFFSET] = (cmd[ids, _GAIT_OFFSET] / 2 - 0.25) % 1
                    cmd[ids, _GAIT_BOUND] = (cmd[ids, _GAIT_BOUND] / 2 - 0.25) % 1
                elif cat == "trot":
                    cmd[ids, _GAIT_PHASE] = cmd[ids, _GAIT_PHASE] / 2 + 0.25
                    cmd[ids, _GAIT_OFFSET] = 0
                    cmd[ids, _GAIT_BOUND] = 0
                elif cat == "pace":
                    cmd[ids, _GAIT_PHASE] = 0
                    cmd[ids, _GAIT_OFFSET] = cmd[ids, _GAIT_OFFSET] / 2 + 0.25
                    cmd[ids, _GAIT_BOUND] = 0
                elif cat == "bound":
                    cmd[ids, _GAIT_PHASE] = 0
                    cmd[ids, _GAIT_OFFSET] = 0
                    cmd[ids, _GAIT_BOUND] = cmd[ids, _GAIT_BOUND] / 2 + 0.25

        elif mode == "exclusive":
            # Randomly zero out two of three offsets.
            trot = env_ids[rand < 0.34]
            pace = env_ids[(0.34 <= rand) & (rand < 0.67)]
            bound = env_ids[rand >= 0.67]
            cmd[pace, _GAIT_PHASE] = 0
            cmd[bound, _GAIT_PHASE] = 0
            cmd[trot, _GAIT_OFFSET] = 0
            cmd[bound, _GAIT_OFFSET] = 0
            cmd[trot, _GAIT_BOUND] = 0
            cmd[pace, _GAIT_BOUND] = 0

        elif mode == "balanced":
            # 25% each for pronk/trot/pace/bound.
            pronk = env_ids[rand <= 0.25]
            trot = env_ids[(0.25 < rand) & (rand <= 0.50)]
            pace = env_ids[(0.50 < rand) & (rand <= 0.75)]
            bound = env_ids[rand > 0.75]
            # Pronk: all ~0
            cmd[pronk, _GAIT_PHASE] = (cmd[pronk, _GAIT_PHASE] / 2 - 0.25) % 1
            cmd[pronk, _GAIT_OFFSET] = (cmd[pronk, _GAIT_OFFSET] / 2 - 0.25) % 1
            cmd[pronk, _GAIT_BOUND] = (cmd[pronk, _GAIT_BOUND] / 2 - 0.25) % 1
            # Trot: phase~0.5, offset=0, bound=0
            cmd[trot, _GAIT_OFFSET] = 0
            cmd[trot, _GAIT_BOUND] = 0
            cmd[trot, _GAIT_PHASE] = cmd[trot, _GAIT_PHASE] / 2 + 0.25
            # Pace: phase=0, offset~0.5, bound=0
            cmd[pace, _GAIT_PHASE] = 0
            cmd[pace, _GAIT_BOUND] = 0
            cmd[pace, _GAIT_OFFSET] = cmd[pace, _GAIT_OFFSET] / 2 + 0.25
            # Bound: phase=0, offset=0, bound~0.5
            cmd[bound, _GAIT_PHASE] = 0
            cmd[bound, _GAIT_OFFSET] = 0
            cmd[bound, _GAIT_BOUND] = cmd[bound, _GAIT_BOUND] / 2 + 0.25
