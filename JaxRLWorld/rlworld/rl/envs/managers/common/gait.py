from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch

from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from rlworld.rl.envs.managers.common.command import CommandManager


# ────────────────────────────────────────────────────
# Foot-offset providers
# ────────────────────────────────────────────────────
#
# A foot-offset provider is a callable that reads gait-related commands
# and returns per-foot phase offsets.
#
#   (CommandManager) -> [num_envs, num_feet]
#
# Users plug one of these into GaitConfig.foot_offset_provider to control how commands map to per-foot timing.


class FootOffsetProvider(Protocol):
    """Protocol for foot-offset providers."""
    def __call__(self, cmd: "CommandManager") -> torch.Tensor: ...


class QuadrupedOffsets:
    """Reads (phase, offset, bound) commands and produces 4-foot offsets.

    Maps the three gait parameters to per-foot phase offsets using
    the Walk-These-Ways convention:
        FL = phase + offset + bound
        FR = offset
        RL = bound
        RR = phase

    This parameterization expresses all symmetric quadrupedal gaits:
        - Trot:  (0.5, 0, 0) -- diagonal legs in sync
        - Pace:  (0, 0.5, 0) -- same-side legs in sync
        - Bound: (0, 0, 0.5) -- front/hind legs in sync
        - Pronk: (0, 0, 0)   -- all legs in sync

    Output order is determined by ``foot_names``: each name is matched
    to FL/FR/RL/RR by looking for those substrings, so the output
    aligns with GaitManager's foot order regardless of config ordering.

    Args:
        foot_names: Foot link names (e.g., ["FR_foot", "FL_foot", "RL_foot", "RR_foot"]).
                    Each name must contain exactly one of "FL", "FR", "RL", "RR".
        phase_cmd:  Command name for phase parameter (θ₁).
        offset_cmd: Command name for offset parameter (θ₂).
        bound_cmd:  Command name for bound parameter (θ₃).
    """

    # Canonical offset formula for each leg.
    _LEG_FORMULAS = {
        "FL": lambda p, o, b: p + o + b,
        "FR": lambda p, o, b: o,
        "RL": lambda p, o, b: b,
        "RR": lambda p, o, b: p,
    }

    def __init__(
        self,
        foot_names: tuple[str, ...] | list[str],
        phase_cmd: str = "gait_phase",
        offset_cmd: str = "gait_offset",
        bound_cmd: str = "gait_bound",
    ):
        self.phase_cmd = phase_cmd
        self.offset_cmd = offset_cmd
        self.bound_cmd = bound_cmd

        self.foot_names = tuple(foot_names)
        self.num_feet = len(foot_names)

        # Build ordered list of formula functions matching foot_names order.
        self._formulas = []
        for name in foot_names:
            matched = [key for key in self._LEG_FORMULAS if key in name]
            if len(matched) != 1:
                raise ValueError(
                    f"Cannot identify leg from foot name '{name}'. "
                    f"Expected exactly one of {list(self._LEG_FORMULAS.keys())} "
                    f"as substring, got {matched}."
                )
            self._formulas.append(self._LEG_FORMULAS[matched[0]])

    def __call__(self, cmd: "CommandManager") -> torch.Tensor:
        phase = getattr(cmd, self.phase_cmd)
        offset = getattr(cmd, self.offset_cmd)
        bound = getattr(cmd, self.bound_cmd)
        return torch.stack(
            [fn(phase, offset, bound) for fn in self._formulas], dim=1
        )


class DirectOffsets:
    """Reads per-foot phase offsets directly from named commands.

    Works with any number of feet.

    Args:
        command_names: One command name per foot, in foot order.
    """
    def __init__(self, command_names: tuple[str, ...]):
        self.command_names = command_names

    def __call__(self, cmd: "CommandManager") -> torch.Tensor:
        return torch.stack(
            [getattr(cmd, name) for name in self.command_names], dim=1
        )


# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────

@dataclass
class GaitManagerConfig:
    """Internal config for GaitManager. Built from GaitConfig by LocomotionEnv.

    Two modes:
        - "fixed":   No commands needed. Foot offsets are evenly spaced,
                     frequency and duration are constants from config.
        - "command": Frequency and duration are read from CommandManager.
                     Foot offsets are produced by ``foot_offset_provider``.
    """
    num_envs: int = 0
    foot_names: tuple[str, ...] | list[str] = field(default_factory=tuple)

    # "fixed" or "command"
    offset_mode: str = "fixed"

    # ── Fixed-mode settings ──
    gait_period: float = 0.8    # seconds per gait cycle (freq = 1/period)
    default_freq: float = 2.5   # Hz, used only if offset_mode == "fixed"
    default_duration: float = 0.5  # stance fraction [0, 1]

    # ── Command-mode settings ─
    freq_command: str = "gait_freq"
    duration_command: str = "gait_duration"

    # Callable: (CommandManager) -> [num_envs, num_feet] foot offsets.
    # Only used when offset_mode == "command".
    # Example providers: QuadrupedOffsets(), DirectOffsets(("off_0", "off_1", ...))
    foot_offset_provider: FootOffsetProvider | None = None

    # Von Mises smoothing sigma for desired_contact_states.
    # Smaller = smoother stance/swing transition.
    contact_smoothing_sigma: float = 0.07


class GaitManager(BaseManager):
    """Manages gait phase generation for legged locomotion.

    Two modes:
        ``"fixed"``
            Phase offsets are evenly distributed across feet.
            Frequency and duration are constants. No commands needed.

        ``"command"``
            Frequency and duration are read from CommandManager each step.
            Per-foot phase offsets are produced by a pluggable
            ``foot_offset_provider`` callable.

    Outputs (updated each ``advance()``):
        ``foot_phases``             [num_envs, num_feet]  Phase in [0, 1).
        ``clock_inputs``            [num_envs, num_feet]  sin(2*pi*warped).
        ``desired_contact_states``  [num_envs, num_feet]  Smooth stance prob.
    """

    def __init__(self, env: "World", config: GaitManagerConfig):
        super().__init__(env)
        self.config = config
        num_envs = config.num_envs

        self.foot_names = tuple(
            self.env.scene_manager.find_body_names(body_names=config.foot_names)
        )
        self.num_feet = len(self.foot_names)

        # ── Phase state ──
        self.gait_timer = torch.zeros(num_envs, device=self.device)
        self.foot_phases = torch.zeros(num_envs, self.num_feet, device=self.device)

        # ── Outputs ──
        self.clock_inputs = torch.zeros(num_envs, self.num_feet, device=self.device)
        self.desired_contact_states = torch.zeros(num_envs, self.num_feet, device=self.device)

        # ── Validate foot_offset_provider foot_names match ──
        if config.offset_mode == "command":
            provider = config.foot_offset_provider

            if hasattr(provider, "foot_names") and tuple(provider.foot_names) != self.foot_names:
                raise ValueError(
                    f"foot_offset_provider foot_names {provider.foot_names} does not match "
                    f"GaitManager foot_names {self.foot_names}. "
                    f"Use the same foot_names for both GaitConfig and QuadrupedOffsets."
                )

        # ── Fixed-mode precomputation ──
        if config.offset_mode == "fixed":
            self._fixed_offsets = torch.tensor(
                [i / self.num_feet for i in range(self.num_feet)],
                device=self.device,
            )

        # ── Von Mises (Normal CDF) for smooth contact targets ──
        self._smoothing_dist = torch.distributions.Normal(0.0, config.contact_smoothing_sigma)

    # ──────────────────────────────────────────────
    # Core update
    # ──────────────────────────────────────────────

    def advance(self) -> None:
        freq, duration, foot_offsets = self._read_gait_params()

        self.gait_timer = (self.gait_timer + self.env.control_dt * freq) % 1.0

        raw_foot_phases = (self.gait_timer.unsqueeze(1) + foot_offsets) % 1.0
        self.foot_phases = raw_foot_phases

        warped = self._apply_duration_warp(raw_foot_phases, duration)
        self.clock_inputs = torch.sin(2.0 * math.pi * warped)
        self.desired_contact_states = self._compute_desired_contacts(warped)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.gait_timer[env_ids] = 0.0
        self.foot_phases[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0

    # ────────────────────────────────────────────
    # Public queries
    # ────────────────────────────────────────────

    def get_swing_mask(self) -> torch.Tensor:
        """Boolean mask: True = swing, False = stance.  [num_envs, num_feet]."""
        return self.desired_contact_states < 0.5

    def get_phase_encoding(self) -> torch.Tensor:
        """Sin/cos encoding. [num_envs, num_feet * 2]: [sin0, cos0, sin1, cos1, ...]."""
        phi = 2.0 * math.pi * self.foot_phases
        return torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1).reshape(
            self.env.num_envs, -1
        )

    def get_swing_progress(self) -> torch.Tensor:
        """Normalized [0,1] progress within swing phase. -1 during stance.

        Returns:
            [num_envs, num_feet].
        """
        swing_mask = self.get_swing_mask()
        _, duration, _ = self._read_gait_params()
        phase_in_cycle = self.foot_phases % 1.0

        swing_progress = (phase_in_cycle - duration.unsqueeze(1)) / (
            1.0 - duration.unsqueeze(1) + 1e-8
        )
        swing_progress = swing_progress.clamp(0.0, 1.0)
        return torch.where(swing_mask, swing_progress, torch.full_like(swing_progress, -1.0))

    def get_target_foot_height(
        self,
        max_height: float,
        profile: str = "sine",
    ) -> torch.Tensor:
        """Target foot height during swing. 0 during stance.

        Args:
            max_height: Peak height (meters).
            profile: ``"sine"`` | ``"cosine"`` | ``"triangle"``.

        Returns:
            [num_envs, num_feet].
        """
        progress = self.get_swing_progress()
        is_swing = progress >= 0
        phi = progress.clamp(min=0.0, max=1.0)

        if profile == "sine":
            height_ratio = torch.sin(math.pi * phi)
        elif profile == "cosine":
            height_ratio = 0.5 * (1.0 - torch.cos(2.0 * math.pi * phi))
        elif profile == "triangle":
            height_ratio = 1.0 - torch.abs(1.0 - 2.0 * phi)
        else:
            raise ValueError(f"Unknown height profile: {profile}")

        target = max_height * height_ratio
        return torch.where(is_swing, target, torch.zeros_like(target))

    # ────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────

    def _read_gait_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (freq, duration, foot_offsets)."""
        cfg = self.config
        N = cfg.num_envs

        if cfg.offset_mode == "fixed":
            freq = torch.full((N,), 1.0 / cfg.gait_period, device=self.device)
            duration = torch.full((N,), cfg.default_duration, device=self.device)
            foot_offsets = self._fixed_offsets.unsqueeze(0).expand(N, -1)
            return freq, duration, foot_offsets

        # offset_mode == "command"
        cmd = self.env.command_manager
        freq = getattr(cmd, cfg.freq_command)
        duration = getattr(cmd, cfg.duration_command)
        foot_offsets = cfg.foot_offset_provider(cmd)
        return freq, duration, foot_offsets

    def _apply_duration_warp(
        self, phases: torch.Tensor, duration: torch.Tensor,
    ) -> torch.Tensor:
        """Remap so stance -> [0, 0.5), swing -> [0.5, 1)."""
        dur = duration.unsqueeze(1)
        p = phases % 1.0
        stance_mask = p < dur
        warped_stance = p * (0.5 / (dur + 1e-8))
        warped_swing = 0.5 + (p - dur) * (0.5 / (1.0 - dur + 1e-8))
        return torch.where(stance_mask, warped_stance, warped_swing)

    def _compute_desired_contacts(self, warped_phases: torch.Tensor) -> torch.Tensor:
        """Smooth desired contact: ~1 during stance, ~0 during swing."""
        cdf = self._smoothing_dist.cdf
        t = warped_phases
        return (
            cdf(t) * (1.0 - cdf(t - 0.5))
            + cdf(t - 1.0) * (1.0 - cdf(t - 1.5))
        )
