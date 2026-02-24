from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class GaitManagerConfig:
    num_envs: int
    gait_period: float  # Full gait cycle duration (seconds). One complete cycle = all feet swing once.
    foot_names: tuple[str, ...] | list[str]


class GaitManager(BaseManager):
    """Manages gait pattern generation for legged locomotion.

    Generates alternating swing/stance patterns for each foot based on a periodic
    phase clock. Each foot swings once per gait_period, with evenly distributed
    phase offsets between feet.

    For bipedal (gait_period=0.8s):
        - Left foot: swing at t=0.0-0.4s, stance at t=0.4-0.8s
        - Right foot: swing at t=0.4-0.8s, stance at t=0.0-0.4s
        - Each foot swings for 50% of the period (phase_width = π)

    For quadrupedal (gait_period=1.0s):
        - Each foot swings for 25% of the period (phase_width = π/2)
        - Phase offsets: LF=0, RF=-π/2, LH=-π, RH=-3π/2
    """

    def __init__(self, env: "World", config: GaitManagerConfig):
        super().__init__(env)
        self.config = config

        self.foot_names = tuple(self.env.scene_manager.find_body_names(body_names=config.foot_names))

        # self.foot_names = config.foot_names
        self.num_feet = len(self.foot_names)

        self.gait_period = config.gait_period
        self.phase_width = 2 * torch.pi / self.num_feet

        self.phase_offsets = torch.tensor(
            [-2 * torch.pi * i / self.num_feet for i in range(self.num_feet)],
            device=self.device
        )

        self.gait_timer = torch.zeros(config.num_envs, device=self.device)

    def advance(self) -> None:
        self.gait_timer += self.env.control_dt

    def get_swing_mask(self) -> torch.Tensor:
        """Get boolean mask indicating which feet are in swing phase.

        A foot is in swing phase when its phase angle φ is in [0, phase_width).
        φ = (2π * timer / period) + offset, wrapped to [0, 2π)

        Returns:
            Boolean tensor of shape (num_envs, num_feet).
            True = swing phase, False = stance phase.
        """
        timer_expanded = self.gait_timer.unsqueeze(1)
        offsets_expanded = self.phase_offsets.unsqueeze(0)

        phi = 2 * torch.pi * (timer_expanded / self.gait_period) + offsets_expanded
        phi = phi % (2 * torch.pi)

        return (0 <= phi) & (phi < self.phase_width)

    def get_phase_encoding(self) -> torch.Tensor:
        """Get sin/cos encoding of each foot's phase for observation.

        Returns:
            Tensor of shape (num_envs, num_feet * 2).
            Format: [cos(φ_0), sin(φ_0), cos(φ_1), sin(φ_1), ...]
        """
        timer_expanded = self.gait_timer.unsqueeze(1)
        offsets_expanded = self.phase_offsets.unsqueeze(0)

        phi = 2 * torch.pi * (timer_expanded / self.gait_period) + offsets_expanded

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        encoding = torch.stack([cos_phi, sin_phi], dim=-1)
        return encoding.reshape(self.env.num_envs, -1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset gait timer for specified environments."""
        self.gait_timer[env_ids] = 0.0

    def get_swing_progress(self) -> torch.Tensor:
        """Get normalized progress within swing phase for each foot.

        Returns:
            Tensor of shape (num_envs, num_feet).
            Value in [0, 1] during swing phase, -1 during stance phase.
            0 = swing start, 1 = swing end.
        """
        timer_expanded = self.gait_timer.unsqueeze(1)
        offsets_expanded = self.phase_offsets.unsqueeze(0)

        phi = 2 * torch.pi * (timer_expanded / self.gait_period) + offsets_expanded
        phi = phi % (2 * torch.pi)

        is_swing = (0 <= phi) & (phi < self.phase_width)

        # Normalize phi to [0, 1] within swing phase
        progress = phi / self.phase_width

        # Mark stance phase as -1
        progress = torch.where(is_swing, progress, torch.full_like(progress, -1.0))

        return progress

    def get_target_foot_height(
        self,
        max_height: float,
        profile: str = "sine",
    ) -> torch.Tensor:
        """Get target foot height based on swing phase progress.

        Args:
            max_height: Peak foot height during swing (meters).
            profile: Height profile function.
                - "sine": sin(π × φ), smooth lift and lower
                - "cosine": 0.5 × (1 - cos(2π × φ)), slower at peak

        Returns:
            Tensor of shape (num_envs, num_feet).
            Target height for each foot. 0 during stance phase.
        """
        progress = self.get_swing_progress()  # (num_envs, num_feet)
        is_swing = progress >= 0

        # Clamp for safety (stance marked as -1)
        phi = progress.clamp(min=0.0, max=1.0)

        if profile == "sine":
            # 0 -> 1 -> 0, symmetric
            height_ratio = torch.sin(torch.pi * phi)
        elif profile == "cosine":
            # 0 -> 1 -> 0, slower at peak
            height_ratio = 0.5 * (1 - torch.cos(2 * torch.pi * phi))
        else:
            raise ValueError(f"Unknown profile: {profile}")

        target_height = max_height * height_ratio

        # Zero during stance
        target_height = torch.where(is_swing, target_height, torch.zeros_like(target_height))

        return target_height