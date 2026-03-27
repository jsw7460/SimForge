"""Base class for actuator models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from .actuator_cfg import ActuatorBaseCfg


class ActuatorBase(ABC):
    """Base class for actuator models over a group of joints.

    An actuator model converts position targets (from the RL policy)
    together with the current joint state into torques that are applied
    directly to the simulation.

    Subclasses must implement :meth:`reset` and :meth:`compute`.
    """

    cfg: ActuatorBaseCfg

    computed_effort: torch.Tensor
    """Raw effort output before clipping.  Shape ``(num_envs, num_joints)``."""

    applied_effort: torch.Tensor
    """Effort after clipping to motor limits.  Shape ``(num_envs, num_joints)``."""

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        num_envs: int,
        num_joints: int,
        device: str,
    ) -> None:
        self.cfg = cfg
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._device = device

        # Effort limit tensor
        if cfg.effort_limit is not None:
            self.effort_limit = torch.full(
                (num_envs, num_joints), cfg.effort_limit, device=device
            )
        else:
            self.effort_limit = torch.full(
                (num_envs, num_joints), float("inf"), device=device
            )

        # Output buffers
        self.computed_effort = torch.zeros(num_envs, num_joints, device=device)
        self.applied_effort = torch.zeros(num_envs, num_joints, device=device)

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @abstractmethod
    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset internal state (e.g. delay buffers, LSTM hidden state).

        Args:
            env_ids: Indices of environments being reset.
        """
        ...

    @abstractmethod
    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Convert position targets to joint torques.

        Args:
            target_pos: Desired joint positions from the action manager,
                shape ``(num_envs, num_joints)``.
            joint_pos: Current joint positions,
                shape ``(num_envs, num_joints)``.
            joint_vel: Current joint velocities,
                shape ``(num_envs, num_joints)``.

        Returns:
            Applied torques after clipping, shape ``(num_envs, num_joints)``.
        """
        ...

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip torques to the configured effort limit."""
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
