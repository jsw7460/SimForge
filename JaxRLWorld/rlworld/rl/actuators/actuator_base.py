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
        joint_names: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._device = device
        self._joint_names = joint_names or []

        # Effort limit tensor — scalar, per-joint-regex dict, or None (no limit).
        self.effort_limit = self._resolve_per_joint_param(cfg.effort_limit, default=float("inf"))

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

    def _resolve_per_joint_param(self, value: float | dict[str, float] | None, default: float = 0.0) -> torch.Tensor:
        """Resolve a scalar or per-joint-regex dict into a (num_envs, num_joints) tensor."""
        from rlworld.rl.utils import string as string_utils

        tensor = torch.full((self._num_envs, self._num_joints), default, device=self._device)
        if value is None:
            return tensor
        if isinstance(value, (int, float)):
            tensor[:] = float(value)
            return tensor
        if isinstance(value, dict) and self._joint_names:
            indices, _, values = string_utils.resolve_matching_names_values(value, self._joint_names)
            tensor[:, indices] = torch.tensor(values, dtype=torch.float32, device=self._device)
            return tensor
        return tensor
