"""ActionTerm abstraction (IsaacLab / mjlab-style).

An action term is responsible for converting a slice of the raw policy
output into an **absolute joint position target** for a subset of the
robot's actuated joints. The :class:`ActionManagerBase` owns a dict of
terms, iterates over them during ``process_actions`` / ``apply_actions``,
and feeds the resulting targets through the configured actuator models
to compute torques that go to the simulator.

Terms differ in two places:

1. :meth:`ActionTerm.process_actions` — how raw policy output maps to
   an intermediate ``processed_actions`` buffer. Subclasses typically
   do ``processed = raw * scale + offset``.

2. :meth:`ActionTerm.compute_target_positions` — how the intermediate
   turns into an **absolute** joint position target the actuator sees.
   Absolute-mode terms return ``processed`` directly; relative-mode
   terms return ``current_joint_pos + processed``.

This abstraction mirrors IsaacLab's
``isaaclab/managers/action_manager.py`` and mjlab's
``mjlab/envs/mdp/actions/actions.py`` so we can port task configs
between frameworks with minimal translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.common.action import ActionManagerBase
    from rlworld.rl.envs.world import World


@dataclass
class ActionTermCfg:
    """Base config for an action term.

    Subclasses extend this with term-specific fields (scale, offset,
    etc.). The ``class_type`` field names the concrete
    :class:`ActionTerm` subclass that should be instantiated from this
    config; the action manager uses it to build the term at
    environment-construction time.

    Attributes:
        class_type: Concrete ActionTerm subclass to instantiate.
        joint_names: Regex patterns resolved against the robot's
            actuated joint name list. The matched joints form this
            term's joint subset. Default ``[".*"]`` = every actuated
            joint.
        clip: Optional ``(low, high)`` clip applied to the raw
            policy output before ``process_actions`` uses it.
            ``None`` = no clipping (policy output is passed through).
    """

    class_type: type[ActionTerm] | None = None
    joint_names: list[str] = field(default_factory=lambda: [".*"])
    clip: tuple[float, float] | None = None


class ActionTerm(ABC):
    """Abstract base for action terms.

    Each term owns a subset of the full actuated joint space. The
    manager orchestrates multiple terms (for now always single-term
    for our existing tasks) and owns the actuator-compute path.

    Subclasses must implement :meth:`process_actions` and
    :meth:`compute_target_positions`. The default :meth:`reset`
    zeroes the term's own buffers for the given envs; override for
    stateful terms that need extra reset behaviour.
    """

    __name__: str = "ActionTerm"

    def __init__(
        self,
        cfg: ActionTermCfg,
        env: World,
        manager: ActionManagerBase,
    ) -> None:
        self._cfg = cfg
        self._env = env
        self._manager = manager

        # Subclasses must populate these in their own __init__ after
        # calling super().__init__.
        self._joint_ids: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._raw_actions: torch.Tensor = torch.empty(0)
        self._processed_actions: torch.Tensor = torch.empty(0)

    # ── Properties ────────────────────────────────────────────────

    @property
    def cfg(self) -> ActionTermCfg:
        return self._cfg

    @property
    def env(self) -> World:
        return self._env

    @property
    def manager(self) -> ActionManagerBase:
        return self._manager

    @property
    def joint_ids(self) -> torch.Tensor:
        """Indices (into the manager's action_dim space) of this term's joints."""
        return self._joint_ids

    @property
    def action_dim(self) -> int:
        return int(self._joint_ids.numel())

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw action slice handed to this term in the last
        :meth:`process_actions` call. Shape ``(num_envs, action_dim)``.
        """
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Term-specific processed value. For absolute position terms
        this is the target itself; for relative/delta terms this is
        the delta to add to the current joint position.
        """
        return self._processed_actions

    # ── Abstract methods ──────────────────────────────────────────

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Consume the raw action slice for this term.

        Implementations must update ``self._raw_actions`` and
        ``self._processed_actions``. The input tensor has shape
        ``(num_envs, action_dim)`` where ``action_dim`` equals the
        number of joints this term owns.
        """

    @abstractmethod
    def compute_target_positions(self) -> torch.Tensor:
        """Return absolute joint position targets for this term's
        joints. Shape: ``(num_envs, action_dim)``.

        The manager calls this once per ``apply_actions`` to build
        the full-robot target tensor that gets routed through the
        actuator models.
        """

    # ── Hooks ─────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the term's buffers for the given envs.

        Override in subclasses that have additional stateful fields.
        """
        if env_ids is None or len(env_ids) == 0:
            return
        if self._raw_actions.numel() > 0:
            self._raw_actions[env_ids] = 0.0
        if self._processed_actions.numel() > 0:
            self._processed_actions[env_ids] = 0.0

    def advance(self) -> None:
        """Optional per-step hook called after every physics step.

        Default is a no-op. Override for terms that need to
        advance internal state (filter history, EMA state, etc.).
        """
