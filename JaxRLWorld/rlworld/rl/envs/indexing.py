"""Centralized articulation indexing for all simulators.

:class:`ArticulationIndexing` is the single source of truth for mapping
between the **canonical joint order** (defined by the action manager's
regex resolution) and each simulator's internal joint order.

Every scene manager builds an ``ArticulationIndexing`` after the scene
is constructed.  The action manager, actuator models, and RobotData
implementations all reference this object instead of maintaining their
own ad-hoc index tensors.

Index spaces
~~~~~~~~~~~~
- **canonical** (local): ``[0, 1, ..., num_joints-1]`` in the order the
  action manager resolves actuated joints.  This is the order of
  ``processed_actions``, ``joint_pos``, ``joint_vel``, and all
  observation/reward tensors.
- **sim**: The simulator's internal joint order, which may differ from
  canonical.  Each simulator stores joints in the order it parses the
  URDF/MJCF/USD file.

The two key tensors:

- ``sim_indices``: ``canonical[i]`` → simulator joint index.
  Used by ``_apply_position`` / ``_apply_force`` to scatter canonical-
  order data into the simulator's buffers.
- ``sim_to_canonical``: inverse mapping so that ``sim_data[:, sim_to_canonical]``
  yields canonical order.  Used by RobotData to return joint_pos/vel
  in canonical order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ArticulationIndexing:
    """Maps between canonical (action manager) and simulator joint orders.

    Created once by each scene manager after the scene is built.
    Shared by the action manager, actuator models, and RobotData.
    """

    # -- Names ----------------------------------------------------------------
    joint_names: tuple[str, ...]
    """Joint names in canonical order (action manager resolution order).
    These are bare names without simulator prefixes."""

    # -- Canonical → simulator ------------------------------------------------
    sim_indices: Tensor
    """Shape ``(num_joints,)``.  ``sim_indices[i]`` is the simulator-internal
    index for canonical joint ``i``.  Used to scatter actions/torques into
    the simulator's buffers."""

    # -- Simulator → canonical ------------------------------------------------
    sim_to_canonical: Tensor
    """Shape ``(num_joints,)``.  Indexes simulator-order data to produce
    canonical order: ``sim_data[:, sim_to_canonical]`` → canonical order.
    Used by RobotData to return joint_pos/vel in canonical order."""

    # -- Joint limits (canonical order) ---------------------------------------
    joint_limits_lower: Tensor
    """Lower joint position limits in canonical order.  Shape ``(num_joints,)``."""

    joint_limits_upper: Tensor
    """Upper joint position limits in canonical order.  Shape ``(num_joints,)``."""

    # -- Newton-specific (optional) -------------------------------------------
    newton_q_indices: Tensor | None = None
    """Newton ``joint_q`` array indices for actuated joints.  Shape ``(num_joints,)``.
    Only set for Newton backend; None for Genesis/MuJoCo."""

    newton_qd_indices: Tensor | None = None
    """Newton ``joint_qd`` array indices for actuated joints.  Shape ``(num_joints,)``.
    Only set for Newton backend; None for Genesis/MuJoCo."""

    # -- Derived properties ---------------------------------------------------

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def device(self) -> torch.device:
        return self.sim_indices.device

    def __repr__(self) -> str:
        return (
            f"ArticulationIndexing(num_joints={self.num_joints}, "
            f"device={self.device}, "
            f"joints={list(self.joint_names[:3])}{'...' if self.num_joints > 3 else ''})"
        )

    def print_mapping(self) -> None:
        """Print canonical ↔ simulator joint mapping for debugging."""
        print(f"\n{'=' * 60}")
        print(f"  ArticulationIndexing ({self.num_joints} joints)")
        print(f"{'=' * 60}")
        print(f"  {'Canon':<6} {'Joint Name':<35} {'→ Sim Idx':<10}")
        print(f"  {'-' * 6} {'-' * 35} {'-' * 10}")
        for i, name in enumerate(self.joint_names):
            sim_idx = self.sim_indices[i].item()
            print(f"  {i:<6} {name:<35} → {sim_idx:<10}")
        print(f"{'=' * 60}\n")
