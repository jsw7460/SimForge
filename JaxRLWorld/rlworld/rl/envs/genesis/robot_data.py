"""GenesisRobotData — thin wrapper that satisfies the RobotData protocol.

Lazily computes each property from the Genesis ``RigidEntity`` API.
Genesis uses **wxyz** quaternions natively, so no reordering is needed.
Genesis velocities are in **world frame**, so we rotate to body frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class GenesisRobotData:
    """RobotData implementation for a Genesis RigidEntity."""

    def __init__(
        self,
        entity: RigidEntity,
        actuated_dof_ids: Tensor | list[int],
        num_envs: int,
        device: torch.device,
    ) -> None:
        self._entity = entity
        self._actuated_dof_ids = actuated_dof_ids
        self._gravity_vec: Tensor | None = None
        self._num_envs = num_envs
        self._device = device

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._device,
                dtype=torch.float32,
            ).expand(self._num_envs, -1).contiguous()
        return self._gravity_vec

    @property
    def root_link_pos_w(self) -> Tensor:
        return self._entity.get_pos()

    @property
    def root_link_quat_w(self) -> Tensor:
        return self._entity.get_quat()  # already wxyz

    @property
    def root_link_lin_vel_w(self) -> Tensor:
        return self._entity.get_vel()

    @property
    def root_link_ang_vel_w(self) -> Tensor:
        return self._entity.get_ang()

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        return quat_rotate_inverse_wxyz(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        return quat_rotate_inverse_wxyz(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def projected_gravity_b(self) -> Tensor:
        quat = self.root_link_quat_w
        return quat_rotate_inverse_wxyz(quat, self._get_gravity_vec())

    @property
    def heading_w(self) -> Tensor:
        euler = quat_to_euler_wxyz(self.root_link_quat_w)
        return euler[:, 2]

    @property
    def joint_pos(self) -> Tensor:
        return self._entity.get_dofs_position(self._actuated_dof_ids)

    @property
    def joint_vel(self) -> Tensor:
        return self._entity.get_dofs_velocity(self._actuated_dof_ids)

    @property
    def joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Hard joint position limits in canonical actuated order.

        Calls Genesis's ``entity.get_dofs_limit(actuated_dof_ids)`` which
        returns ``(lower, upper)`` each of shape ``(1, num_joints)``. We
        squeeze the leading dim so the result matches the protocol shape
        ``(num_joints,)`` consistent with Newton's implementation.

        Returns:
            ``(lower, upper)``, each shape ``(num_actuated_joints,)``.
        """
        lower, upper = self._entity.get_dofs_limit(dofs_idx_local=self._actuated_dof_ids)
        # Genesis returns shape (1, N); squeeze to (N,)
        return lower.squeeze(0), upper.squeeze(0)

    # ------------------------------------------------------------------
    # Body-level reads
    # ------------------------------------------------------------------

    def find_body_index(self, body_name: str) -> int:
        """Resolve a link name to Genesis's local link index."""
        link = self._entity.get_link(name=body_name)
        return link.idx_local

    def body_ang_vel_w(self, body_index: int) -> Tensor:
        """World-frame angular velocity of a single link.

        Thin wrapper around :attr:`body_ang_vel_w_all` that selects one
        body from the batched view. Kept for backward compatibility with
        Phase D-2 callers.
        """
        return self.body_ang_vel_w_all[:, body_index, :]

    # ------------------------------------------------------------------
    # Batched per-body reads
    # ------------------------------------------------------------------

    @property
    def body_pos_w_all(self) -> Tensor:
        """World-frame positions of all links. Shape ``(num_envs, num_links, 3)``."""
        return self._entity.get_links_pos()

    @property
    def body_quat_w_all(self) -> Tensor:
        """World-frame orientations of all links, wxyz. Shape ``(num_envs, num_links, 4)``.

        Genesis uses wxyz natively — no reordering needed.
        """
        return self._entity.get_links_quat()

    @property
    def body_lin_vel_w_all(self) -> Tensor:
        """World-frame linear velocities of all links. Shape ``(num_envs, num_links, 3)``.

        Uses Genesis's default reference point (link origin for
        ``RigidEntity``).
        """
        return self._entity.get_links_vel()

    @property
    def body_ang_vel_w_all(self) -> Tensor:
        """World-frame angular velocities of all links. Shape ``(num_envs, num_links, 3)``."""
        return self._entity.get_links_ang()

    # ------------------------------------------------------------------
    # Per-name body/site reads
    # ------------------------------------------------------------------

    def _resolve_link_indices(self, names: "list[str]") -> "list[int]":
        """Resolve link names to local link indices, preserving input order."""
        return [self._entity.get_link(name=n).idx_local for n in names]

    def body_pos_w(self, names: "list[str]") -> Tensor:
        idxs = self._resolve_link_indices(list(names))
        return self._entity.get_links_pos(links_idx_local=idxs)

    def body_lin_vel_w(self, names: "list[str]") -> Tensor:
        idxs = self._resolve_link_indices(list(names))
        return self._entity.get_links_vel(links_idx_local=idxs)

    def site_pos_w(self, names: "list[str]") -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_pos_w. Genesis has "
            "no equivalent of MuJoCo sites — use body_pos_w with link names."
        )

    def site_lin_vel_w(self, names: "list[str]") -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_lin_vel_w. Genesis has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w with link names."
        )

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def angular_momentum_w(self, sensor_name: str | None = None) -> Tensor:
        """Whole-body angular momentum — not implemented for Genesis.

        Genesis does not expose a batched per-link inertia tensor (only
        scalar mass via ``get_links_inertial_mass``), and there is no
        active reward in JaxRLWorld that uses Genesis angular momentum.
        Implementing this would require either reading the static
        ``link.inertial_i`` per link and rotating to the body frame
        manually, or adding a Genesis sensor abstraction. Defer until a
        consumer actually needs it.
        """
        raise NotImplementedError(
            "GenesisRobotData.angular_momentum_w is not implemented. "
            "Genesis has no batched per-link inertia accessor and no "
            "JaxRLWorld preset currently uses angular_momentum_penalty "
            "with Genesis. If you need this, either implement the manual "
            "I @ omega path using static link.inertial_i values or add a "
            "Genesis sensor wrapper."
        )
