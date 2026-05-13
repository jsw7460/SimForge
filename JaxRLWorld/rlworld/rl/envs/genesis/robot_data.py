"""GenesisRobotData — thin wrapper that satisfies the RobotData protocol.

Lazily computes each property from the Genesis ``RigidEntity`` API.
Genesis uses **wxyz** quaternions natively, so no reordering is needed.
Genesis velocities are in **world frame**, so we rotate to body frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.misc import qd_to_torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_rotate_wxyz, quat_to_euler_wxyz

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
        default_joint_pos: Tensor | None = None,
    ) -> None:
        self._entity = entity
        self._actuated_dof_ids = actuated_dof_ids
        self._gravity_vec: Tensor | None = None
        self._num_envs = num_envs
        self._device = device
        self._default_joint_pos = default_joint_pos
        # Global solver link indices for this entity, in local order — the
        # exact list ``RigidEntity.get_links_pos(None)`` uses internally
        # (``_get_global_idx(None, n_links, link_start)``).  Needed to query
        # the ``ref="link_com"`` variant via the solver (RigidEntity's own
        # get_links_* don't expose ``ref``).
        self._global_link_ids = list(range(entity._link_start, entity._link_start + entity.n_links))

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = (
                torch.tensor(
                    [[0.0, 0.0, -1.0]],
                    device=self._device,
                    dtype=torch.float32,
                )
                .expand(self._num_envs, -1)
                .contiguous()
            )
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

    # ── Root center-of-mass variants ─────────────────────────────────
    # Genesis' solver computes link quantities at a chosen reference point.
    # ``RigidEntity.get_pos`` / ``get_vel`` (used above) call the solver with
    # the default ``ref="link_origin"``.  For the CoM we go straight to the
    # solver with ``ref="link_com"`` (RigidEntity's wrappers don't pass ``ref``).
    #   get_links_pos(ref="link_com") = links_state.i_pos + links_state.root_COM   (link CoM, world)
    #   get_links_vel(ref="link_com") = cd_vel + cd_ang x links_state.i_pos        (vel at link CoM, world)

    @property
    def root_com_pos_w(self) -> Tensor:
        return self._entity._solver.get_links_pos(self._entity.base_link_idx, ref="link_com")[..., 0, :]

    @property
    def root_com_lin_vel_w(self) -> Tensor:
        return self._entity._solver.get_links_vel(self._entity.base_link_idx, ref="link_com")[..., 0, :]

    @property
    def root_com_lin_vel_b(self) -> Tensor:
        # Expressed in the *link* frame (same orientation reference as
        # root_link_lin_vel_b — the body has one orientation).
        return quat_rotate_inverse_wxyz(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def projected_gravity_b(self) -> Tensor:
        quat = self.root_link_quat_w
        return quat_rotate_inverse_wxyz(quat, self._get_gravity_vec())

    @property
    def heading_w(self) -> Tensor:
        euler = quat_to_euler_wxyz(self.root_link_quat_w)
        return euler[:, 2]

    @property
    def default_joint_pos(self) -> Tensor:
        return self._default_joint_pos

    @property
    def joint_pos(self) -> Tensor:
        return self._entity.get_dofs_position(self._actuated_dof_ids)

    @property
    def joint_vel(self) -> Tensor:
        return self._entity.get_dofs_velocity(self._actuated_dof_ids)

    @property
    def applied_torque(self) -> Tensor:
        """Per-DOF actuator control force for actuated joints.

        Calls Genesis's ``get_dofs_control_force`` which runs the PD-law
        kernel (``kernel_get_dofs_control_force`` in
        ``Genesis/.../rigid/abd/accessor.py``) and returns the torque
        clipped to the joint's ``force_range`` — the analog of
        MuJoCo's ``qfrc_actuator``. This is distinct from
        ``get_dofs_force``, which returns the net joint-space force
        including passive damping and gravity bias.
        """
        return self._entity.get_dofs_control_force(dofs_idx_local=self._actuated_dof_ids)

    @property
    def joint_pos_limits(self) -> tuple[Tensor, Tensor]:
        """Hard joint position limits in canonical actuated order.

        Calls Genesis's ``entity.get_dofs_limit(actuated_dof_ids)``. Per
        the Genesis docstring, the return shape is *either*
        ``(n_dofs,)`` or ``(n_envs, n_dofs)`` depending on whether the
        scene is batched. We normalise both cases to the 1-D
        ``(n_dofs,)`` shape the RobotData protocol promises — for the
        batched case we take the first env's row since joint limits are
        constant across envs in all current Genesis configs.

        Returns:
            ``(lower, upper)``, each shape ``(num_actuated_joints,)``.
        """
        lower, upper = self._entity.get_dofs_limit(dofs_idx_local=self._actuated_dof_ids)
        if lower.ndim == 2:
            lower = lower[0]
            upper = upper[0]
        return lower, upper

    @property
    def soft_joint_pos_limits(self) -> tuple[Tensor, Tensor]:
        """Soft joint position limits (hard * 0.9) in actuated order.

        Genesis only stores hard limits via ``get_dofs_limit``; the
        soft flavour is hard × 0.9 to match mjlab's default
        ``soft_joint_pos_limit_factor=0.9``. Returns a tuple of
        ``(num_joints,)`` tensors.
        """
        lo, hi = self.joint_pos_limits
        return lo * 0.9, hi * 0.9

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

    @property
    def body_com_pos_w_all(self) -> Tensor:
        """World-frame positions of all links' centers of mass. Shape ``(num_envs, num_links, 3)``.

        Same link order as ``body_pos_w_all`` (``_global_link_ids`` is the
        same index list ``RigidEntity.get_links_pos(None)`` uses).
        """
        return self._entity._solver.get_links_pos(self._global_link_ids, ref="link_com")

    @property
    def body_com_lin_vel_w_all(self) -> Tensor:
        """World-frame linear velocities of all links at their centers of mass. Shape ``(num_envs, num_links, 3)``."""
        return self._entity._solver.get_links_vel(self._global_link_ids, ref="link_com")

    # ------------------------------------------------------------------
    # Per-name body/site reads
    # ------------------------------------------------------------------

    def _resolve_link_indices(self, names: list[str]) -> list[int]:
        """Resolve link names to local link indices, preserving input order."""
        return [self._entity.get_link(name=n).idx_local for n in names]

    def body_pos_w(self, names: list[str]) -> Tensor:
        idxs = self._resolve_link_indices(list(names))
        return self._entity.get_links_pos(links_idx_local=idxs)

    def body_lin_vel_w(self, names: list[str]) -> Tensor:
        idxs = self._resolve_link_indices(list(names))
        return self._entity.get_links_vel(links_idx_local=idxs)

    def site_pos_w(self, names: list[str]) -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_pos_w. Genesis has "
            "no equivalent of MuJoCo sites — use body_pos_w with link names."
        )

    def site_lin_vel_w(self, names: list[str]) -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_lin_vel_w. Genesis has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w with link names."
        )

    def body_pos_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_pos_w_all[:, body_ids, :]

    def body_lin_vel_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_lin_vel_w_all[:, body_ids, :]

    def site_pos_w_by_ids(self, site_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_pos_w_by_ids. Genesis has "
            "no equivalent of MuJoCo sites — use body_pos_w_by_ids with link ids."
        )

    def site_lin_vel_w_by_ids(self, site_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            "GenesisRobotData does not implement site_lin_vel_w_by_ids. Genesis has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w_by_ids with link ids."
        )

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def angular_momentum_w(self, sensor_name: str | None = None) -> Tensor:
        """Whole-body angular momentum (world frame) about the system CoM.

        Matches Newton and MuJoCo's ``subtreeangmom`` via König's
        decomposition:

            L = sum_i [ m_i * (r_i - r_c) x v_i              # orbital
                        +  R_i @ I_i_local @ R_i^T @ omega_i ]  # spin

        Reads ``solver.links_info.inertial_mass`` / ``inertial_i`` for
        per-body model values and the existing
        ``body_com_pos_w_all`` / ``body_com_lin_vel_w_all`` /
        ``body_ang_vel_w_all`` / ``body_quat_w_all`` accessors for state.
        ``sensor_name`` is ignored.
        """
        solver = self._entity._solver
        link_ids = self._global_link_ids

        # Per-body mass: (W, B) if batch_links_info else (B,) → broadcast to (W, B).
        m = qd_to_torch(solver.links_info.inertial_mass, None, link_ids, transpose=True, copy=True)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(self._num_envs, -1)
        # Per-body local inertia 3x3: (W, B, 3, 3) if batched else (B, 3, 3) → broadcast.
        I_body = qd_to_torch(solver.links_info.inertial_i, None, link_ids, transpose=True, copy=True)
        if I_body.dim() == 3:
            I_body = I_body.unsqueeze(0).expand(self._num_envs, -1, -1, -1)

        # Per-body state.
        r_i = self.body_com_pos_w_all  # (W, B, 3)
        v_i = self.body_com_lin_vel_w_all  # (W, B, 3)
        omega_i = self.body_ang_vel_w_all  # (W, B, 3)
        q_i = self.body_quat_w_all  # (W, B, 4) wxyz

        # Spin: sum_i R_i @ I_i_local @ R_i^T @ omega_i.
        omega_body = quat_rotate_inverse_wxyz(q_i, omega_i)
        spin_body = torch.einsum("nbij,nbj->nbi", I_body, omega_body)
        spin = quat_rotate_wxyz(q_i, spin_body).sum(dim=1)  # (W, 3)

        # Orbital: sum_i m_i * (r_i - r_c) x v_i.
        total_mass = m.sum(dim=1)  # (W,)
        r_c = (m.unsqueeze(-1) * r_i).sum(dim=1) / total_mass.unsqueeze(-1)  # (W, 3)
        r_rel = r_i - r_c.unsqueeze(1)  # (W, B, 3)
        orbital = (m.unsqueeze(-1) * torch.cross(r_rel, v_i, dim=-1)).sum(dim=1)  # (W, 3)

        return spin + orbital
