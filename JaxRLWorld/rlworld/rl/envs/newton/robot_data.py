"""NewtonRobotData — ArticulationView-backed state accessor and RobotData protocol.

Provides both read-only properties (RobotData protocol) and write methods
for reset/event terms. Wraps Newton's ArticulationView with the
count_per_world=1 dimension squeezed out, and handles xyzw ↔ wxyz
quaternion conversion.

Newton uses **xyzw** quaternions; the protocol uses **wxyz**.
Newton velocities are in **world frame**; body-frame properties rotate them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from newton.selection import ArticulationView
    from newton import State
    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRobotData:
    """RobotData implementation + state write API for Newton environments."""

    def __init__(
        self,
        env: "NewtonEnv",
        view: "ArticulationView",
        default_joint_pos: Tensor | None = None,
    ) -> None:
        self._env = env
        self._view = view
        self._gravity_vec: Tensor | None = None
        self._default_joint_pos = default_joint_pos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _state(self) -> "State":
        return self._env.scene_manager.state

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._env.device,
                dtype=torch.float32,
            ).expand(self._env.num_envs, -1).contiguous()
        return self._gravity_vec

    def _root_transform_floats(self, state: "State") -> Tensor:
        """(W, 7) float tensor from root transforms."""
        wp_arr = self._view.get_root_transforms(state)
        return wp.to_torch(wp_arr).reshape(-1, 7)

    def _root_velocity_floats(self, state: "State") -> Tensor:
        """(W, 6) float tensor from root velocities."""
        wp_arr = self._view.get_root_velocities(state)
        return wp.to_torch(wp_arr).reshape(-1, 6)

    # ------------------------------------------------------------------
    # Read: raw state (used by observations, event terms, etc.)
    # ------------------------------------------------------------------

    def dof_positions(self, state: "State") -> Tensor:
        """Joint coordinate positions. Shape: (W, joint_coord_count)."""
        return wp.to_torch(self._view.get_dof_positions(state)).squeeze(1)

    def dof_velocities(self, state: "State") -> Tensor:
        """Joint coordinate velocities. Shape: (W, joint_dof_count)."""
        return wp.to_torch(self._view.get_dof_velocities(state)).squeeze(1)

    def root_pos_w(self, state: "State") -> Tensor:
        """Root position in world frame. Shape: (W, 3)."""
        return self._root_transform_floats(state)[:, 0:3]

    def root_quat_wxyz(self, state: "State") -> Tensor:
        """Root quaternion in wxyz convention. Shape: (W, 4)."""
        xyzw = self._root_transform_floats(state)[:, 3:7]
        return xyzw[:, [3, 0, 1, 2]]

    def root_quat_xyzw(self, state: "State") -> Tensor:
        """Root quaternion in xyzw convention (Newton native). Shape: (W, 4)."""
        return self._root_transform_floats(state)[:, 3:7]

    def root_lin_vel_w(self, state: "State") -> Tensor:
        """Root linear velocity in world frame. Shape: (W, 3)."""
        return self._root_velocity_floats(state)[:, 0:3]

    def root_ang_vel_w(self, state: "State") -> Tensor:
        """Root angular velocity in world frame. Shape: (W, 3)."""
        return self._root_velocity_floats(state)[:, 3:6]

    # ------------------------------------------------------------------
    # Read: RobotData protocol (body-frame, wxyz)
    # ------------------------------------------------------------------

    @property
    def root_link_pos_w(self) -> Tensor:
        return self.root_pos_w(self._state)

    @property
    def root_link_quat_w(self) -> Tensor:
        """Quaternion in wxyz."""
        return self.root_quat_wxyz(self._state)

    @property
    def root_link_lin_vel_w(self) -> Tensor:
        return self.root_lin_vel_w(self._state)

    @property
    def root_link_ang_vel_w(self) -> Tensor:
        return self.root_ang_vel_w(self._state)

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        quat_wxyz = self.root_quat_wxyz(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        quat_wxyz = self.root_quat_wxyz(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, self.root_link_ang_vel_w)

    @property
    def projected_gravity_b(self) -> Tensor:
        return quat_rotate_inverse_wxyz(self.root_link_quat_w, self._get_gravity_vec())

    @property
    def heading_w(self) -> Tensor:
        return quat_to_euler_wxyz(self.root_link_quat_w)[:, 2]

    @property
    def default_joint_pos(self) -> Tensor:
        return self._default_joint_pos

    @property
    def joint_pos(self) -> Tensor:
        dof_pos = self.dof_positions(self._state)
        return dof_pos[:, self._env.act_manager.indexing.newton_q_indices]

    @property
    def joint_vel(self) -> Tensor:
        dof_vel = self.dof_velocities(self._state)
        return dof_vel[:, self._env.act_manager.indexing.newton_qd_indices]

    @property
    def joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Hard joint position limits in canonical actuated order.

        Reads ``model.joint_limit_lower`` / ``joint_limit_upper`` (which
        are flattened across worlds), takes the first world's slice, and
        indexes by ``newton_qd_indices`` to select actuated DOFs in the
        same order as ``joint_pos`` / ``joint_vel``.

        Returns:
            ``(lower, upper)``, each shape ``(num_actuated_joints,)``.
        """
        model = self._env.scene_manager.model
        dofs_per_world = model.joint_dof_count // model.world_count
        lower_all = wp.to_torch(model.joint_limit_lower)[:dofs_per_world]
        upper_all = wp.to_torch(model.joint_limit_upper)[:dofs_per_world]
        qd_indices = self._env.act_manager.indexing.newton_qd_indices
        return lower_all[qd_indices], upper_all[qd_indices]

    @property
    def soft_joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Soft joint position limits (hard * 0.9).

        Newton only stores hard limits; the soft flavour is hard ×
        ``soft_limit_factor`` where the factor is hardcoded to 0.9 to
        match mjlab's default ``soft_joint_pos_limit_factor=0.9``.
        Returned as a tuple of ``(num_joints,)`` tensors in actuated
        order, same shape as :attr:`joint_pos_limits`.
        """
        lo, hi = self.joint_pos_limits
        return lo * 0.9, hi * 0.9

    # ------------------------------------------------------------------
    # Body-level reads
    # ------------------------------------------------------------------

    def find_body_index(self, body_name: str) -> int:
        """Resolve a body name to its per-env body index in Newton.

        Uses the singleton ``NewtonBodyCache`` which builds a name→index
        map at first access. For Newton, body names typically include the
        entity prefix (e.g. ``"g1_29dof/torso_link"``).
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache
        cache = get_cache(self._env)
        indices = cache.get_body_indices(body_name)
        if not indices:
            raise ValueError(
                f"Body name {body_name!r} not found in Newton model. "
                f"Available bodies: {cache.body_names}"
            )
        return indices[0]

    def body_ang_vel_w(self, body_index: int) -> Tensor:
        """World-frame angular velocity of a single body.

        Thin wrapper around :attr:`body_ang_vel_w_all` that selects one
        body from the batched view. Kept for backward compatibility with
        Phase D-2 callers.
        """
        return self.body_ang_vel_w_all[:, body_index, :]

    # ------------------------------------------------------------------
    # Batched per-body reads
    # ------------------------------------------------------------------

    def _body_q_view(self) -> Tensor:
        """Helper: state.body_q reshaped to (num_envs, bodies_per_env, 7).

        Newton stores body_q as a flat ``wp.array[wp.transform]`` across
        all worlds. The standard JaxRLWorld setup replicates the same
        body layout per world, so a simple ``view(...)`` is correct.
        Each transform is ``(pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w)``
        — note Newton's native quaternion is **xyzw**.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache
        cache = get_cache(self._env)
        state = self._env.scene_manager.state
        return wp.to_torch(state.body_q).view(self._env.num_envs, cache.bodies_per_env, 7)

    def _body_qd_view(self) -> Tensor:
        """Helper: state.body_qd reshaped to (num_envs, bodies_per_env, 6).

        Newton stores body_qd as flat ``wp.array[wp.spatial_vector]``.
        Each spatial_vector is ``(lin.x, lin.y, lin.z, ang.x, ang.y, ang.z)``
        — linear velocity first, then angular, both world frame.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache
        cache = get_cache(self._env)
        state = self._env.scene_manager.state
        return wp.to_torch(state.body_qd).view(self._env.num_envs, cache.bodies_per_env, 6)

    @property
    def body_pos_w_all(self) -> Tensor:
        """World-frame positions of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._body_q_view()[:, :, 0:3]

    @property
    def body_quat_w_all(self) -> Tensor:
        """World-frame orientations of all bodies, wxyz. Shape ``(num_envs, num_bodies, 4)``.

        Newton stores quaternions as xyzw natively (positions 3..7 of
        the transform). We reorder to wxyz canonical via index gather.
        """
        body_q = self._body_q_view()
        quat_xyzw = body_q[:, :, 3:7]
        # xyzw -> wxyz
        return quat_xyzw[..., [3, 0, 1, 2]]

    @property
    def body_lin_vel_w_all(self) -> Tensor:
        """World-frame linear velocities of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._body_qd_view()[:, :, 0:3]

    @property
    def body_ang_vel_w_all(self) -> Tensor:
        """World-frame angular velocities of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._body_qd_view()[:, :, 3:6]

    # ------------------------------------------------------------------
    # Per-name body/site reads
    # ------------------------------------------------------------------

    def _resolve_body_indices(self, names: "list[str]") -> "list[int]":
        """Resolve a list of body names to per-env body indices.

        Uses the singleton ``NewtonBodyCache``. Names should be the
        Newton-prefixed body names (e.g. ``"go2_description/FL_foot"``)
        and must each match exactly one body. Returned indices preserve
        the input order so the resulting tensor columns line up with
        ``names``.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(self._env)
        out: list[int] = []
        for name in names:
            indices = cache.get_body_indices(name)
            if not indices:
                raise ValueError(
                    f"Body name {name!r} not found in Newton model. "
                    f"Available bodies: {cache.body_names}"
                )
            out.append(indices[0])
        return out

    def body_pos_w(self, names: "list[str]") -> Tensor:
        idxs = self._resolve_body_indices(list(names))
        return self.body_pos_w_all[:, idxs, :]

    def body_lin_vel_w(self, names: "list[str]") -> Tensor:
        idxs = self._resolve_body_indices(list(names))
        return self.body_lin_vel_w_all[:, idxs, :]

    def site_pos_w(self, names: "list[str]") -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_pos_w. Newton has "
            "no equivalent of MuJoCo sites — use body_pos_w with body names."
        )

    def site_lin_vel_w(self, names: "list[str]") -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_lin_vel_w. Newton has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w with body names."
        )

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def angular_momentum_w(self, sensor_name: str | None = None) -> Tensor:
        """Whole-body angular momentum (world frame) via manual ``sum_i I_i @ omega_i``.

        Reads ``model.body_inertia`` (per-body 3x3 in body-local frame)
        and the current state's per-body world quaternion + angular
        velocity. For each body:

            omega_body = quat_inverse_rotate(quat_world, omega_world)
            L_body = I_body @ omega_body
            L_world = quat_rotate(quat_world, L_body)

        then sums L_world across all bodies.

        The body-frame quat rotation uses Newton's native xyzw helper to
        be bit-identical to the legacy ``angular_momentum_penalty`` reward
        in ``mdp/rewards/newton/mjlab_rewards.py``. ``sensor_name`` is
        ignored (Newton has no built-in angular momentum sensor).
        """
        from rlworld.rl.envs.mdp.observations.newton.state import (
            _quat_rotate_inverse,
            _quat_rotate,
        )
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(self._env)
        num_envs = self._env.num_envs

        # Reshape model.body_inertia (flat) -> (num_envs, bodies_per_env, 3, 3)
        body_inertia = wp.to_torch(self._env.scene_manager.model.body_inertia).view(
            num_envs, cache.bodies_per_env, 3, 3
        )

        # Body quat (xyzw, native Newton order) and ang vel from the state.
        body_q = self._body_q_view()
        body_qd = self._body_qd_view()
        body_quat_xyzw = body_q[:, :, 3:7]
        ang_vel_world = body_qd[:, :, 3:6]

        # I @ omega in body frame, then rotate back to world.
        ang_vel_body = _quat_rotate_inverse(body_quat_xyzw, ang_vel_world)
        ang_momentum_body = torch.einsum("nbij,nbj->nbi", body_inertia, ang_vel_body)
        ang_momentum_world = _quat_rotate(body_quat_xyzw, ang_momentum_body)

        return torch.sum(ang_momentum_world, dim=1)  # (num_envs, 3)

