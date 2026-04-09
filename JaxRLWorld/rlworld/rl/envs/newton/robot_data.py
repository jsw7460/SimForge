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

    def __init__(self, env: "NewtonEnv", view: "ArticulationView") -> None:
        self._env = env
        self._view = view
        self._gravity_vec: Tensor | None = None

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
    def root_link_lin_vel_b(self) -> Tensor:
        quat_wxyz = self.root_quat_wxyz(self._state)
        lin_vel_w = self.root_lin_vel_w(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        quat_wxyz = self.root_quat_wxyz(self._state)
        ang_vel_w = self.root_ang_vel_w(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, ang_vel_w)

    @property
    def projected_gravity_b(self) -> Tensor:
        return quat_rotate_inverse_wxyz(self.root_link_quat_w, self._get_gravity_vec())

    @property
    def heading_w(self) -> Tensor:
        return quat_to_euler_wxyz(self.root_link_quat_w)[:, 2]

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

        Reads ``state.body_qd`` (shape flattened across worlds), reshapes
        to ``(num_envs, bodies_per_env, 6)``, and indexes the (3:6) slice
        which holds angular velocity in world frame.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache
        cache = get_cache(self._env)
        state = self._env.scene_manager.state
        body_qd = wp.to_torch(state.body_qd).view(self._env.num_envs, cache.bodies_per_env, 6)
        return body_qd[:, body_index, 3:6]

    # ------------------------------------------------------------------
    # Write helpers (used by event terms, scene reset)
    # ------------------------------------------------------------------

    def set_dof_positions(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate positions. values shape: (W, joint_coord_count)."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_positions(state, wp_arr, mask=self._resolve_mask(mask))

    def set_dof_velocities(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate velocities. values shape: (W, joint_dof_count)."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_velocities(state, wp_arr, mask=self._resolve_mask(mask))

    def set_root_state(
        self,
        state: "State",
        pos: Tensor,
        quat_xyzw: Tensor,
        lin_vel: Tensor | None = None,
        ang_vel: Tensor | None = None,
        mask=None,
    ) -> None:
        """Set root transforms and optionally velocities.

        Args:
            pos: (W, 3) position
            quat_xyzw: (W, 4) quaternion in xyzw (Newton native)
            lin_vel: (W, 3) linear velocity, optional
            ang_vel: (W, 3) angular velocity, optional
        """
        resolved_mask = self._resolve_mask(mask)

        transform = torch.cat([pos, quat_xyzw], dim=-1)
        wp_t = wp.from_torch(transform.unsqueeze(1).contiguous(), dtype=wp.transform)
        self._view.set_root_transforms(state, wp_t, mask=resolved_mask)

        if lin_vel is not None and ang_vel is not None:
            vel = torch.cat([lin_vel, ang_vel], dim=-1)
            wp_v = wp.from_torch(vel.unsqueeze(1).contiguous(), dtype=wp.spatial_vector)
            self._view.set_root_velocities(state, wp_v, mask=resolved_mask)

    def eval_fk(self, state: "State", mask=None) -> None:
        """Evaluate forward kinematics for selected environments."""
        self._view.eval_fk(state, mask=self._resolve_mask(mask))

    # ------------------------------------------------------------------
    # Mask utility
    # ------------------------------------------------------------------

    @staticmethod
    def env_ids_to_mask(
        env_ids: torch.Tensor, num_worlds: int, device: torch.device
    ) -> "wp.array":
        """Convert integer env_ids to a boolean warp mask of shape (num_worlds,)."""
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=device)
        mask[env_ids] = True
        return wp.from_torch(mask)

    def _resolve_mask(self, mask):
        if mask is None:
            return None
        if isinstance(mask, wp.array):
            return mask
        if isinstance(mask, torch.Tensor):
            return wp.from_torch(mask)
        return mask
