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

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_rotate_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from newton import State
    from newton.selection import ArticulationView

    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRobotData:
    """RobotData implementation + state write API for Newton environments."""

    def __init__(
        self,
        env: NewtonEnv,
        view: ArticulationView,
        default_joint_pos: Tensor | None = None,
    ) -> None:
        self._env = env
        self._view = view
        self._gravity_vec: Tensor | None = None
        self._default_joint_pos = default_joint_pos
        # Per-body CoM offset *in the body frame* (model.body_com), shape
        # (bodies_per_env, 3) — constant; lazily fetched once. Same per-world
        # layout as state.body_q / body_qd (parallel Model/State arrays), so it
        # broadcasts against _body_q_view() / _body_qd_view() and its row 0 is
        # the floating-base root body.
        self._body_com_local: Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _state(self) -> State:
        return self._env.scene_manager.state

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = (
                torch.tensor(
                    [[0.0, 0.0, -1.0]],
                    device=self._env.device,
                    dtype=torch.float32,
                )
                .expand(self._env.num_envs, -1)
                .contiguous()
            )
        return self._gravity_vec

    def _root_transform_floats(self, state: State) -> Tensor:
        """(W, 7) float tensor from root transforms."""
        wp_arr = self._view.get_root_transforms(state)
        return wp.to_torch(wp_arr).reshape(-1, 7)

    def _root_velocity_floats(self, state: State) -> Tensor:
        """(W, 6) float tensor from root velocities."""
        wp_arr = self._view.get_root_velocities(state)
        return wp.to_torch(wp_arr).reshape(-1, 6)

    # ------------------------------------------------------------------
    # Read: raw state (used by observations, event terms, etc.)
    # ------------------------------------------------------------------

    def dof_positions(self, state: State) -> Tensor:
        """Joint coordinate positions. Shape: (W, joint_coord_count)."""
        return wp.to_torch(self._view.get_dof_positions(state)).squeeze(1)

    def dof_velocities(self, state: State) -> Tensor:
        """Joint coordinate velocities. Shape: (W, joint_dof_count)."""
        return wp.to_torch(self._view.get_dof_velocities(state)).squeeze(1)

    def root_pos_w(self, state: State) -> Tensor:
        """Root position in world frame. Shape: (W, 3)."""
        return self._root_transform_floats(state)[:, 0:3]

    def root_quat_wxyz(self, state: State) -> Tensor:
        """Root quaternion in wxyz convention. Shape: (W, 4)."""
        xyzw = self._root_transform_floats(state)[:, 3:7]
        return xyzw[:, [3, 0, 1, 2]]

    def root_quat_xyzw(self, state: State) -> Tensor:
        """Root quaternion in xyzw convention (Newton native). Shape: (W, 4)."""
        return self._root_transform_floats(state)[:, 3:7]

    def root_lin_vel_w(self, state: State) -> Tensor:
        """Root linear velocity in world frame, **at the body center of mass**. Shape: (W, 3).

        This is the raw ``joint_qd[0:3]`` of the floating base, which by
        Newton's documented ``body_qd`` convention is ``(v_com_world, ...)``
        — i.e. the velocity of the *CoM*, not of the link frame origin.
        The RobotData properties below split this: ``root_com_lin_vel_w``
        returns this value as-is, ``root_link_lin_vel_w`` transfers it to
        the link frame origin.
        """
        return self._root_velocity_floats(state)[:, 0:3]

    def root_ang_vel_w(self, state: State) -> Tensor:
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
        # Newton's joint_qd[0:3] is the velocity AT the CoM. Transfer it to the
        # link frame origin O:  v_O = v_C - omega x (R @ c)
        #   c = body_com[root] (CoM offset in the body frame),
        #   R = body->world rotation (from the root quaternion),
        #   omega = root angular velocity in world frame.
        state = self._state
        v_com = self.root_lin_vel_w(state)  # (W, 3) — at CoM
        omega = self.root_ang_vel_w(state)  # (W, 3) — world frame
        quat_wxyz = self.root_quat_wxyz(state)  # (W, 4)
        c = self._body_com_local_view()[0]  # (3,) — root body's CoM offset, body frame
        r_world = quat_rotate_wxyz(quat_wxyz, c.expand_as(v_com))  # R @ c, world
        return v_com - torch.cross(omega, r_world, dim=-1)

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

    # ── Root center-of-mass variants ─────────────────────────────────
    # Newton's body_qd is already CoM-referenced (and body_q + body_com gives
    # the CoM position), so these are the "native" reads.

    @property
    def root_com_pos_w(self) -> Tensor:
        # r_C = r_O + R @ c
        quat_wxyz = self.root_quat_wxyz(self._state)
        c = self._body_com_local_view()[0]  # (3,)
        link_pos = self.root_link_pos_w  # (W, 3) — link frame origin
        return link_pos + quat_rotate_wxyz(quat_wxyz, c.expand_as(link_pos))

    @property
    def root_com_lin_vel_w(self) -> Tensor:
        return self.root_lin_vel_w(self._state)  # raw joint_qd[0:3] = v at CoM

    @property
    def root_com_lin_vel_b(self) -> Tensor:
        return quat_rotate_inverse_wxyz(self.root_quat_wxyz(self._state), self.root_com_lin_vel_w)

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
    def applied_torque(self) -> Tensor:
        """Per-DOF actuator torque in actuated order.

        Reads ``state.mujoco.qfrc_actuator`` — the MuJoCo solver's
        per-DOF actuator force after PD-law evaluation and
        ``effort_limit`` clipping, transposed into Newton's DOF
        frame by ``convert_qfrc_actuator_from_mj_kernel``. The flat
        warp array is reshaped into ``(num_envs, dofs_per_world)`` and
        indexed by ``newton_qd_indices`` so columns line up with
        :attr:`joint_pos` / :attr:`joint_vel`.

        Raises ``AttributeError`` if the scene was built without
        requesting ``mujoco:qfrc_actuator`` (the scene manager requests
        it automatically when ``solver_type == "mujoco"``).
        """
        state = self._state
        model = self._env.scene_manager.model
        dofs_per_world = model.joint_dof_count // model.world_count
        qfrc_flat = wp.to_torch(state.mujoco.qfrc_actuator)
        qfrc = qfrc_flat.view(model.world_count, dofs_per_world)
        return qfrc[:, self._env.act_manager.indexing.newton_qd_indices]

    @property
    def joint_pos_limits(self) -> tuple[Tensor, Tensor]:
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
    def soft_joint_pos_limits(self) -> tuple[Tensor, Tensor]:
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
            raise ValueError(f"Body name {body_name!r} not found in Newton model. Available bodies: {cache.body_names}")
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
        Each spatial_vector is ``(lin.x, lin.y, lin.z, ang.x, ang.y, ang.z)``,
        both in the world frame — but per Newton's documented convention the
        **linear** part is the velocity AT THE BODY CoM (not the link frame
        origin). ``body_lin_vel_w_all`` transfers it to the link origin;
        ``body_com_lin_vel_w_all`` returns it as-is.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(self._env)
        state = self._env.scene_manager.state
        return wp.to_torch(state.body_qd).view(self._env.num_envs, cache.bodies_per_env, 6)

    def _body_com_local_view(self) -> Tensor:
        """``model.body_com`` for one world: (bodies_per_env, 3) — each body's CoM
        offset *expressed in that body's link frame*. Constant; fetched once.

        ``model.body_com`` is a parallel array to ``state.body_q`` (same
        per-world layout), so taking the first ``bodies_per_env`` rows yields
        world 0's bodies in the same order ``_body_q_view`` / ``_body_qd_view``
        use, and row 0 is the floating-base root body.
        """
        if self._body_com_local is None:
            from rlworld.rl.envs.utils.newton.body_cache import get_cache

            cache = get_cache(self._env)
            model = self._env.scene_manager.model
            self._body_com_local = wp.to_torch(model.body_com)[: cache.bodies_per_env].contiguous()
        return self._body_com_local

    @property
    def body_pos_w_all(self) -> Tensor:
        """World-frame positions of all bodies' link frame origins. Shape ``(num_envs, num_bodies, 3)``.

        ``state.body_q`` is the link frame transform, so its translation IS
        the link frame origin.
        """
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
        """World-frame linear velocities of all bodies, at their link frame origins. Shape ``(num_envs, num_bodies, 3)``.

        ``body_qd[:, 0:3]`` is the velocity at each body's CoM; transfer it to
        the link frame origin O:  v_O = v_C - omega x (R @ c).
        """
        qd = self._body_qd_view()
        v_com = qd[:, :, 0:3]  # (W, B, 3) — at CoM
        omega = qd[:, :, 3:6]  # (W, B, 3) — world frame
        c = self._body_com_local_view()  # (B, 3) — CoM offset, body frame
        r_world = quat_rotate_wxyz(self.body_quat_w_all, c.expand_as(v_com))  # R @ c, world
        return v_com - torch.cross(omega, r_world, dim=-1)

    @property
    def body_ang_vel_w_all(self) -> Tensor:
        """World-frame angular velocities of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._body_qd_view()[:, :, 3:6]

    @property
    def body_com_pos_w_all(self) -> Tensor:
        """World-frame positions of all bodies' centers of mass. Shape ``(num_envs, num_bodies, 3)``.

        r_C = r_O + R @ c, where r_O = link frame origin (``state.body_q[:, 0:3]``).
        """
        link_pos = self._body_q_view()[:, :, 0:3]  # (W, B, 3)
        c = self._body_com_local_view()  # (B, 3)
        return link_pos + quat_rotate_wxyz(self.body_quat_w_all, c.expand_as(link_pos))

    @property
    def body_com_lin_vel_w_all(self) -> Tensor:
        """World-frame linear velocities of all bodies at their centers of mass. Shape ``(num_envs, num_bodies, 3)``.

        Newton's native ``body_qd[:, 0:3]`` — already CoM-referenced.
        """
        return self._body_qd_view()[:, :, 0:3]

    # ------------------------------------------------------------------
    # Per-name body/site reads
    # ------------------------------------------------------------------

    def _resolve_body_indices(self, names: list[str]) -> list[int]:
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
                raise ValueError(f"Body name {name!r} not found in Newton model. Available bodies: {cache.body_names}")
            out.append(indices[0])
        return out

    def body_pos_w(self, names: list[str]) -> Tensor:
        idxs = self._resolve_body_indices(list(names))
        return self.body_pos_w_all[:, idxs, :]

    def body_lin_vel_w(self, names: list[str]) -> Tensor:
        idxs = self._resolve_body_indices(list(names))
        return self.body_lin_vel_w_all[:, idxs, :]

    def site_pos_w(self, names: list[str]) -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_pos_w. Newton has "
            "no equivalent of MuJoCo sites — use body_pos_w with body names."
        )

    def site_lin_vel_w(self, names: list[str]) -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_lin_vel_w. Newton has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w with body names."
        )

    def body_pos_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_pos_w_all[:, body_ids, :]

    def body_lin_vel_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_lin_vel_w_all[:, body_ids, :]

    def site_pos_w_by_ids(self, site_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_pos_w_by_ids. Newton has "
            "no equivalent of MuJoCo sites — use body_pos_w_by_ids with body ids."
        )

    def site_lin_vel_w_by_ids(self, site_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            "NewtonRobotData does not implement site_lin_vel_w_by_ids. Newton has "
            "no equivalent of MuJoCo sites — use body_lin_vel_w_by_ids with body ids."
        )

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def angular_momentum_w(self, sensor_name: str | None = None) -> Tensor:
        """Whole-body angular momentum (world frame) about the system CoM.

        Matches MuJoCo's ``subtreeangmom`` sensor (subtree rooted at the
        floating-base root = whole robot). König's decomposition:

            L = sum_i [ m_i * (r_i - r_c) x v_i              # orbital
                        +  R_i @ I_i_local @ R_i^T @ omega_i ]  # spin

        where r_i, v_i are body CoMs / CoM velocities, r_c is the system
        CoM, R_i / omega_i / I_i are the body world rotation, world
        angular velocity, and body-frame inertia. ``sensor_name`` is
        ignored (no Newton sensor to read from).

        Before this, Newton returned only the spin sum. The orbital term
        is dominant when limbs swing out from the body CoM, which is why
        the previous version underestimated by 1-2 orders of magnitude
        relative to ``subtreeangmom``.
        """
        from rlworld.rl.envs.mdp.observations.newton.state import (
            _quat_rotate,
            _quat_rotate_inverse,
        )
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(self._env)
        num_envs = self._env.num_envs
        bodies_per_env = cache.bodies_per_env
        model = self._env.scene_manager.model

        # Per-body, per-env model arrays (flat -> (W, B, ...)).
        body_inertia = wp.to_torch(model.body_inertia).view(num_envs, bodies_per_env, 3, 3)
        body_mass = wp.to_torch(model.body_mass).view(num_envs, bodies_per_env)  # (W, B)

        body_q = self._body_q_view()
        body_qd = self._body_qd_view()
        body_quat_xyzw = body_q[:, :, 3:7]
        ang_vel_world = body_qd[:, :, 3:6]  # (W, B, 3)
        # Newton's ``body_qd[:, :, 0:3]`` is linear velocity AT each body's CoM
        # — exactly what the orbital term wants. No transfer needed.
        v_com_w = body_qd[:, :, 0:3]  # (W, B, 3)

        # Spin: sum_i R_i @ I_i_local @ R_i^T @ omega_i.
        ang_vel_body = _quat_rotate_inverse(body_quat_xyzw, ang_vel_world)
        spin_body = torch.einsum("nbij,nbj->nbi", body_inertia, ang_vel_body)
        spin_world = _quat_rotate(body_quat_xyzw, spin_body)  # (W, B, 3)

        # Orbital: sum_i m_i * (r_i - r_c) x v_i.
        r_com_w = self.body_com_pos_w_all  # (W, B, 3) — body CoM in world
        total_mass = body_mass.sum(dim=1)  # (W,)
        # Guard against the degenerate case (no massive bodies in env) — should
        # not happen with a real robot but keeps the divide explicit.
        r_c = (body_mass.unsqueeze(-1) * r_com_w).sum(dim=1) / total_mass.unsqueeze(-1)  # (W, 3)
        r_rel = r_com_w - r_c.unsqueeze(1)  # (W, B, 3)
        orbital_per_body = body_mass.unsqueeze(-1) * torch.cross(r_rel, v_com_w, dim=-1)
        orbital = orbital_per_body.sum(dim=1)  # (W, 3)

        return spin_world.sum(dim=1) + orbital  # (W, 3)
