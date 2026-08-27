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

import newton
import torch
import warp as wp
from torch import Tensor

from rlworld.rl.envs.site_frames import SiteReaderMixin
from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_rotate_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from newton import State
    from newton.selection import ArticulationView

    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRigidObjectData(SiteReaderMixin):
    """RigidObjectData implementation (root + body reads) for Newton entities.

    Joint-free entity state backed by an ``ArticulationView``.
    :class:`NewtonRobotData` extends this with the actuated-joint accessors. A
    passive rigid object (a graspable box, a table) is a 0-actuator
    articulation in Newton, so it uses this base directly; an articulated robot
    uses :class:`NewtonRobotData`.
    """

    def __init__(
        self,
        env: NewtonEnv,
        view: ArticulationView,
        entity_name: str,
        default_joint_pos: Tensor | None = None,
    ) -> None:
        self._env = env
        self._view = view
        self._entity_name = entity_name
        self._gravity_vec: Tensor | None = None
        self._default_joint_pos = default_joint_pos
        # Per-body CoM offset *in the body frame* (model.body_com), shape
        # (bodies_per_env, 3) — constant; lazily fetched once. Same per-world
        # layout as state.body_q / body_qd (parallel Model/State arrays), so it
        # broadcasts against _body_q_view() / _body_qd_view().
        self._body_com_local: Tensor | None = None
        # Index of THIS entity's root body within that per-world layout; see
        # :attr:`_root_body_index`. Lazily resolved once.
        self._root_body_idx: int | None = None
        # Immovability, resolved once from the model — see :attr:`is_fixed_base`.
        self._is_fixed_base: bool | None = None

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
        """(W, 6) float tensor from root velocities.

        A welded articulation has no free joint, so Newton's
        ``ArticulationView`` returns ``None`` for its root velocities —
        "Non-floating articulations have no root velocities"
        (``newton/_src/utils/selection.py``). Such a body cannot move, so zero
        is what the protocol reports; without the branch the read dies inside
        ``wp.to_torch(None)``. Passive fixtures do not come through here — they
        are loaded as kinematic free bodies and keep real velocity DOFs (pinned
        by a huge armature).
        """
        if not self._view.is_floating_base:
            return torch.zeros(self._env.num_envs, 6, device=self._env.device, dtype=torch.float32)
        wp_arr = self._view.get_root_velocities(state)
        return wp.to_torch(wp_arr).reshape(-1, 6)

    @property
    def _root_body_index(self) -> int:
        """Index of *this entity's* root body in the flat per-world body arrays.

        ``_body_q_view`` / ``_body_qd_view`` / ``_body_com_local_view`` span
        every body in a world, because Newton keeps one flat ``body_q`` for the
        whole model. Row 0 is therefore the first entity's root body — the
        robot's — not necessarily this entity's, so a rigid object must locate
        its own row. The view's ``link_labels`` are the very strings
        ``model.body_label`` holds (``selection.py`` copies them), and a
        Newton articulation lists its root link first, so the label lookup is
        exact.
        """
        if self._root_body_idx is None:
            from rlworld.rl.envs.utils.newton.body_cache import get_cache

            cache = get_cache(self._env)
            labels = self._env.scene_manager.model.body_label[: cache.bodies_per_env]
            self._root_body_idx = labels.index(self._view.link_labels[0])
        return self._root_body_idx

    # ------------------------------------------------------------------
    # Read: raw state (used by observations, event terms, etc.)
    # ------------------------------------------------------------------

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
        c = self._body_com_local_view()[self._root_body_index]  # (3,) — CoM offset, body frame
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
        c = self._body_com_local_view()[self._root_body_index]  # (3,)
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
    def is_fixed_base(self) -> bool:
        """Whether this entity is immovable — welded, or kinematic.

        Two Newton shapes mean "does not move under physics". An articulation
        with no root joint is welded outright. A passive fixture is instead a
        free body flagged ``BodyFlags.KINEMATIC``: it keeps a writable pose
        (which is what lets a reset event place it per environment) but does
        not respond to applied forces. Both report True, because what callers
        branch on is the behaviour — no velocity state worth writing — not the
        joint topology underneath.
        """
        if self._is_fixed_base is None:
            if not self._view.is_floating_base:
                self._is_fixed_base = True
            else:
                flags = wp.to_torch(self._env.scene_manager.model.body_flags)
                self._is_fixed_base = bool(int(flags[self._root_body_index]) & int(newton.BodyFlags.KINEMATIC))
        return self._is_fixed_base

    # ------------------------------------------------------------------
    # Body-level reads
    # ------------------------------------------------------------------

    def find_body_index(self, body_name: str) -> int:
        """Resolve a body name to its per-env body index, ON THIS ENTITY.

        The body cache is model-wide and stores bare leaf names, so with
        two robots built from one asset every name appears twice. Taking
        the first match then hands the second robot the FIRST robot's
        body — silently, and only for the second robot, which is the
        hardest kind of wrong to see: one arm behaves and the other
        reports the other arm's hand as its own.

        So the search is scoped by the entity's label prefix, which the
        scene manager assigns whenever a scene holds more than one robot
        (``scene.py``: ``add_builder(label_prefix=entity_name)``). With a
        single robot there is no prefix and no ambiguity to resolve.
        """
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(self._env)
        indices = cache.get_body_indices(body_name)
        if not indices:
            raise ValueError(f"Body name {body_name!r} not found in Newton model. Available bodies: {cache.body_names}")
        if len(indices) == 1:
            return indices[0]

        labels = self._env.scene_manager.model.body_label
        prefix = self._body_label_prefix()
        if prefix is None:
            raise ValueError(
                f"Body name {body_name!r} matches {len(indices)} bodies in the Newton model and entity "
                f"{self._entity_name!r} has no label prefix to tell them apart. Two entities built from one "
                "asset need distinct label prefixes."
            )
        scoped = [i for i in indices if labels[i].startswith(f"{prefix}/")]
        if len(scoped) != 1:
            raise ValueError(
                f"Body name {body_name!r} matches {len(scoped)} bodies under prefix {prefix!r} "
                f"(entity {self._entity_name!r}); it must name exactly one."
            )
        return scoped[0]

    def _body_label_prefix(self) -> str | None:
        """This entity's Newton label prefix, or None when it has none."""
        entity = self._env.scene_manager.entities.get(self._entity_name)
        if entity is None:
            return None
        return getattr(entity["config"], "body_label_prefix", None)

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
        use. Row 0 is the first entity's root body, which is this entity's root
        only for the robot — use :attr:`_root_body_index` for the root row.
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

    def body_pos_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_pos_w_all[:, body_ids, :]

    def body_lin_vel_w_by_ids(self, body_ids: Tensor) -> Tensor:
        return self.body_lin_vel_w_all[:, body_ids, :]

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


class NewtonRobotData(NewtonRigidObjectData):
    """Articulation state for Newton: RigidObjectData + actuated-joint reads.

    Adds the actuated-joint accessors on top of :class:`NewtonRigidObjectData`.
    Constructed for articulated robots; ``default_joint_pos`` is supplied via
    the base ``__init__``.
    """

    def __init__(self, env, view, *, entity_name: str, soft_joint_pos_limit_factor: float, **kwargs) -> None:
        super().__init__(env, view, entity_name=entity_name, **kwargs)
        # From ArticulationCfg — used by soft_joint_pos_limits (the same
        # factor mjlab applies to its data.soft_joint_pos_limits).
        self._soft_joint_pos_limit_factor = float(soft_joint_pos_limit_factor)

    @property
    def _indexing(self):
        """This entity's joint indexing.

        Every joint read below used to go through the ACTION MANAGER's,
        which belongs to the driven robot: a second robot's joint_pos
        returned the first robot's joints, in the first robot's count.
        Nothing raised, because those indices are valid — they simply
        describe another machine.
        """
        return self._env.entity_indexing(self._entity_name)

    def _joint_coords(self) -> Tensor:
        """One world's slice of the model-wide joint coordinate array.

        The reads are indexed the same way the writes are — Newton keeps
        one flat ``joint_q`` for the whole model and ``newton_q_indices``
        addresses it. Reading the entity VIEW's own array instead would
        need view-local offsets, and mixing the two is what made the
        driven robot work by coincidence: as the first entity its global
        offsets start at zero and happen to equal its local ones.
        """
        model = self._env.scene_manager.model
        worlds = model.world_count
        return wp.to_torch(self._state.joint_q).view(worlds, model.joint_coord_count // worlds)

    def _joint_dofs(self) -> Tensor:
        """One world's slice of the model-wide joint velocity array."""
        model = self._env.scene_manager.model
        worlds = model.world_count
        return wp.to_torch(self._state.joint_qd).view(worlds, model.joint_dof_count // worlds)

    @property
    def default_joint_pos(self) -> Tensor:
        return self._default_joint_pos

    @property
    def joint_pos(self) -> Tensor:
        return self._joint_coords()[:, self._indexing.newton_q_indices]

    @property
    def joint_vel(self) -> Tensor:
        return self._joint_dofs()[:, self._indexing.newton_qd_indices]

    @property
    def applied_torque(self) -> Tensor:
        """Per-DOF actuator torque in actuated order.

        Two sources, matching how the joint is actually driven:

        * **Explicit actuators** (IdealPD / DelayedPD / ...): torque is
          computed in Python and written to ``control.joint_f`` — no mjwarp
          actuator exists (``joint_target_mode=NONE``), so
          ``qfrc_actuator`` would read ~0. Return the action manager's
          post-clip applied torque instead: the exact tensor written to
          ``joint_f``, ``(num_envs, total_action_dim)`` in the same
          canonical actuated order as :attr:`joint_pos`.
        * **Implicit actuators** (mjwarp position actuator): read
          ``state.mujoco.qfrc_actuator`` — the MuJoCo solver's per-DOF
          actuator force after PD-law evaluation and ``effort_limit``
          clipping, transposed into Newton's DOF frame by
          ``convert_qfrc_actuator_from_mj_kernel``. The flat warp array is
          reshaped into ``(num_envs, dofs_per_world)`` and indexed by
          ``newton_qd_indices`` so columns line up with :attr:`joint_pos`
          / :attr:`joint_vel`. Raises ``AttributeError`` if the scene was
          built without requesting ``mujoco:qfrc_actuator`` (the scene
          manager requests it automatically when ``solver_type ==
          "mujoco"``).
        """
        if self._env.act_manager.has_explicit_actuators:
            return self._env.act_manager.applied_torque
        state = self._state
        model = self._env.scene_manager.model
        dofs_per_world = model.joint_dof_count // model.world_count
        qfrc_flat = wp.to_torch(state.mujoco.qfrc_actuator)
        qfrc = qfrc_flat.view(model.world_count, dofs_per_world)
        return qfrc[:, self._indexing.newton_qd_indices]

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
        qd_indices = self._indexing.newton_qd_indices
        return lower_all[qd_indices], upper_all[qd_indices]

    @property
    def soft_joint_pos_limits(self) -> tuple[Tensor, Tensor]:
        """Soft joint position limits in actuated order.

        mjlab/IsaacLab convention: the hard range shrunk around its
        midpoint by ``ArticulationCfg.soft_joint_pos_limit_factor``
        (``mid ± 0.5 · range · factor``) — the same numbers mjlab
        serves via ``data.soft_joint_pos_limits``, so all three
        backends agree for any factor. With the cfg default 1.0 this
        is exactly the hard limits. Returned as ``(num_joints,)``
        tensors, same shape as :attr:`joint_pos_limits`.
        """
        lo, hi = self.joint_pos_limits
        # A joint with no declared range arrives as +-inf (mid = NaN)
        # or as a degenerate 0..0 band (penalty = |q|) depending on the
        # backend's encoding. Both mean "nothing to violate": serve an
        # identity band so limit penalties read 0 for such joints on
        # every backend.
        unlimited = ~(torch.isfinite(lo) & torch.isfinite(hi)) | (hi <= lo)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        f = self._soft_joint_pos_limit_factor
        neg_inf = torch.full_like(lo, -float("inf"))
        pos_inf = torch.full_like(hi, float("inf"))
        return (
            torch.where(unlimited, neg_inf, mid - half * f),
            torch.where(unlimited, pos_inf, mid + half * f),
        )
