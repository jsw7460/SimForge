"""NewtonRobotStateWriter — zero-copy write API for Newton entity state.

Implements :class:`RobotStateWriterProtocol` using **zero-copy torch
views** of Newton's warp state arrays.  ``wp.to_torch()`` returns a
tensor that shares the underlying GPU memory with the warp array, so
writing to the torch view directly mutates the simulator state — no
conversion overhead.

The views are created once at construction time and reused on every
call.  This mirrors the pattern used by Newton's own RL examples
(``newton/solvers/kamino/examples/rl/simulation.py``).

Conventions
-----------

**wxyz quaternion.** ``set_root_pose`` accepts wxyz and converts to
Newton's native xyzw before writing.

**Actuated-only values.** ``set_dof_positions`` / ``set_dof_velocities``
accept tensors of shape ``(N, num_actuated)`` and write them to the
correct generalized-coordinate indices via ``actuated_q_indices`` /
``actuated_qd_indices``.

**eval_fk.** Must be called after joint/root writes to recompute
body transforms (``body_q``) from the updated ``joint_q``.

**Root destination is resolved once, by the view.** Where a root pose
lives depends on the base type: a floating articulation keeps it in the
first seven coordinates of ``joint_q``, a welded one has no such
coordinates at all and its pose is the weld anchor in
``model.joint_X_p``. Slicing ``joint_q[:, 0:7]`` at every call is only
correct for the first case — on a welded arm those seven slots are
``joint1`` through the gripper, so a root write silently replaces joint
angles with coordinates and no exception is raised. So the destination
is bound at construction from ``ArticulationView.get_root_transforms``,
which already branches on base type, and every later write goes there.
This mirrors IsaacLab's Newton backend, which binds the same array once
(``isaaclab_newton/.../articulation_data.py``) instead of indexing raw
generalized coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from torch import Tensor

if TYPE_CHECKING:
    from newton.selection import ArticulationView

    from rlworld.rl.envs.newton.newton_env import NewtonEnv


def _bind_root_view(arr, *, source, view: ArticulationView, what: str) -> Tensor:
    """Bind one root array as a writable ``(num_worlds, n)`` torch view.

    ``ArticulationView`` hands back either a strided view into the
    simulator's own memory or, when the articulation selection needs
    index gathering, a freshly allocated **staging copy**
    (``newton/_src/utils/selection.py``, ``_get_attribute_values``).
    Reads cannot tell the difference; a write into the copy is discarded
    with no error, which would look like "reset silently stopped
    working". IsaacLab binds the same array without checking. Verify the
    binding aliases the source and refuse otherwise.
    """
    if view.count_per_world != 1:
        raise RuntimeError(
            f"Newton root writer expects one articulation per world for this entity, "
            f"got count_per_world={view.count_per_world}."
        )
    if arr.ptr is None or source.ptr is None:
        raise RuntimeError(f"Newton {what} bound to an empty array; the entity has no root state to write.")
    if not (source.ptr <= arr.ptr < source.ptr + source.capacity):
        raise RuntimeError(
            f"Newton {what} resolved to a staging copy rather than a view of simulator memory, "
            "so writes would be silently discarded. This happens when the articulation selection "
            "is index-gathered rather than regularly strided."
        )
    tensor = wp.to_torch(arr)
    if tensor.data_ptr() != arr.ptr:
        raise RuntimeError(f"Newton {what} did not survive the torch conversion as a zero-copy view.")
    # Basic indexing keeps this a view; ``reshape`` would silently copy a
    # non-contiguous tensor and drop every subsequent write.
    return tensor[:, 0, :]


class NewtonRobotStateWriter:
    """Write-side companion to :class:`NewtonRobotData`.

    Uses zero-copy torch views of ``state.joint_q`` and
    ``state.joint_qd`` for maximum write performance.
    """

    def __init__(self, env: NewtonEnv, view: ArticulationView) -> None:
        self._env = env
        self._view = view

        # Actuated joint index mappings
        self._q_indices = env.act_manager.actuated_q_indices
        self._qd_indices = env.act_manager.actuated_qd_indices

        # Zero-copy torch views of the warp state arrays.
        # These share GPU memory — torch writes update warp directly.
        model = env.scene_manager.model
        num_worlds = model.world_count
        coords_per_world = model.joint_coord_count // num_worlds
        dofs_per_world = model.joint_dof_count // num_worlds
        state = env.scene_manager.state

        self._joint_q = wp.to_torch(state.joint_q).reshape(num_worlds, coords_per_world)
        self._joint_qd = wp.to_torch(state.joint_qd).reshape(num_worlds, dofs_per_world)

        # Root destinations, resolved once (see the module docstring).
        self._root_pose = _bind_root_view(
            view.get_root_transforms(state),
            source=state.joint_q if view.is_floating_base else model.joint_X_p,
            view=view,
            what="root transforms",
        )
        root_vel = view.get_root_velocities(state)
        # ``None`` means the base is welded: no root velocity state exists to
        # write. Kept as None so the write raises rather than landing anywhere.
        self._root_vel = (
            None
            if root_vel is None
            else _bind_root_view(root_vel, source=state.joint_qd, view=view, what="root velocities")
        )

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(self, values: Tensor, env_ids: Tensor | None = None) -> None:
        """Write actuated joint positions via zero-copy view."""
        if env_ids is not None:
            self._joint_q[env_ids.unsqueeze(1), self._q_indices.unsqueeze(0)] = values
        else:
            self._joint_q[:, self._q_indices] = values

    def set_dof_velocities(self, values: Tensor, env_ids: Tensor | None = None) -> None:
        """Write actuated joint velocities via zero-copy view."""
        if env_ids is not None:
            self._joint_qd[env_ids.unsqueeze(1), self._qd_indices.unsqueeze(0)] = values
        else:
            self._joint_qd[:, self._qd_indices] = values

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Write root link position + orientation (wxyz → xyzw)."""
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        rows = env_ids if env_ids is not None else slice(None)
        self._root_pose[rows, 0:3] = pos
        self._root_pose[rows, 3:7] = quat_xyzw

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Write root link linear + angular velocity."""
        if self._root_vel is None:
            raise ValueError(
                "Cannot write root velocity for a welded articulation: it has no root "
                "velocity coordinates. Its pose can still be written."
            )
        rows = env_ids if env_ids is not None else slice(None)
        self._root_vel[rows, 0:3] = lin_vel
        self._root_vel[rows, 3:6] = ang_vel

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: Tensor | None = None) -> None:
        """Re-evaluate forward kinematics for the selected environments."""
        self._view.eval_fk(self._env.scene_manager.state, mask=self._mask(env_ids))

    # ==================================================================
    # Internals
    # ==================================================================

    def _mask(self, env_ids: Tensor | None):
        if env_ids is None:
            return None
        num_worlds = self._env.scene_manager.model.world_count
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=self._env.device)
        mask[env_ids] = True
        return wp.from_torch(mask)


class NewtonRigidObjectStateWriter:
    """Root-only write API for a passive rigid object.

    Covers both a free body and an immovable fixture: the scene manager loads
    ``floating=False`` rigid objects as *kinematic* free bodies, so every
    passive object has a real root joint and the same write path.

    Unlike :class:`NewtonRobotStateWriter`, which writes the *first*
    articulation's root via a hardcoded ``joint_q[0:7]`` slice, a rigid
    object's free joint sits at a different coordinate offset within the
    per-world ``joint_q``. We therefore go through the object's own
    :class:`~newton.selection.ArticulationView`, whose
    ``set_root_transforms`` / ``set_root_velocities`` resolve the correct
    per-entity coordinates — the write-side mirror of how
    :class:`NewtonRigidObjectData` reads root state via the same view.

    The view setters take a value array sized for *all* articulations in the
    view (one per world) plus a mask selecting which to write, so we stage a
    full-width buffer and only fill the reset rows.
    """

    def __init__(self, env: NewtonEnv, view: ArticulationView, immovable: bool = False) -> None:
        self._env = env
        self._view = view
        self._immovable = immovable

    def _staged(self, env_ids: Tensor | None, values: Tensor, width: int) -> Tensor:
        """Full ``(num_worlds, width)`` buffer with ``values`` at the reset rows.

        Masked-out rows are never written by the view kernel, so their content
        (zeros) is irrelevant.
        """
        num_worlds = self._env.scene_manager.model.world_count
        buf = torch.zeros((num_worlds, width), device=self._env.device, dtype=torch.float32)
        buf[env_ids if env_ids is not None else slice(None)] = values
        return buf

    def set_root_pose(self, pos: Tensor, quat_wxyz: Tensor, env_ids: Tensor | None = None) -> None:
        """Write root pose (wxyz → Newton-native xyzw transform).

        Immovable fixtures take this same path: the scene manager loads a
        ``floating=False`` rigid object as a *kinematic* free body rather than
        welding it, so its pose is ordinary per-environment joint state.
        (A welded body's pose lives in ``model.joint_X_p``, which Newton's
        ``ArticulationView`` cannot write at all — it slices the root joint out
        with an integer index and the masked-write kernels only implement the
        resulting array's ndim 3 and 4, never 2.)
        """
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        transforms = self._staged(env_ids, torch.cat([pos, quat_xyzw], dim=-1), 7)
        state = self._env.scene_manager.state
        self._view.set_root_transforms(state, wp.from_torch(transforms, dtype=wp.transform), mask=self._mask(env_ids))

    def set_root_velocity(self, lin_vel: Tensor, ang_vel: Tensor, env_ids: Tensor | None = None) -> None:
        """Write root spatial velocity (Newton layout: linear, angular).

        Raises:
            ValueError: If the entity is immovable. A kinematic body does have
                velocity DOFs, but they are pinned by a huge armature and mean
                nothing; refusing matches Genesis and mjlab so a preset that
                tries fails the same way everywhere.
        """
        if self._immovable:
            raise ValueError(
                "Cannot write root velocity for fixed-base entity: it does not respond to applied "
                "forces. Its pose can still be set per environment."
            )
        velocities = self._staged(env_ids, torch.cat([lin_vel, ang_vel], dim=-1), 6)
        state = self._env.scene_manager.state
        self._view.set_root_velocities(
            state, wp.from_torch(velocities, dtype=wp.spatial_vector), mask=self._mask(env_ids)
        )

    def eval_fk(self, env_ids: Tensor | None = None) -> None:
        """Re-evaluate forward kinematics for the selected environments."""
        self._view.eval_fk(self._env.scene_manager.state, mask=self._mask(env_ids))

    def _mask(self, env_ids: Tensor | None):
        if env_ids is None:
            return None
        num_worlds = self._env.scene_manager.model.world_count
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=self._env.device)
        mask[env_ids] = True
        return wp.from_torch(mask)
