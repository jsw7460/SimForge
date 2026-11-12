"""GenesisRobotStateWriter — write API for a Genesis ``RigidEntity``.

Implements :class:`RobotStateWriterProtocol` against Genesis's native
``entity.set_*`` mutation methods. Genesis hides simulation state
inside ``gs.Scene`` and scopes per-env writes with ``envs_idx``.

How the writes address Genesis matters as much as what they write.
Genesis takes an env selection either as an index tensor, which it
bounds-checks and de-duplicates on the host (a synchronisation per
call, five in ``set_dofs_position``), or as a boolean mask over every
env with a value row per env, which it applies on the device with a
masked copy and masked kernels (forward kinematics, collider clear).
The protocol hands this writer env id tensors; it scatters them into a
mask and every-row buffers (device-side, no synchronisation) and takes
the masked path for every free-base write. DOFs are selected the same
way, with the :class:`GenesisDofSelection` range forms.

Genesis-specific quirks the writer hides from callers:

- Joint positions are written as configuration (``set_qpos``) rather
  than through ``set_dofs_position``, which has no masked path. For
  the hinge and slide joints the selection covers, the configuration
  of a DOF is its reference ``qpos0`` plus its position, exactly the
  sum ``set_dofs_position``'s kernel forms, so the two writes land the
  same bits.
- Root velocity is not a separate API call — Genesis treats the base
  6 DOFs (x, y, z, roll, pitch, yaw) as plain DOFs that you write
  through ``set_dofs_velocity`` on ``range(6)``. ``set_root_velocity``
  packs the linear + angular vectors into a single 6-vec and forwards.
- ``eval_fk`` is a no-op: Genesis re-evaluates kinematics inside its
  next ``scene.step()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.misc import qd_to_torch
from torch import Tensor

from jaxrlworld.rl.envs.genesis.dof_selection import GenesisDofSelection
from jaxrlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

    from jaxrlworld.rl.envs.genesis.genesis_env import GenesisEnv

_ROOT_DOFS = range(6)


class GenesisRobotStateWriter:
    """Write-side companion to :class:`GenesisRobotData`."""

    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        dofs: GenesisDofSelection,
    ) -> None:
        self._env = env
        self._entity = entity
        self._dofs = dofs
        self._num_envs = env.num_envs
        # The solver's reference configuration of this entity, (num_envs, n_qs), as a live view: a joint's
        # configuration is this plus its DOF position (see the module docstring).
        qpos0 = qd_to_torch(entity.solver.rigid_info.qpos0, transpose=True, copy=False)
        self._qpos0 = qpos0[:, entity.q_start : entity.q_start + entity.n_qs]

    # ------------------------------------------------------------------
    # Env selection
    # ------------------------------------------------------------------

    def _mask(self, env_ids: Tensor | None) -> Tensor | None:
        if env_ids is None:
            return None
        mask = torch.zeros(self._num_envs, dtype=torch.bool, device=env_ids.device)
        mask[env_ids] = True
        return mask

    def _rows(self, env_ids: Tensor | None, values: Tensor) -> Tensor:
        """``values`` (one row per entry of ``env_ids``) scattered into a buffer with one row per env."""
        if env_ids is None:
            return values
        rows = torch.zeros((self._num_envs, values.shape[-1]), dtype=values.dtype, device=values.device)
        rows[env_ids] = values
        return rows

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(self, values: Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write actuated joint positions.

        ``zero_velocity=False`` is load-bearing on EVERY Genesis position
        write in this class: Genesis's ``set_pos`` / ``set_quat`` /
        ``set_qpos`` default to ``zero_velocity=True``, which zeroes ALL
        of the entity's dof velocities as a teleport safety measure.
        Under the reset event order (``reset_root`` writes the
        randomized root velocity, then ``reset_joints`` writes joint
        positions) that default silently wiped the root velocity — every
        Genesis episode started from perfect rest while Newton/mjlab
        started with the sampled |v|≈0.25-0.4 m/s, skewing first-step
        tracking and feet rewards on all Genesis presets. The writer must
        only write what it was asked to; velocities are written
        explicitly by the events.
        """
        rows = self._rows(env_ids, self._dofs.to_sim_order(values))
        self._entity.set_qpos(
            rows + self._qpos0[:, self._dofs.qs],
            qs_idx_local=self._dofs.qs,
            envs_idx=self._mask(env_ids),
            zero_velocity=False,
        )

    def set_dof_velocities(self, values: Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write actuated joint velocities."""
        self._entity.set_dofs_velocity(
            velocity=self._rows(env_ids, self._dofs.to_sim_order(values)),
            dofs_idx_local=self._dofs.dofs,
            envs_idx=self._mask(env_ids),
        )

    def set_dof_state(self, positions: Tensor, velocities: Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write joint positions and velocities with a single forward pass.

        Velocity first with ``skip_forward=True``, then the position
        write, whose unconditional FK recomputes the link buffers from
        BOTH the new qpos and the new dof velocities — one forward
        instead of two. ``zero_velocity=False``: see
        :meth:`set_dof_positions`.
        """
        mask = self._mask(env_ids)
        self._entity.set_dofs_velocity(
            velocity=self._rows(env_ids, self._dofs.to_sim_order(velocities)),
            dofs_idx_local=self._dofs.dofs,
            envs_idx=mask,
            skip_forward=True,
        )
        rows = self._rows(env_ids, self._dofs.to_sim_order(positions))
        self._entity.set_qpos(
            rows + self._qpos0[:, self._dofs.qs],
            qs_idx_local=self._dofs.qs,
            envs_idx=mask,
            zero_velocity=False,
        )

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def _reject_velocity_if_fixed_base(self) -> None:
        """Refuse a root *velocity* write on a welded entity.

        A fixed-base entity has no root joint, so it has no velocity DOFs.
        Genesis would index six free-joint DOFs that do not exist and trip a
        device-side assert, poisoning the CUDA context for the rest of the
        process — the failure then surfaces somewhere unrelated. mjlab refuses
        the same call with a ``ValueError``, so raising here keeps the contract
        identical across backends.

        The *pose* write is a different matter and is allowed: see
        :meth:`set_root_pose`.

        Raises:
            ValueError: If this entity is welded to the world.
        """
        if self._entity.base_link.is_fixed:
            raise ValueError(
                "Cannot write root velocity for fixed-base entity: it is welded to the world and has "
                "no velocity DOFs. Its pose can still be set per environment."
            )

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write root link position + orientation. Genesis is wxyz native.

        Works for welded entities too: Genesis's solver writes a fixed link's
        pose straight into ``dyn_state.links.pos`` / ``.quat`` instead of the
        free-joint coordinates (``rigid_solver.set_base_links_pos``), and those
        buffers are per-environment. That is what lets a table or tank be
        placed per env by a reset event — the same capability mjlab provides
        through its mocap base and Newton through its fixed-root mocap bodies.
        A welded entity is addressed with the env ids as given, so Genesis's
        own check that a fixed link's geometry is batched before it takes a
        per-env pose still runs.

        ``zero_velocity=False``: see :meth:`set_dof_positions`.

        ``skip_forward=True`` on the position write only: the quat write
        that immediately follows runs the forward pass, refreshing the
        link state both writes touched in one FK instead of two. The
        quat write must NOT skip — ``set_root_velocity`` reads
        ``get_quat`` (FK-refreshed link state, not qpos) right after,
        and standalone velocity writes (pushes) rely on fresh link
        buffers for the very next observation.
        """
        if self._entity.base_link.is_fixed:
            self._entity.set_pos(pos, envs_idx=env_ids, zero_velocity=False, skip_forward=True)
            self._entity.set_quat(quat_wxyz, envs_idx=env_ids, zero_velocity=False)
            return
        mask = self._mask(env_ids)
        self._entity.set_pos(self._rows(env_ids, pos), envs_idx=mask, zero_velocity=False, skip_forward=True)
        self._entity.set_quat(self._rows(env_ids, quat_wxyz), envs_idx=mask, zero_velocity=False)

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write root link linear + angular velocity (both WORLD frame).

        Genesis has no dedicated root-velocity setter; root velocity is
        the first 6 DOFs (3 linear + 3 angular). The two triplets use
        DIFFERENT frames in Genesis's free joint: the linear dof axes
        are the world unit vectors, but the rotational dof axes are the
        link rotation matrix columns (``cdof_ang = xmat_T`` rows,
        abd/forward_kinematics.py FREE branch) — i.e. the angular dof
        velocities are BODY-frame components, same as MuJoCo's free
        joint. Rotate the world angular velocity into the link frame
        before writing; the pose write precedes this call in the reset
        protocol, so ``get_quat`` already reflects the new orientation.
        The quaternion is read for every env (one zero-copy read) and
        the rows of ``env_ids`` picked on the device.
        """
        self._reject_velocity_if_fixed_base()
        quat_wxyz = self._entity.get_quat()
        if env_ids is not None:
            quat_wxyz = quat_wxyz[env_ids]
        ang_local = quat_rotate_inverse_wxyz(quat_wxyz, ang_vel)
        combined = torch.cat([lin_vel, ang_local], dim=-1)
        self._entity.set_dofs_velocity(
            velocity=self._rows(env_ids, combined),
            dofs_idx_local=_ROOT_DOFS,
            envs_idx=self._mask(env_ids),
        )

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: torch.Tensor | None = None) -> None:
        """No-op: Genesis updates kinematics during ``scene.step()``."""
        return None
