"""MujocoRobotData — wrapper that reindexes mjlab EntityData to action manager order.

mjlab's ``EntityData`` exposes joint states in MuJoCo's internal joint
definition order, which may differ from the action manager's actuated
joint order.  This wrapper applies ``joint_ids`` (from the action
manager) so that :attr:`joint_pos` and :attr:`joint_vel` are aligned
with the action/observation ordering used by the rest of JaxRLWorld.

All other properties (root pose, velocities, gravity) are forwarded
from the underlying ``EntityData`` without reindexing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from typing import Any

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from mjlab.entity import Entity


class MujocoRobotData:
    """RobotData implementation for MuJoCo/mjlab entities."""

    def __init__(
        self,
        entity: Entity,
        joint_ids: Tensor,
        num_envs: int,
        device: torch.device,
        env: "Any | None" = None,
        default_joint_pos: Tensor | None = None,
    ) -> None:
        self._entity = entity
        self._joint_ids = joint_ids
        self._gravity_vec: Tensor | None = None
        self._num_envs = num_envs
        self._device = device
        self._env = env
        self._default_joint_pos = default_joint_pos

    def _get_gravity_vec(self) -> Tensor:
        """Lazily create gravity vector matching current batch size."""
        # Use quat batch size to handle eval env with different num_envs
        n = self._entity.data.root_link_quat_w.shape[0]
        if self._gravity_vec is None or self._gravity_vec.shape[0] != n:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._device,
                dtype=torch.float32,
            ).expand(n, -1).contiguous()
        return self._gravity_vec

    @property
    def root_link_pos_w(self) -> Tensor:
        return self._entity.data.root_link_pos_w

    @property
    def root_link_quat_w(self) -> Tensor:
        return self._entity.data.root_link_quat_w

    @property
    def root_link_lin_vel_w(self) -> Tensor:
        return self._entity.data.root_link_lin_vel_w

    @property
    def root_link_ang_vel_w(self) -> Tensor:
        return self._entity.data.root_link_ang_vel_w

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        return self._entity.data.root_link_lin_vel_b

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        return self._entity.data.root_link_ang_vel_b

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
        """Actuated joint positions in action manager order."""
        return self._entity.data.joint_pos[:, self._joint_ids]

    @property
    def joint_vel(self) -> Tensor:
        """Actuated joint velocities in action manager order."""
        return self._entity.data.joint_vel[:, self._joint_ids]

    @property
    def applied_torque(self) -> Tensor:
        """Per-DOF actuator torque in action manager order.

        Reads mjlab's ``EntityData.qfrc_actuator`` which is the
        MuJoCo ``qfrc_actuator`` field sliced to this entity's joint
        DoFs — i.e. post-PD-law, post-actfrcrange clip. Re-indexed by
        ``_joint_ids`` to match action manager ordering.
        """
        return self._entity.data.qfrc_actuator[:, self._joint_ids]

    @property
    def joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Hard joint position limits — not exposed by mjlab.

        mjlab only stores the *soft* limits (already scaled by the soft
        limit factor) in ``entity.data.soft_joint_pos_limits`` with shape
        ``(num_envs, num_joints, 2)``. There is no separate hard-limit
        accessor.

        Phase D-1 only migrates Newton + Genesis ``joint_pos_limits_mjlab``,
        so this stub is never called from active code paths. MuJoCo's
        ``joint_pos_limits`` reward function (in
        ``mdp/rewards/mujoco/reward_terms.py``) reads
        ``soft_joint_pos_limits`` directly and is unchanged.

        Raises:
            NotImplementedError: Always. See note above for the alternative.
        """
        raise NotImplementedError(
            "MujocoRobotData does not expose hard joint position limits. "
            "mjlab only stores soft limits via "
            "``entity.data.soft_joint_pos_limits``. Use mjlab's "
            "``joint_pos_limits`` reward function in "
            "``mdp/rewards/mujoco/reward_terms.py`` instead."
        )

    @property
    def soft_joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Soft joint position limits (mjlab-scaled) in actuated order.

        Reads ``entity.data.soft_joint_pos_limits`` which mjlab exposes
        as ``(num_envs, num_joints, 2)``. We take the first env's slice
        (limits are shared across envs in the common case) and index
        by ``_joint_ids`` to align with the action manager's joint
        ordering.

        Returns:
            ``(lower, upper)``, each shape ``(num_actuated_joints,)``.
        """
        limits = self._entity.data.soft_joint_pos_limits
        sliced = limits[0, self._joint_ids]
        return sliced[:, 0], sliced[:, 1]

    # ------------------------------------------------------------------
    # Body-level reads
    # ------------------------------------------------------------------

    def find_body_index(self, body_name: str) -> int:
        """Resolve a body name to mjlab's body index.

        Calls ``entity.find_bodies([body_name])`` which returns a tuple
        ``(body_ids: list[int], body_names: list[str])``. We return the
        first index. mjlab's name→index map is precomputed at scene
        compile time, so this lookup is cheap.
        """
        body_ids, _ = self._entity.find_bodies([body_name], preserve_order=True)
        if not body_ids:
            raise ValueError(
                f"Body name {body_name!r} not found in mjlab entity"
            )
        return body_ids[0]

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

    @property
    def body_pos_w_all(self) -> Tensor:
        """World-frame positions of all bodies. Shape ``(num_envs, num_bodies, 3)``.

        Reads mjlab's pre-computed ``entity.data.body_link_pos_w``.
        """
        return self._entity.data.body_link_pos_w

    @property
    def body_quat_w_all(self) -> Tensor:
        """World-frame orientations of all bodies, wxyz. Shape ``(num_envs, num_bodies, 4)``.

        mjlab uses wxyz natively — no reordering needed.
        """
        return self._entity.data.body_link_quat_w

    @property
    def body_lin_vel_w_all(self) -> Tensor:
        """World-frame linear velocities of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._entity.data.body_link_lin_vel_w

    @property
    def body_ang_vel_w_all(self) -> Tensor:
        """World-frame angular velocities of all bodies. Shape ``(num_envs, num_bodies, 3)``."""
        return self._entity.data.body_link_ang_vel_w

    # ------------------------------------------------------------------
    # Per-name body/site reads
    # ------------------------------------------------------------------

    def body_pos_w(self, names: "list[str]") -> Tensor:
        body_ids, _ = self._entity.find_bodies(list(names), preserve_order=True)
        return self._entity.data.body_link_pos_w[:, body_ids, :]

    def body_lin_vel_w(self, names: "list[str]") -> Tensor:
        body_ids, _ = self._entity.find_bodies(list(names), preserve_order=True)
        return self._entity.data.body_link_lin_vel_w[:, body_ids, :]

    def site_pos_w(self, names: "list[str]") -> Tensor:
        site_ids, _ = self._entity.find_sites(list(names), preserve_order=True)
        return self._entity.data.site_pos_w[:, site_ids, :]

    def site_lin_vel_w(self, names: "list[str]") -> Tensor:
        site_ids, _ = self._entity.find_sites(list(names), preserve_order=True)
        return self._entity.data.site_lin_vel_w[:, site_ids, :]

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def angular_momentum_w(self, sensor_name: str | None = None) -> Tensor:
        """Whole-body angular momentum read from an mjlab subtreeangmom sensor.

        mjlab supports MuJoCo's built-in ``subtreeangmom`` sensor which
        computes the angular momentum of a subtree about a body's
        reference frame. The sensor is registered in the robot XML
        (e.g. ``<subtreeangmom name="root_angmom" body="pelvis"/>``)
        and looked up at runtime via ``env.scene_manager.get_sensor``.

        Args:
            sensor_name: Full sensor identifier in
                ``"<entity>/<sensor>"`` form (e.g.
                ``"robot/root_angmom"``). Required — there is no
                automatic default for now.

        Returns:
            Tensor of shape ``(num_envs, 3)`` — angular momentum in
            world frame.

        Raises:
            ValueError: If ``sensor_name`` is None or the parent env was
                not provided to this RobotData instance.
        """
        if sensor_name is None:
            raise ValueError(
                "MujocoRobotData.angular_momentum_w requires sensor_name. "
                "Pass the full mjlab sensor identifier (e.g. "
                '"robot/root_angmom").'
            )
        if self._env is None:
            raise RuntimeError(
                "MujocoRobotData.angular_momentum_w needs access to the "
                "parent env (for env.scene_manager.get_sensor) but no env "
                "reference was provided to the RobotData constructor. "
                "Update MujocoEnv._build_sim_managers to pass env=self."
            )
        sensor = self._env.scene_manager.get_sensor(sensor_name)
        return sensor.data
