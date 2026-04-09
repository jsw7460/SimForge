"""Unified proprioception observations using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(entity_name)`` or
``env.contact_manager``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils.quat_utils import quat_to_euler_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def base_lin_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base linear velocity in body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).root_link_lin_vel_b


def base_ang_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base angular velocity in body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).root_link_ang_vel_b


def projected_gravity(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Gravity vector projected into the body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).projected_gravity_b


def base_quat(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base quaternion in world frame, **wxyz** convention.

    Returns:
        Tensor of shape (num_envs, 4).

    Note:
        The legacy ``observations.newton.state.base_quat`` returned the
        Newton-native **xyzw** order. Migrating to this common helper
        normalizes Newton onto the wxyz convention shared by Genesis,
        mjlab, and the ``RobotData`` protocol — old Newton checkpoints
        whose critic obs included ``base_quat`` will need to retrain.
    """
    return env.get_robot_data(entity_name).root_link_quat_w


def base_euler(
    env: World, entity_name: str = "robot", degrees: bool = False
) -> torch.Tensor:
    """Base orientation as Euler angles ``(roll, pitch, yaw)`` in radians.

    Args:
        env: Any environment with a ``RobotData``.
        entity_name: Entity to query.
        degrees: If True, return angles in degrees instead of radians.

    Returns:
        Tensor of shape (num_envs, 3) — ``[roll, pitch, yaw]``.
    """
    quat_wxyz = env.get_robot_data(entity_name).root_link_quat_w
    euler = quat_to_euler_wxyz(quat_wxyz)
    if degrees:
        euler = euler * (180.0 / torch.pi)
    return euler


def _actuated_joint_ids(env: World) -> torch.Tensor | None:
    """Return act_manager._joint_ids if it exists (MuJoCo needs reindexing)."""
    ids = getattr(env.act_manager, "_joint_ids", None)
    if ids is None:
        return None
    # Only return if it's actually a permutation (not identity).
    n = len(ids)
    if n > 0 and not torch.equal(ids, torch.arange(n, device=ids.device)):
        return ids
    return None


def dof_pos(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint positions in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    pos = env.get_robot_data(entity_name).joint_pos
    joint_ids = _actuated_joint_ids(env)
    if joint_ids is not None:
        return pos[:, joint_ids]
    return pos


def dof_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint velocities in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    vel = env.get_robot_data(entity_name).joint_vel
    joint_ids = _actuated_joint_ids(env)
    if joint_ids is not None:
        return vel[:, joint_ids]
    return vel


def dof_pos_nominal_difference(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Joint positions relative to nominal (default) positions, in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    return dof_pos(env, entity_name) - env.act_manager.offset


def base_height(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base height (z-coordinate) above world origin.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    return env.get_robot_data(entity_name).root_link_pos_w[:, 2:3]


def prev_processed_actions(env: World) -> torch.Tensor:
    """Current step's processed actions (used as observation input).

    Note: Despite the name, this returns the *current* processed actions,
    matching the existing Newton/Genesis observation behavior.

    Returns:
        Tensor of shape (num_envs, num_actions).
    """
    return env.act_manager.processed_actions.clone()

def prev_raw_actions(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Current step's processed actions (used as observation input)."""

    return env.act_manager.prev_raw_actions


def raw_actions(env: World) -> torch.Tensor:
    """Current step's raw (unprocessed) actions.

    Returns:
        Tensor of shape (num_envs, num_actions).
    """
    return env.act_manager.raw_actions


def last_processed_actions(env: World) -> torch.Tensor:
    """Previous step's processed actions.

    This is the action applied one step before the current one.
    Matches Walk-These-Ways ``self.last_actions`` in observations.

    Returns:
        Tensor of shape (num_envs, num_actions).
    """
    return env.act_manager.prev_processed_actions.clone()


def clock_inputs(env: World) -> torch.Tensor:
    """Gait clock signals from GaitManager.

    Returns sin(2*pi * warped_foot_phase) for each foot.
    Requires the environment to have a ``gait_manager`` attribute.

    Returns:
        Tensor of shape (num_envs, num_feet).
    """
    return env.gait_manager.clock_inputs


def all_commands(env: World) -> torch.Tensor:
    """All command terms concatenated.

    Returns all registered command terms (e.g., velocity + gait)
    as a single tensor via CommandManager.get_commands_tensor().

    Returns:
        Tensor of shape (num_envs, total_command_dim).
    """
    return env.command_manager.get_commands_tensor()


# Short alias kept for the historical name used by every preset that
# previously imported `command` from `genesis.exteroception` (which was
# misplaced — it was sim-agnostic from day one).
command = all_commands


# ── Contact-based observations ───────────────────────────────────────


def foot_air_time(
    env: World,
    contact_group: str = "feet_ground_contact",
    body_names: "list[str] | None" = None,
    use_last: bool = False,
) -> torch.Tensor:
    """Per-foot air-time observation.

    Args:
        env: Any environment with a contact_manager.
        contact_group: Name of the registered contact group.
        body_names: Optional ordered subset of body names to read; when
            given, the helper passes ``order=body_names`` to the contact
            manager so the result columns line up with the caller's
            foot ordering.
        use_last: If True, return the last completed air-time interval
            (frozen at landing) instead of the live current air-time
            counter. The legacy Newton presets used the "last" variant;
            new code should prefer ``False`` (the current counter).

    Returns:
        Tensor of shape ``(num_envs, num_feet)``.
    """
    if use_last:
        return env.contact_manager.last_air_time(contact_group, order=body_names)
    return env.contact_manager.current_air_time(contact_group, order=body_names)


def foot_contact_indicator(
    env: World,
    contact_group: str = "feet_ground_contact",
    body_names: "list[str] | None" = None,
) -> torch.Tensor:
    """Binary per-foot contact indicator (1.0 = in contact).

    Returns:
        Tensor of shape ``(num_envs, num_feet)``.
    """
    return env.contact_manager.is_contact(contact_group, order=body_names).float()


def foot_contact_forces(
    env: World,
    contact_group: str = "feet_ground_contact",
    body_names: "list[str] | None" = None,
) -> torch.Tensor:
    """Per-foot 3-D contact force, log-scaled and flattened.

    Applies ``sign(F) * log1p(|F|)`` to compress the dynamic range of
    raw contact forces, then flattens the per-foot 3-vectors into a
    single per-env feature vector.

    Returns:
        Tensor of shape ``(num_envs, num_feet * 3)``.
    """
    forces_3d = env.contact_manager.contact_force(contact_group, order=body_names)
    flat = forces_3d.flatten(start_dim=1)
    return torch.sign(flat) * torch.log1p(torch.abs(flat))
