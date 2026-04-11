"""Sim-agnostic event / reset terms.

These functions work on any ``World`` subclass by reading state through
``env.get_robot_data()`` and writing through
``env.get_robot_state_writer()``. They replace the per-simulator
``push_robot`` / ``reset_root_state_uniform`` implementations that
lived in ``newton_event_terms.py``, ``event_terms.py`` (Genesis), and
``mujoco_event_terms.py``.

Quaternion convention
---------------------
All quaternion parameters (``default_quat_wxyz``) and internal
arithmetic use **wxyz**. Newton presets must convert their native xyzw
values before passing.

Subset convention
-----------------
All writer calls use subset-shaped tensors + ``env_ids``, matching the
``RobotStateWriterProtocol`` contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_mul_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


# ── Sampling utility ─────────────────────────────────────────────────


def _sample_uniform(
    lower: torch.Tensor,
    upper: torch.Tensor,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(shape, device=device) + lower


# ── Push ─────────────────────────────────────────────────────────────


def push_by_setting_velocity(
    env: "World",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    entity_name: str = "robot",
) -> None:
    """Add a random velocity perturbation to the robot's root link.

    Reads the current world-frame root velocity via ``RobotData``,
    samples a uniform perturbation, and writes the result through the
    ``RobotStateWriter``. No pose change — only velocity.

    Works identically across Newton, Genesis, and MuJoCo.
    """
    if len(env_ids) == 0:
        return

    rd = env.get_robot_data(entity_name)
    writer = env.get_robot_state_writer(entity_name)
    device = env.device
    n = len(env_ids)

    lin_vel = rd.root_link_lin_vel_w[env_ids].clone()
    ang_vel = rd.root_link_ang_vel_w[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, key in enumerate(keys):
        lo, hi = velocity_range.get(key, (0.0, 0.0))
        if lo == 0.0 and hi == 0.0:
            continue
        delta = torch.empty(n, device=device).uniform_(lo, hi)
        if i < 3:
            lin_vel[:, i] += delta
        else:
            ang_vel[:, i - 3] += delta

    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)


# ── Reset root state ────────────────────────────────────────────────


def reset_root_state_uniform(
    env: "World",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    default_pos: tuple[float, ...] = (0.0, 0.0, 0.34),
    default_quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    entity_name: str = "robot",
) -> None:
    """Reset root pose + velocity with uniform random perturbations.

    Works identically across Newton, Genesis, and MuJoCo. The
    sim-specific ``reset_root_state_uniform`` implementations can now
    be replaced by thin preset entries that pass the right
    ``default_pos`` / ``default_quat_wxyz`` from the robot config.

    **mjlab env_origins**: if ``env.scene_manager`` has a ``.scene``
    with an ``env_origins`` tensor (mjlab's multi-env offset), the
    function adds ``env_origins[env_ids]`` to the position
    automatically. Newton and Genesis ignore this because their scene
    managers have no ``scene.env_origins`` attribute.

    Args:
        env: Any environment satisfying the RobotData + Writer APIs.
        env_ids: Environments to reset.
        pose_range: Per-axis ``(min, max)`` perturbation for position
            (``x/y/z``) and orientation (``roll/pitch/yaw`` in radians).
        velocity_range: Optional per-axis ``(min, max)`` for initial
            root velocity. ``None`` → zero velocity.
        default_pos: Default root position ``(x, y, z)`` before
            perturbation. Comes from robot config.
        default_quat_wxyz: Default root orientation ``(w, x, y, z)``
            before perturbation. **wxyz** convention.
        entity_name: Entity to reset.
    """
    if len(env_ids) == 0:
        return

    writer = env.get_robot_state_writer(entity_name)
    device = env.device
    n = len(env_ids)

    # ── Sample pose perturbation ──────────────────────────────────
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

    # ── Position: default + perturbation ──────────────────────────
    default_pos_t = torch.tensor(default_pos, device=device, dtype=torch.float32)
    pos = default_pos_t.unsqueeze(0).expand(n, -1) + pose_samples[:, 0:3]

    # Auto-detect mjlab env_origins offset
    _scene = getattr(env.scene_manager, "scene", None)
    env_origins = getattr(_scene, "env_origins", None)
    if env_origins is not None:
        pos = pos + env_origins[env_ids]

    # ── Orientation: default quat * delta quat (all wxyz) ─────────
    default_quat_t = torch.tensor(
        default_quat_wxyz, device=device, dtype=torch.float32
    ).unsqueeze(0).expand(n, -1)

    # Euler perturbation → quaternion delta
    roll, pitch, yaw = pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=device)
    axis_y = torch.tensor([0.0, 1.0, 0.0], device=device)
    axis_z = torch.tensor([0.0, 0.0, 1.0], device=device)
    q_roll = quat_from_angle_axis_wxyz(roll, axis_x)
    q_pitch = quat_from_angle_axis_wxyz(pitch, axis_y)
    q_yaw = quat_from_angle_axis_wxyz(yaw, axis_z)
    delta_quat = quat_mul_wxyz(quat_mul_wxyz(q_yaw, q_pitch), q_roll)

    quat_wxyz = quat_mul_wxyz(default_quat_t, delta_quat)

    # ── Velocity ──────────────────────────────────────────────────
    lin_vel = torch.zeros((n, 3), device=device)
    ang_vel = torch.zeros((n, 3), device=device)
    if velocity_range:
        vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
        vel_ranges = torch.tensor(vel_range_list, device=device)
        vel_samples = _sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device)
        lin_vel = vel_samples[:, 0:3]
        ang_vel = vel_samples[:, 3:6]

    # ── Write ─────────────────────────────────────────────────────
    writer.set_root_pose(pos, quat_wxyz, env_ids=env_ids)
    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)


# ── Reset joint state ──────────────────────────────────────────────


def reset_joints_by_offset(
    env: "World",
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    entity_name: str = "robot",
) -> None:
    """Reset actuated joint positions/velocities with uniform noise.

    Uses ``act_manager.offset`` as the default joint positions (the
    canonical cross-sim source for default DOF values) and
    ``joint_pos_limits`` from ``RobotData`` for clamping.

    Works identically across Newton, Genesis, and MuJoCo.

    Args:
        env: Any environment satisfying the RobotData + Writer APIs.
        env_ids: Environments to reset.
        position_range: ``(min, max)`` uniform noise added to defaults.
        velocity_range: ``(min, max)`` uniform noise for velocities.
            Defaults to ``(0.0, 0.0)`` (zero velocity).
        entity_name: Entity to reset.
    """
    if len(env_ids) == 0:
        return

    writer = env.get_robot_state_writer(entity_name)
    device = env.device
    n = len(env_ids)

    # Default joint positions from action manager offset.
    # offset shape: (num_envs, num_actuated) — slice by env_ids.
    default_pos = env.act_manager.offset[env_ids].clone()
    num_joints = default_pos.shape[-1]

    # Ensure default_pos is always 2-D (n, num_joints) even for 1 env.
    if default_pos.dim() == 1:
        default_pos = default_pos.unsqueeze(0)

    # Add position noise
    if position_range != (0.0, 0.0):
        noise = torch.empty(n, num_joints, device=device).uniform_(
            position_range[0], position_range[1],
        )
        default_pos = default_pos + noise

    # Joint velocities
    if velocity_range == (0.0, 0.0):
        joint_vel = torch.zeros(n, num_joints, device=device)
    else:
        joint_vel = torch.empty(n, num_joints, device=device).uniform_(
            velocity_range[0], velocity_range[1],
        )

    writer.set_dof_positions(default_pos, env_ids=env_ids)
    writer.set_dof_velocities(joint_vel, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)
