"""Newton state observation functions.

These functions extract base state information from Newton environments.
Newton uses warp arrays internally, which are converted to torch tensors.

Newton joint_q format (per world):
    [x, y, z, qx, qy, qz, qw, j0, j1, ...]
    - positions 0-2: base position
    - positions 3-6: base quaternion (x, y, z, w format)
    - positions 7+: joint positions

Newton joint_qd format (per world):
    [vx, vy, vz, wx, wy, wz, dj0, dj1, ...]
    - positions 0-2: base linear velocity (world frame)
    - positions 3-5: base angular velocity (world frame)
    - positions 6+: joint velocities
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from genesis.utils.geom import quat_to_xyz
from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_height_with_contact
from rlworld.rl.envs.utils.utils import EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def _get_newton_state_tensors(env: "NewtonEnv"):
    """Helper to get joint_q and joint_qd as reshaped torch tensors."""
    scene_manager = env.scene_manager
    state = scene_manager.state
    model = scene_manager.model
    num_worlds = model.world_count

    joint_q = wp.to_torch(state.joint_q)
    joint_qd = wp.to_torch(state.joint_qd)

    # Calculate coords/dofs per world from actual tensor sizes
    # This is more robust than using model attributes which may not match
    coords_per_world = joint_q.numel() // num_worlds
    dofs_per_world = joint_qd.numel() // num_worlds

    joint_q = joint_q.reshape(num_worlds, coords_per_world)
    joint_qd = joint_qd.reshape(num_worlds, dofs_per_world)

    return joint_q, joint_qd


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by inverse of quaternion.

    Args:
        q: Quaternion in (x, y, z, w) format, shape [..., 4]
        v: Vector to rotate, shape [..., 3]

    Returns:
        Rotated vector, shape [..., 3]
    """
    q_w = q[..., 3:4]
    q_vec = q[..., :3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0

    return a - b + c


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion.

    Args:
        q: Quaternion in (x, y, z, w) format, shape [..., 4]
        v: Vector to rotate, shape [..., 3]

    Returns:
        Rotated vector, shape [..., 3]
    """
    q_w = q[..., 3:4]
    q_vec = q[..., :3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0

    return a + b + c


@EnvStepCache()
def base_pos(env: "NewtonEnv") -> torch.Tensor:
    """Get base position in world frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    joint_q, _ = _get_newton_state_tensors(env)
    return joint_q[:, :3]


@EnvStepCache()
def base_quat(env: "NewtonEnv") -> torch.Tensor:
    """Get base quaternion (x, y, z, w format).

    Returns:
        Tensor of shape [num_envs, 4]
    """
    joint_q, _ = _get_newton_state_tensors(env)
    return joint_q[:, 3:7]


@EnvStepCache()
def base_euler(env: "NewtonEnv", rpy: bool = True, degrees: bool = False) -> torch.Tensor:
    """Get base orientation as Euler angles.

    Args:
        env: Newton environment.
        rpy: If True, return (roll, pitch, yaw).
        degrees: If True, return angles in degrees.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    quat_xyzw = base_quat(env)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return quat_to_xyz(quat_wxyz, rpy=rpy, degrees=degrees)


@EnvStepCache()
def base_height(env: "NewtonEnv") -> torch.Tensor:
    """Get base height (z-coordinate).

    Returns:
        Tensor of shape [num_envs, 1]
    """
    return base_pos(env)[:, 2:3]


@EnvStepCache()
def base_lin_vel(env: "NewtonEnv") -> torch.Tensor:
    """Get base linear velocity in body frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    joint_q, joint_qd = _get_newton_state_tensors(env)

    quat = joint_q[:, 3:7]  # (x, y, z, w)
    lin_vel_world = joint_qd[:, :3]

    return _quat_rotate_inverse(quat, lin_vel_world)


@EnvStepCache()
def base_ang_vel(env: "NewtonEnv") -> torch.Tensor:
    """Get base angular velocity in body frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    joint_q, joint_qd = _get_newton_state_tensors(env)

    quat = joint_q[:, 3:7]  # (x, y, z, w)
    ang_vel_world = joint_qd[:, 3:6]

    return _quat_rotate_inverse(quat, ang_vel_world)


@EnvStepCache()
def base_lin_vel_world(env: "NewtonEnv") -> torch.Tensor:
    """Get base linear velocity in world frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    _, joint_qd = _get_newton_state_tensors(env)
    return joint_qd[:, :3]


@EnvStepCache()
def base_ang_vel_world(env: "NewtonEnv") -> torch.Tensor:
    """Get base angular velocity in world frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    _, joint_qd = _get_newton_state_tensors(env)
    return joint_qd[:, 3:6]


@EnvStepCache()
def feet_air_time(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Get air time for each foot.

    Returns:
        Tensor of shape [num_envs, num_feet]
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    return env.contact_manager.last_air_time[:, result.contact_indices]


@EnvStepCache()
def feet_contact_indicator(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Get binary contact indicator for each foot.

    Returns:
        Tensor of shape [num_envs, num_feet] (float: 0.0 or 1.0)
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    return env.contact_manager.is_contact[:, result.contact_indices].float()


@EnvStepCache()
def feet_height(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Get height of each foot.

    Returns:
        Tensor of shape [num_envs, num_feet]
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    return result.data


@EnvStepCache()
def feet_contact_force(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Get contact force magnitude for each foot.

    Returns:
        Tensor of shape [num_envs, num_feet]
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    force = env.contact_manager.contact_force[:, result.contact_indices]  # (num_envs, num_feet, 3)
    return torch.norm(force, dim=-1)


@EnvStepCache()
def feet_contact_force_3d(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Get 3D contact force vector for each foot (flattened).

    Returns:
        Tensor of shape [num_envs, num_feet * 3]
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    force = env.contact_manager.contact_force[:, result.contact_indices]  # (num_envs, num_feet, 3)
    return force.reshape(env.num_envs, -1)
