"""Newton state observation functions.

Uses RobotStateAccessor (backed by ArticulationView) to access Newton state.
No manual warp tensor reshaping — all state access goes through the accessor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.utils.geom import quat_to_xyz
from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_height_with_contact
from rlworld.rl.envs.utils.utils import EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def _accessor(env: "NewtonEnv"):
    return env.scene_manager.robot_state


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by inverse of quaternion.

    Args:
        q: Quaternion in (x, y, z, w) format, shape [..., 4]
        v: Vector to rotate, shape [..., 3]
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
    """
    q_w = q[..., 3:4]
    q_vec = q[..., :3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0

    return a + b + c


@EnvStepCache()
def base_pos(env: "NewtonEnv") -> torch.Tensor:
    """Base position in world frame. Shape: [num_envs, 3]."""
    return _accessor(env).root_pos_w(env.scene_manager.state)


@EnvStepCache()
def base_quat(env: "NewtonEnv") -> torch.Tensor:
    """Base quaternion in (x, y, z, w) format. Shape: [num_envs, 4]."""
    return _accessor(env).root_quat_xyzw(env.scene_manager.state)


@EnvStepCache()
def base_euler(env: "NewtonEnv", rpy: bool = True, degrees: bool = False) -> torch.Tensor:
    """Base orientation as Euler angles. Shape: [num_envs, 3]."""
    quat_xyzw = base_quat(env)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return quat_to_xyz(quat_wxyz, rpy=rpy, degrees=degrees)


@EnvStepCache()
def base_height(env: "NewtonEnv") -> torch.Tensor:
    """Base height (z-coordinate). Shape: [num_envs, 1]."""
    return base_pos(env)[:, 2:3]


@EnvStepCache()
def base_lin_vel(env: "NewtonEnv") -> torch.Tensor:
    """Base linear velocity in body frame. Shape: [num_envs, 3]."""
    acc = _accessor(env)
    state = env.scene_manager.state
    quat_xyzw = acc.root_quat_xyzw(state)
    lin_vel_w = acc.root_lin_vel_w(state)
    return _quat_rotate_inverse(quat_xyzw, lin_vel_w)


@EnvStepCache()
def base_ang_vel(env: "NewtonEnv") -> torch.Tensor:
    """Base angular velocity in body frame. Shape: [num_envs, 3]."""
    acc = _accessor(env)
    state = env.scene_manager.state
    quat_xyzw = acc.root_quat_xyzw(state)
    ang_vel_w = acc.root_ang_vel_w(state)
    return _quat_rotate_inverse(quat_xyzw, ang_vel_w)


@EnvStepCache()
def base_lin_vel_world(env: "NewtonEnv") -> torch.Tensor:
    """Base linear velocity in world frame. Shape: [num_envs, 3]."""
    return _accessor(env).root_lin_vel_w(env.scene_manager.state)


@EnvStepCache()
def base_ang_vel_world(env: "NewtonEnv") -> torch.Tensor:
    """Base angular velocity in world frame. Shape: [num_envs, 3]."""
    return _accessor(env).root_ang_vel_w(env.scene_manager.state)


@EnvStepCache()
def feet_air_time(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Air time for each foot. Shape: [num_envs, num_feet]."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    return env.contact_manager.last_air_time("foot_contact")[:, result.contact_indices]


@EnvStepCache()
def feet_contact_indicator(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Binary contact indicator for each foot. Shape: [num_envs, num_feet]."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    return env.contact_manager.is_contact("foot_contact")[:, result.contact_indices].float()


@EnvStepCache()
def feet_height(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Height of each foot. Shape: [num_envs, num_feet]."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    return result.data


@EnvStepCache()
def feet_contact_force(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """Contact force magnitude for each foot. Shape: [num_envs, num_feet]."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    force = env.contact_manager.contact_force("foot_contact")[:, result.contact_indices]
    return torch.norm(force, dim=-1)


@EnvStepCache()
def feet_contact_force_3d(env: "NewtonEnv", feet_bodies: str | list[str]) -> torch.Tensor:
    """3D contact force vector for each foot (log-scaled, flattened). Shape: [num_envs, num_feet * 3]."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    force = env.contact_manager.contact_force("foot_contact")[:, result.contact_indices]
    flat = force.reshape(env.num_envs, -1)
    return torch.sign(flat) * torch.log1p(torch.abs(flat))
