from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import newton as newton_lib

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def _sample_uniform(
    lower: torch.Tensor,
    upper: torch.Tensor,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(shape, device=device) + lower


def _quat_from_euler_xyz_xyzw(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor,
) -> torch.Tensor:
    """Euler angles (radians) to quaternion (xyzw — Newton convention)."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([x, y, z, w], dim=-1)  # xyzw


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication (xyzw convention)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=-1)


def reset_root_state_uniform(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset root state with uniform random perturbations (Newton).

    Matches the MuJoCo/Genesis version's interface.
    Newton joint_q uses xyzw quaternion convention.
    """
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    coords_per_world = model.joint_coord_count // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_q = wp.to_torch(state.joint_q).reshape(num_worlds, coords_per_world)
    joint_qd = wp.to_torch(state.joint_qd).reshape(num_worlds, dofs_per_world)

    n = len(env_ids)
    device = env.device

    # Sample pose perturbations
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

    # Position: add perturbation to current
    joint_q[env_ids, 0:3] += pose_samples[:, 0:3]

    # Orientation: multiply delta quat (xyzw)
    default_quat = joint_q[env_ids, 3:7].clone()
    delta_quat = _quat_from_euler_xyz_xyzw(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    joint_q[env_ids, 3:7] = _quat_mul_xyzw(default_quat, delta_quat)

    # Velocity perturbation
    if velocity_range:
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
        ranges = torch.tensor(range_list, device=device)
        vel_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)
        joint_qd[env_ids, 0:6] += vel_samples

    wp.copy(state.joint_q, wp.from_torch(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))

    indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
    newton_lib.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices)


def randomize_friction(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.2),
) -> None:
    """Randomize ground-contact friction by scaling shape_material_mu.

    Modifies the friction coefficient for all shapes in the specified
    environments and notifies the solver of the change.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        friction_range: (min, max) absolute friction coefficient range.
    """
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model

    num_worlds = model.world_count
    shape_count_total = len(wp.to_torch(model.shape_material_mu))
    shapes_per_world = shape_count_total // num_worlds
    import ipdb; ipdb.set_trace()
    mu = wp.to_torch(model.shape_material_mu).reshape(num_worlds, shapes_per_world)

    n = len(env_ids)
    random_mu = (
        torch.rand(n, shapes_per_world, device=env.device)
        * (friction_range[1] - friction_range[0])
        + friction_range[0]
    )
    mu[env_ids] = random_mu

    wp.copy(model.shape_material_mu, wp.from_torch(mu.flatten(), dtype=wp.float32))

    from newton.solvers import SolverNotifyFlags
    scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


def push_robot(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Push robot by adding velocity perturbation.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to apply push.
        velocity_range: Dict with keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
                       and tuple (min, max) values.
    """
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_qd = wp.to_torch(state.joint_qd).reshape(num_worlds, dofs_per_world)

    n_envs = len(env_ids)
    device = env.device

    # Linear velocity perturbation (indices 0, 1, 2)
    if "x" in velocity_range:
        joint_qd[env_ids, 0] += torch.empty(n_envs, device=device).uniform_(*velocity_range["x"])
    if "y" in velocity_range:
        joint_qd[env_ids, 1] += torch.empty(n_envs, device=device).uniform_(*velocity_range["y"])
    if "z" in velocity_range:
        joint_qd[env_ids, 2] += torch.empty(n_envs, device=device).uniform_(*velocity_range["z"])

    # Angular velocity perturbation (indices 3, 4, 5)
    if "roll" in velocity_range:
        joint_qd[env_ids, 3] += torch.empty(n_envs, device=device).uniform_(*velocity_range["roll"])
    if "pitch" in velocity_range:
        joint_qd[env_ids, 4] += torch.empty(n_envs, device=device).uniform_(*velocity_range["pitch"])
    if "yaw" in velocity_range:
        joint_qd[env_ids, 5] += torch.empty(n_envs, device=device).uniform_(*velocity_range["yaw"])

    wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))