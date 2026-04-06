from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import newton as newton_lib
from newton.solvers import SolverNotifyFlags

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
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def reset_root_state_uniform(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset root state with uniform random perturbations (Newton).

    Uses RobotStateAccessor (ArticulationView) for state access.
    Newton joint_q uses xyzw quaternion convention.
    """
    if len(env_ids) == 0:
        return

    from rlworld.rl.envs.newton.robot_state_accessor import RobotStateAccessor

    scene_manager = env.scene_manager
    accessor = scene_manager.robot_state
    state = scene_manager.state_0

    n = len(env_ids)
    device = env.device
    mask = RobotStateAccessor.env_ids_to_mask(env_ids, scene_manager.model.world_count, device)

    # Sample pose perturbations
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

    # Default position/orientation from entity init_state config
    entity_cfg = scene_manager.config.entities["robot"]
    init_state = entity_cfg.init_state
    default_pos = torch.tensor(init_state.pos, device=device)
    default_rot = torch.tensor(init_state.rot, device=device)

    # Position: default + perturbation
    pos = accessor.root_pos_w(state).clone()
    pos[env_ids] = default_pos.unsqueeze(0) + pose_samples[:, 0:3]

    # Orientation: default quat * delta quat (xyzw)
    quat_xyzw = accessor.root_quat_xyzw(state).clone()
    default_quat = default_rot.unsqueeze(0).expand(n, -1)
    delta_quat = _quat_from_euler_xyz_xyzw(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    quat_xyzw[env_ids] = _quat_mul_xyzw(default_quat, delta_quat)

    # Velocity: zero out first to avoid stale values from previous episode
    lin_vel = accessor.root_lin_vel_w(state).clone()
    ang_vel = accessor.root_ang_vel_w(state).clone()
    lin_vel[env_ids] = 0.0
    ang_vel[env_ids] = 0.0
    if velocity_range:
        vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
        vel_ranges = torch.tensor(vel_range_list, device=device)
        vel_samples = _sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device)
        lin_vel[env_ids] += vel_samples[:, 0:3]
        ang_vel[env_ids] += vel_samples[:, 3:6]

    # Also zero out joint velocities (non-root DOFs)
    dof_vel = accessor.dof_velocities(state).clone()
    dof_vel[env_ids, :] = 0.0
    accessor.set_dof_velocities(state, dof_vel, mask=mask)

    # Set root state and evaluate FK
    accessor.set_root_state(state, pos, quat_xyzw, lin_vel, ang_vel, mask=mask)
    accessor.eval_fk(state, mask=mask)


def randomize_friction(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.2),
) -> None:
    """Randomize friction for the robot's shapes via ArticulationView.

    Uses the scene manager's ``robot_view`` to read/write only the robot's
    shape_material_mu, leaving global shapes (ground plane) untouched.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        friction_range: (min, max) absolute friction coefficient range.
    """
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    view = scene_manager.robot_view
    model = scene_manager.model

    # warp array → torch: (num_worlds, num_articulations, shapes_per_articulation)
    mu_wp = view.get_attribute("shape_material_mu", model)
    mu = wp.to_torch(mu_wp)

    n_shapes = mu.shape[-1]
    random_mu = (
        torch.rand(len(env_ids), mu.shape[1], n_shapes, device=env.device)
        * (friction_range[1] - friction_range[0])
        + friction_range[0]
    )
    mu[env_ids] = random_mu

    # torch → warp and write back
    view.set_attribute("shape_material_mu", model, mu)

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

    from rlworld.rl.envs.newton.robot_state_accessor import RobotStateAccessor

    accessor = env.scene_manager.robot_state
    state = env.scene_manager.state_0
    mask = RobotStateAccessor.env_ids_to_mask(env_ids, env.scene_manager.model.world_count, env.device)

    n_envs = len(env_ids)
    device = env.device

    lin_vel = accessor.root_lin_vel_w(state).clone()
    ang_vel = accessor.root_ang_vel_w(state).clone()

    # Linear velocity perturbation
    if "x" in velocity_range:
        lin_vel[env_ids, 0] += torch.empty(n_envs, device=device).uniform_(*velocity_range["x"])
    if "y" in velocity_range:
        lin_vel[env_ids, 1] += torch.empty(n_envs, device=device).uniform_(*velocity_range["y"])
    if "z" in velocity_range:
        lin_vel[env_ids, 2] += torch.empty(n_envs, device=device).uniform_(*velocity_range["z"])

    # Angular velocity perturbation
    if "roll" in velocity_range:
        ang_vel[env_ids, 0] += torch.empty(n_envs, device=device).uniform_(*velocity_range["roll"])
    if "pitch" in velocity_range:
        ang_vel[env_ids, 1] += torch.empty(n_envs, device=device).uniform_(*velocity_range["pitch"])
    if "yaw" in velocity_range:
        ang_vel[env_ids, 2] += torch.empty(n_envs, device=device).uniform_(*velocity_range["yaw"])

    # Write back via accessor (only velocities, no transforms change)
    pos = accessor.root_pos_w(state)
    quat_xyzw = accessor.root_quat_xyzw(state)
    accessor.set_root_state(state, pos, quat_xyzw, lin_vel, ang_vel, mask=mask)


def randomize_body_com_offset(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    body_patterns: str | list[str] = ("torso_link",),
) -> None:
    """Randomize body COM offset for specified bodies (Newton).

    Adds random offsets to the original body COM positions,
    matching MuJoCo's ``randomize_body_com_offset`` with ``operation="add"``.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        ranges: Per-axis (min, max) offset ranges. Keys are axis indices (0=x, 1=y, 2=z).
        body_patterns: Body name patterns to randomize COM for.
    """
    if len(env_ids) == 0:
        return

    from rlworld.rl.envs.mdp.observations.newton.body_utils import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_patterns)

    # body_com: wp.array(dtype=wp.vec3) → torch [total_bodies, 3]
    body_com = wp.to_torch(model.body_com).reshape(env.num_envs, cache.bodies_per_env, 3)

    # Cache original COM on first call
    if not hasattr(cache, '_original_body_com'):
        cache._original_body_com = body_com.clone()

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    original = cache._original_body_com[:, body_indices, :][env_ids]  # [n_envs, n_bodies, 3]

    offsets = torch.zeros(n_envs, n_bodies, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        offsets[:, :, axis] = torch.empty(n_envs, n_bodies, device=env.device).uniform_(lo, hi)

    body_com[env_ids.unsqueeze(1), body_indices] = original + offsets

    wp.copy(model.body_com, wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3))

    solver = env.scene_manager.solver
    solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)


# -- Backward-compatible re-exports from the new dr module -----------
# New code should import from ``rlworld.rl.envs.mdp.events.dr.newton``.
from rlworld.rl.envs.mdp.events.dr.newton import (  # noqa: E402, F401
    randomize_body_mass,
    randomize_joint_armature,
    randomize_joint_friction,
    randomize_pd_gains,
)
