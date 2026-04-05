from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils import entity_utils as eu

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


def _sample_uniform(
    lower: torch.Tensor,
    upper: torch.Tensor,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(shape, device=device) + lower


def _quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor,
) -> torch.Tensor:
    """Euler angles (radians) to quaternion (wxyz). Batched."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication (wxyz convention)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def reset_root_state_uniform(
    env: "GenesisEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    entity_name: str = "robot",
) -> None:
    """Reset root state with uniform random perturbations (Genesis).

    Matches the MuJoCo version's interface: pose_range and velocity_range
    dicts with keys x, y, z, roll, pitch, yaw.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager[entity_name]
    n = len(env_ids)
    device = env.device

    # Sample pose perturbations
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

    # Default position/orientation from entity init_state config
    entity_cfg = env.scene_manager.config.entities[entity_name]
    init_state = entity_cfg.init_state
    default_pos = torch.tensor(init_state.pos, device=device).unsqueeze(0).expand(n, -1)
    default_quat = torch.tensor(init_state.rot, device=device).unsqueeze(0).expand(n, -1)

    positions = default_pos + pose_samples[:, 0:3]

    # Orientation: apply euler perturbation to default quat
    delta_quat = _quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = _quat_mul_wxyz(default_quat, delta_quat)

    robot.set_pos(positions, envs_idx=env_ids)
    robot.set_quat(orientations, envs_idx=env_ids)

    # Velocities: Genesis has no set_vel/set_ang, use set_dofs_velocity
    # Base DOFs are the first 6 (3 linear + 3 angular)
    if velocity_range:
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
        ranges = torch.tensor(range_list, device=device)
        vel_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

        robot.set_dofs_velocity(
            velocity=vel_samples,
            dofs_idx_local=list(range(6)),
            envs_idx=env_ids,
        )


def push_by_setting_velocity(
    env: "GenesisEnv",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    entity_name: str = "robot",
) -> None:
    """Push robot by adding velocity perturbation to the base DOFs (Genesis).

    Matches the MuJoCo/Newton push interface: adds random velocity offsets
    to the base 6 DOFs (linear xyz + angular rpy).

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to apply push.
        velocity_range: Dict with keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
                       and tuple (min, max) values.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager[entity_name]
    n = len(env_ids)
    device = env.device

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_delta = torch.zeros(n, 6, device=device)
    for i, key in enumerate(keys):
        lo, hi = velocity_range.get(key, (0.0, 0.0))
        vel_delta[:, i] = torch.empty(n, device=device).uniform_(lo, hi)

    # Read current base velocity (first 6 DOFs) and add perturbation
    current_vel = robot.get_dofs_velocity(dofs_idx_local=list(range(6)), envs_idx=env_ids)
    robot.set_dofs_velocity(
        velocity=current_vel + vel_delta,
        dofs_idx_local=list(range(6)),
        envs_idx=env_ids,
    )


def apply_external_force_torque(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]] | None = None,
    body_name: str = "base",
) -> None:
    """Apply random external force and torque to robot body.

    Args:
        env: The environment instance.
        env_ids: Environment indices to apply force to.
        force_range: Dict with 'x', 'y', 'z' keys, each mapping to (min, max) range.
        torque_range: Dict with 'x', 'y', 'z' keys, each mapping to (min, max) range.
            If None, no torque is applied.
        body_name: Name of the body to apply force to.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager["robot"]
    rigid_solver = env.scene.rigid_solver

    # Get global link index
    link_ids_global, _ = eu.find_links(robot, [body_name], global_ids=True)

    # Sample random forces: (n, 3)
    n = len(env_ids)
    forces = torch.zeros(n, 3, device=env.device)
    for i, key in enumerate(["x", "y", "z"]):
        lo, hi = force_range.get(key, (0.0, 0.0))
        forces[:, i] = torch.empty(n, device=env.device).uniform_(lo, hi)

    rigid_solver.apply_links_external_force(
        force=forces,
        links_idx=link_ids_global,
        envs_idx=env_ids.tolist(),
        ref="link_com",
        local=False,
    )

    # Apply torque if specified
    if torque_range is not None:
        torques = torch.zeros(n, 3, device=env.device)
        for i, key in enumerate(["x", "y", "z"]):
            lo, hi = torque_range.get(key, (0.0, 0.0))
            torques[:, i] = torch.empty(n, device=env.device).uniform_(lo, hi)

        rigid_solver.apply_links_external_torque(
            torque=torques,
            links_idx=link_ids_global,
            envs_idx=env_ids.tolist(),
            ref="link_com",
            local=False,
        )


def randomize_body_com_offset(
    env: "GenesisEnv",
    env_ids: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    link_names: tuple[str, ...] = ("torso_link",),
    entity_name: str = "robot",
) -> None:
    """Randomize body COM offset for specified links (Genesis).

    Uses Genesis's ``set_COM_shift`` API to apply additive COM offsets,
    matching MuJoCo's ``randomize_body_com_offset`` with ``operation="add"``.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        ranges: Per-axis (min, max) offset ranges. Keys are axis indices (0=x, 1=y, 2=z).
        link_names: Tuple of link names to randomize COM for.
        entity_name: Name of the robot entity.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(robot, list(link_names), global_ids=False)

    n_envs = len(env_ids)
    n_links = len(links_idx_local)

    com_shift = torch.zeros(n_envs, n_links, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        com_shift[:, :, axis] = torch.empty(n_envs, n_links, device=env.device).uniform_(lo, hi)

    robot.set_COM_shift(
        com_shift=com_shift,
        links_idx_local=links_idx_local,
        envs_idx=env_ids,
    )
