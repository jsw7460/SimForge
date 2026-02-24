from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


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