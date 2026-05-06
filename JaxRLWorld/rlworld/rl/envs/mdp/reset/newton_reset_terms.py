"""Newton-specific state initialization/reset functions.

These functions follow the same signature as Genesis reset functions:
    func(env: NewtonEnv, env_ids: torch.Tensor, **params)

They can be used with StateInitializationTermConfig to compose
flexible initialization behaviors.

Example:
    from rlworld.rl.envs.mdp.reset.newton_reset_terms import (
        initialize_base_pose,
        initialize_dof_pos,
    )
    from rlworld.rl.configs import StateInitializationTermConfig

    config = NewtonStateInitConfig(
        initialization_terms=[
            StateInitializationTermConfig(func=initialize_base_pose),
            StateInitializationTermConfig(
                func=initialize_dof_pos,
                params={"noise_range": (-0.1, 0.1)}
            ),
        ],
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import newton
from rlworld.rl.envs.mdp.observations.newton.body_utils import get_cache
if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def initialize_base_pose(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    base_init_pos: list[float],
    base_init_quat: list[float],
    zero_velocity: bool = True,
) -> None:
    """Initialize base position and orientation.

    Args:
        env: Newton environment instance
        env_ids: Indices of environments to reset
        base_init_pos: [x, y, z] position (default: [0, 0, 0.42])
        base_init_quat: [x, y, z, w] quaternion (default: [0, 0, 0, 1])
        zero_velocity: Whether to zero base velocities
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
    base_pos = torch.tensor(base_init_pos, device=env.device)
    base_quat = torch.tensor(base_init_quat, device=env.device)

    joint_q[env_ids, 0:3] = base_pos
    joint_q[env_ids, 3:7] = base_quat

    if zero_velocity:
        joint_qd[env_ids, 0:3] = 0.0
        joint_qd[env_ids, 3:6] = 0.0

    wp.copy(state.joint_q, wp.from_torch(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))

    indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices)


def initialize_dof_pos(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    zero_velocity: bool = True,
    noise_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Initialize joint positions from action manager offset with optional noise.

    Args:
        env: Newton environment instance
        env_ids: Indices of environments to reset
        zero_velocity: Whether to zero joint velocities
        noise_range: (min, max) uniform noise to add to joint positions
    """
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    coords_per_world = model.joint_coord_count // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    # Get joint_q and joint_qd as torch tensors
    joint_q = wp.to_torch(state.joint_q).reshape(num_worlds, coords_per_world)
    joint_qd = wp.to_torch(state.joint_qd).reshape(num_worlds, dofs_per_world)

    # Get default joint positions from action manager
    default_dof_pos = env.act_manager.offset[env_ids].clone()  # [n_reset, num_joints]

    # Add noise if specified
    if noise_range != (0.0, 0.0):
        noise = torch.empty_like(default_dof_pos).uniform_(
            noise_range[0], noise_range[1]
        )
        default_dof_pos = default_dof_pos + noise

    # Build mapping from joint name to joint_q index
    joint_q_start = wp.to_torch(model.joint_q_start).cpu().numpy()
    joints_per_world = len(model.joint_label) // num_worlds
    all_joint_names = list(model.joint_label)[:joints_per_world]
    name_to_q_idx = {name: int(joint_q_start[i]) for i, name in enumerate(all_joint_names)}

    # Build mapping from joint name to joint_qd index (DOF index)
    joint_qd_start = wp.to_torch(model.joint_qd_start).cpu().numpy()
    name_to_qd_idx = {name: int(joint_qd_start[i]) for i, name in enumerate(all_joint_names)}

    # Map offset values to correct joint_q positions
    act_names = env.act_manager.actuated_joint_names
    for i, name in enumerate(act_names):
        q_idx = name_to_q_idx[name]
        joint_q[env_ids, q_idx] = default_dof_pos[:, i]

        if zero_velocity:
            qd_idx = name_to_qd_idx[name]
            joint_qd[env_ids, qd_idx] = 0.0

    # Copy back to warp arrays
    wp.copy(state.joint_q, wp.from_torch(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))

    # Re-evaluate forward kinematics for only the reset environments
    indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices)


def initialize_dof_pos_with_noise(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    position_noise_range: tuple[float, float] = (0.0, 0.0),
    velocity_noise_range: tuple[float, float] = (0.0, 0.0),
) -> None:
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

    dof_pos = env.act_manager.offset[env_ids].clone()
    if position_noise_range != (0.0, 0.0):
        noise = torch.empty_like(dof_pos).uniform_(position_noise_range[0], position_noise_range[1])
        dof_pos = dof_pos + noise

    # Cache indices (build once, reuse)
    if not hasattr(env, '_dof_init_q_indices'):
        joint_q_start = wp.to_torch(model.joint_q_start).cpu().numpy()
        joint_qd_start = wp.to_torch(model.joint_qd_start).cpu().numpy()
        joints_per_world = len(model.joint_label) // num_worlds
        all_joint_names = list(model.joint_label)[:joints_per_world]
        name_to_q_idx = {name: int(joint_q_start[i]) for i, name in enumerate(all_joint_names)}
        name_to_qd_idx = {name: int(joint_qd_start[i]) for i, name in enumerate(all_joint_names)}

        act_names = env.act_manager.actuated_joint_names
        env._dof_init_q_indices = torch.tensor(
            [name_to_q_idx[name] for name in act_names],
            device=env.device, dtype=torch.long
        )
        env._dof_init_qd_indices = torch.tensor(
            [name_to_qd_idx[name] for name in act_names],
            device=env.device, dtype=torch.long
        )

    q_indices = env._dof_init_q_indices
    qd_indices = env._dof_init_qd_indices
    zero_velocity = velocity_noise_range == (0.0, 0.0)

    # Vectorized assignment using advanced indexing
    joint_q[env_ids.unsqueeze(1), q_indices.unsqueeze(0)] = dof_pos

    if zero_velocity:
        joint_qd[env_ids.unsqueeze(1), qd_indices.unsqueeze(0)] = 0.0
    else:
        dof_vel = torch.empty(len(env_ids), len(q_indices), device=env.device).uniform_(
            velocity_noise_range[0], velocity_noise_range[1]
        )
        joint_qd[env_ids.unsqueeze(1), qd_indices.unsqueeze(0)] = dof_vel

    # Copy back
    wp.copy(state.joint_q, wp.from_torch(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp.from_torch(joint_qd.flatten(), dtype=wp.float32))

    # FK
    indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=indices)


from newton.solvers import SolverNotifyFlags

def randomize_body_mass(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    body_patterns: str | list[str],
    mass_ratio_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize body mass for specified environments."""
    if len(env_ids) == 0:
        return
    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_patterns)

    body_mass = wp.to_torch(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)
    original_mass = cache.original_body_mass[body_indices]

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    ratios = torch.empty(n_envs, n_bodies, device=env.device).uniform_(
        mass_ratio_range[0], mass_ratio_range[1]
    )

    body_mass[env_ids.unsqueeze(1), body_indices] = original_mass * ratios

    wp.copy(model.body_mass, wp.from_torch(body_mass.flatten(), dtype=wp.float32))

    # Notify solver to recompute derived quantities
    solver = env.scene_manager.solver
    solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

