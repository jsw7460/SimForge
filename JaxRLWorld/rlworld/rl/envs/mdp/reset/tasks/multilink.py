from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.engine.entities import RigidEntity

if TYPE_CHECKING:
    from rlworld.rl.envs import RLEnv


def randomize_all_links_mass(
    env: RLEnv,
    env_ids: torch.Tensor,
    mass_ratio_range: tuple[float, float] = (0.5, 2.0),
    entity_name: str = "robot",
):
    """
    Randomize mass of all links independently.

    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices to apply mass randomization.
        mass_ratio_range: Tuple of (min_ratio, max_ratio) for mass scaling.
        entity_name: Name of the robot entity.
    """
    entity: RigidEntity = env.scene_manager[entity_name]
    n_envs = len(env_ids)
    n_links = entity.n_links

    # Sample random mass ratios for each link independently
    # Shape: (n_envs, n_links)
    mass_ratios = (
        torch.rand(n_envs, n_links, device=env_ids.device)
        * (mass_ratio_range[1] - mass_ratio_range[0])
        + mass_ratio_range[0]
    )

    # Get original masses for all links
    all_link_indices = list(range(n_links))
    original_masses = entity.get_links_inertial_mass(
        links_idx_local=all_link_indices
    )  # (n_links,) or (n_envs, n_links)

    # Calculate mass shifts
    # mass_shift = original_mass * (ratio - 1)
    mass_shifts = original_masses * (mass_ratios - 1.0)  # (n_envs, n_links)

    entity.set_mass_shift(
        mass_shift=mass_shifts,
        links_idx_local=all_link_indices,
        envs_idx=env_ids,
    )


def initialize_dof_pos_random(
    env: RLEnv,
    env_ids: torch.Tensor,
    entity_name: str = "robot",
    first_dof_range: tuple[float, float] = (-1.5, 1.5),
    other_dof_range: tuple[float, float] = (-0.175, 0.175),  # ~10 degrees in radians
    zero_velocity: bool = True
):
    """
    Initialize joint positions with random values.

    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices.
        entity_name: Name of the robot entity.
        first_dof_range: Tuple of (min, max) for first joint position sampling.
        other_dof_range: Tuple of (min, max) for other joint positions sampling.
        zero_velocity: Whether to zero velocities after setting positions.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager[entity_name]
    n_envs = len(env_ids)
    n_dofs = len(env.act_manager.actuated_dof_ids)

    # Sample random joint positions
    dof_pos = torch.empty(n_envs, n_dofs, device=env.device)

    # First DOF: larger range
    dof_pos[:, 0] = (
        torch.rand(n_envs, device=env.device)
        * (first_dof_range[1] - first_dof_range[0])
        + first_dof_range[0]
    )

    dof_pos[:, 1:] = (
        torch.rand(n_envs, n_dofs - 1, device=env.device)
        * (other_dof_range[1] - other_dof_range[0])
        + other_dof_range[0]
    )

    robot.set_dofs_position(
        position=dof_pos,
        dofs_idx_local=env.act_manager.actuated_dof_ids,
        zero_velocity=zero_velocity,
        envs_idx=env_ids
    )

    if zero_velocity:
        robot.zero_all_dofs_velocity(env_ids)