from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv

from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.envs.mdp.observations.genesis import state
from rlworld.rl.utils import entity_utils as eu


def roll_pitch_violation(
    env: GenesisEnv, roll_threshold_degree: float = 15.0, pitch_threshold_degree: float = 15.0
) -> TerminationResult:
    """Check if robot's roll or pitch exceeds safe thresholds.

    Terminates episodes where the robot has tilted too much, indicating
    a fall or unstable behavior.

    Args:
        env: The locomotion environment.
        roll_threshold_degree: Maximum allowed roll angle in degrees.
        pitch_threshold_degree: Maximum allowed pitch angle in degrees.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments
        should be terminated due to roll/pitch violation.
    """
    # Get base orientation in Euler angles (roll, pitch, yaw)
    base_euler = state.base_euler(env, rpy=True, degrees=True)
    roll = base_euler[:, 0]
    pitch = base_euler[:, 1]

    # Check violations
    roll_violated = torch.abs(roll) > roll_threshold_degree
    pitch_violated = torch.abs(pitch) > pitch_threshold_degree
    return TerminationResult(roll_violated | pitch_violated)


def out_of_terrain_bounds(env: GenesisEnv, margin: float = 1.0) -> TerminationResult:
    """
    Terminate if robot goes outside terrain bounds.

    Args:
        env: The locomotion environment.
        margin: Distance from edge to trigger termination (meters)

    Returns:
        TerminationResult indicating which environments went out of bounds.
    """
    base_pos = state.base_pos(env)

    terrain = env.scene_manager["base_entity"]
    terrain_geom = terrain.geoms[0]
    height_field = terrain_geom.metadata["height_field"]
    terrain_morph = terrain.morph

    # Terrain boundaries
    horizontal_scale = terrain_morph.horizontal_scale
    terrain_size_x = height_field.shape[0] * horizontal_scale
    terrain_size_y = height_field.shape[1] * horizontal_scale

    # Check if out of bounds (with margin)
    out_of_bounds = (
        (base_pos[:, 0] < margin)
        | (base_pos[:, 0] > terrain_size_x - margin)
        | (base_pos[:, 1] < margin)
        | (base_pos[:, 1] > terrain_size_y - margin)
    )

    return TerminationResult(out_of_bounds)


def invalid_contact(env: GenesisEnv, contact_allowed_links: list[str]) -> TerminationResult:
    entity = env.scene_manager["robot"]

    # Get allowed link indices (global)
    allowed_ids, _ = eu.find_links(entity, contact_allowed_links, global_ids=True)
    allowed_ids_tensor = torch.tensor(allowed_ids, dtype=torch.int32, device=env.device)

    # Get all robot link indices
    all_robot_link_ids = torch.arange(entity.link_start, entity.link_end, dtype=torch.int32, device=env.device)

    # Get contact information
    contact_info = entity.get_contacts(exclude_self_contact=True)

    valid_mask = contact_info["valid_mask"]  # (n_envs, n_contacts)
    link_a = contact_info["link_a"]  # (n_envs, n_contacts)
    link_b = contact_info["link_b"]  # (n_envs, n_contacts)

    # Use torch.isin for efficient membership check
    is_robot_link_a = torch.isin(link_a, all_robot_link_ids)
    is_robot_link_b = torch.isin(link_b, all_robot_link_ids)

    link_a_allowed = torch.isin(link_a, allowed_ids_tensor)
    link_b_allowed = torch.isin(link_b, allowed_ids_tensor)

    # Invalid contact: valid & robot link & not allowed
    invalid_contact_a = valid_mask & is_robot_link_a & ~link_a_allowed
    invalid_contact_b = valid_mask & is_robot_link_b & ~link_b_allowed

    has_invalid_contact = torch.any(invalid_contact_a | invalid_contact_b, dim=1)
    return TerminationResult(has_invalid_contact)
