from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.mdp.observations.genesis import state, proprioception
from rlworld.rl.utils import entity_utils as eu
from rlworld.rl.utils import string as string_utils


if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv, LocomotionEnv


def reward_alive(env: GenesisEnv) -> torch.Tensor:
    """Reward for staying alive.

    Returns:
        Constant 1.0 for all environments.
    """
    return torch.ones(env.num_envs, device=env.device)


def penalize_dof_pos_limits(
    env: GenesisEnv,
    entity_name: str = "robot"
) -> torch.Tensor:
    """Penalize joint positions too close to the limit."""
    entity = env.scene_manager[entity_name]

    # Get only actuated DOFs
    actuated_ids = env.act_manager.actuated_dof_ids

    dof_pos = entity.get_dofs_position(dofs_idx_local=actuated_ids)
    dof_lower, dof_upper = entity.get_dofs_limit(dofs_idx_local=actuated_ids)

    # Penalize dof positions too close to the limit
    out_of_limits = -(dof_pos - dof_lower).clamp(max=0.0)  # lower limit
    out_of_limits += (dof_pos - dof_upper).clamp(min=0.0)  # upper limit

    return -torch.sum(out_of_limits, dim=1)


def penalize_hip_pos(
    env: GenesisEnv,
    hip_joints: list[str],
    entity_name: str = "robot"
) -> torch.Tensor:
    """Penalize hip joint positions deviating from default position.

    Encourages hip roll/yaw joints to stay near default pose.

    Args:
        env: Genesis environment.
        hip_joints: List of hip joint names to penalize (e.g., [".*hip_roll.*", ".*hip_yaw.*"]).
        entity_name: Name of the robot entity.

    Returns:
        Penalty for each environment.
    """

    indices, _ = string_utils.resolve_matching_names(
        hip_joints,
        env.act_manager._actuated_joint_names
    )

    dof_pos = proprioception.dof_pos(env)
    hip_pos = dof_pos[:, indices]
    default_pos = env.act_manager.offset[:, indices]

    return -torch.sum(torch.square(hip_pos - default_pos), dim=1)


def penalize_feet_swing_height(
    env: "LocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalize feet height error from smooth trajectory during swing phase."""
    entity = env.scene_manager["robot"]
    foot_names = env.gait_manager.foot_names

    # Height: preserve_order=True
    links_idx_local, _ = eu.find_links(
        entity, list(foot_names), global_ids=False, preserve_order=True
    )
    feet_pos = entity.get_links_pos(links_idx_local=links_idx_local)
    feet_height = feet_pos[..., 2]

    target_height = env.gait_manager.get_target_foot_height(max_height, profile)

    # Contact from named group
    is_contact = env.contact_manager.is_contact(contact_group)
    is_swing = ~is_contact

    height_error = torch.square(feet_height - target_height) * is_swing.float()

    return -torch.sum(height_error, dim=-1)


def penalize_feet_swing_height_gait(
    env: "LocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
) -> torch.Tensor:
    """Penalize feet height error during commanded swing phase.

    Applies penalty only when gait manager commands swing, not based on actual contact.
    """
    entity = env.scene_manager["robot"]
    foot_names = env.gait_manager.foot_names

    links_idx_local, _ = eu.find_links(
        entity, list(foot_names), global_ids=False, preserve_order=True
    )
    feet_pos = entity.get_links_pos(links_idx_local=links_idx_local)
    feet_height = feet_pos[..., 2]

    target_height = env.gait_manager.get_target_foot_height(max_height, profile)
    swing_mask = env.gait_manager.get_swing_mask()

    height_error = torch.square(feet_height - target_height) * swing_mask.float()

    return -torch.sum(height_error, dim=-1)