from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.utils import EnvStepCache
from rlworld.rl.utils import entity_utils as eu

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


@EnvStepCache()
def base_pos(env: GenesisEnv):
    """
    :return: [num_envs, 3]
    """
    return env.robot.get_pos()


@EnvStepCache()
def base_quat(env: GenesisEnv):
    return env.robot.get_quat()


@EnvStepCache()
def base_height(env: GenesisEnv):
    return base_pos(env)[:, 2].unsqueeze(-1)


@EnvStepCache()
def base_lin_vel(env: GenesisEnv):
    # This might be moved to proprioception
    inv_base_quat = inv_quat(base_quat(env))
    return transform_by_quat(env.robot.get_vel(), inv_base_quat)


@EnvStepCache()
def feet_height(env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR):
    """World-frame z of the bodies selected by ``asset_cfg.body_names``. Shape (num_envs, num_feet)."""
    if asset_cfg.body_ids is None:
        raise ValueError("feet_height requires asset_cfg with body_names (got none).")
    return env.get_robot_data(asset_cfg.name).body_pos_w_by_ids(asset_cfg.body_ids)[:, :, 2]


@EnvStepCache()
def base_euler(env: GenesisEnv, rpy: bool = False, degrees: bool = False):
    return quat_to_xyz(base_quat(env), rpy=rpy, degrees=degrees)


@EnvStepCache()
def foot_air_time(env: GenesisEnv, contact_group: str = "feet_ground_contact"):
    current_air_time = env.contact_manager.current_air_time(contact_group)
    return current_air_time


@EnvStepCache()
def contact_indicator(
    env: GenesisEnv,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Binary contact indicator. Shape: (num_envs, N)."""
    return env.contact_manager.is_contact(contact_group).float()


@EnvStepCache()
def contact_force(
    env: GenesisEnv,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Per-link contact force magnitude. Shape: (num_envs, N)."""
    forces_3d = env.contact_manager.contact_force(contact_group)  # (num_envs, N, 3)
    return torch.norm(forces_3d, dim=-1)  # (num_envs, N)


@EnvStepCache()
def contact_force_3d(
    env: GenesisEnv,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Per-link 3D contact force (log-scaled, flattened). Shape: (num_envs, N*3)."""
    forces_3d = env.contact_manager.contact_force(contact_group)  # (num_envs, N, 3)
    flat = forces_3d.flatten(start_dim=1)  # (num_envs, N*3)
    return torch.sign(flat) * torch.log1p(torch.abs(flat))


@EnvStepCache()
def links_acc(
    env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, links: tuple[str, ...] | None = None
) -> torch.Tensor:
    entity = env.scene_manager[asset_cfg.name]
    if links is None:
        links_idx_local = None
    else:
        links_idx_local, _ = eu.find_links(entity, links, global_ids=False)

    links_acc = entity.get_links_acc(links_idx_local=links_idx_local)
    return links_acc.reshape(env.num_envs, -1)


@EnvStepCache()
def dof_force(
    env: GenesisEnv,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
    dof_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """
    Output: [num_envs, n_dofs]
    """
    entity = env.scene_manager[asset_cfg.name]
    if dof_names is None:
        dofs_idx_local = None
    else:
        dofs_idx_local, _ = eu.find_dofs(entity, list(dof_names))

    return entity.get_dofs_force(dofs_idx_local=dofs_idx_local)


@EnvStepCache()
def actuated_dof_force(env: GenesisEnv) -> torch.Tensor:
    """
    Output: [num_envs, num_actuated_dofs]
    """
    return env.robot.get_dofs_force(dofs_idx_local=env.act_manager.actuated_dof_ids)
