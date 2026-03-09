from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.utils import EnvStepCache
from rlworld.rl.utils import entity_utils as eu
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


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
def feet_height(env: GenesisEnv, links: tuple[str, ...], entity_name: str = "robot"):
    entity = env.scene_manager[entity_name]

    if isinstance(links, str):
        links = [links]

    links_idx_local, _ = eu.find_links(entity, list(links), global_ids=False, preserve_order=True)
    foot_pos = entity.get_links_pos(links_idx_local=links_idx_local)  # (num_envs, num_feet, 3)
    foot_z = foot_pos[:, :, 2]
    return foot_z


@EnvStepCache()
def base_euler(env: GenesisEnv, rpy: bool = False, degrees: bool = False):
    return quat_to_xyz(base_quat(env), rpy=rpy, degrees=degrees)


@EnvStepCache()
def foot_air_time(env: GenesisEnv, links: tuple[str, ...], entity_name: str = "robot"):
    link_indices = env.contact_manager.get_link_indices(links, entity_name)

    current_air_time = env.contact_manager.current_air_time[:, link_indices]
    return current_air_time

@EnvStepCache()
def contact_indicator(
    env: GenesisEnv,
    entity_name: str = "robot",
    links: tuple[str, ...] | None = None,
    preserve_order: bool = False
) -> torch.Tensor:
    """
    Output: [num_envs, num_links]
    """
    entity = env.scene_manager[entity_name]
    if links is None:
        link_ids = list(range(entity.link_start, entity.link_end))
    else:
        link_ids, _ = eu.find_links(entity, list(links), global_ids=True, preserve_order=preserve_order)

    link_ids_tensor = torch.tensor(link_ids, dtype=torch.int32, device=env.device)

    contact_info = entity.get_contacts(exclude_self_contact=True)
    valid_mask = contact_info["valid_mask"]
    link_a = contact_info["link_a"]
    link_b = contact_info["link_b"]

    # Vectorized checking: [num_envs, num_contacts, num_links]
    link_a_match = link_a.unsqueeze(-1) == link_ids_tensor.unsqueeze(0).unsqueeze(0)
    link_b_match = link_b.unsqueeze(-1) == link_ids_tensor.unsqueeze(0).unsqueeze(0)

    # Check if each link has any contact
    valid_mask_expanded = valid_mask.unsqueeze(-1)  # [num_envs, num_contacts, 1]
    has_contact = torch.any(
        valid_mask_expanded & (link_a_match | link_b_match),
        dim=1  # Reduce over contacts dimension
    )  # [num_envs, num_links]

    return has_contact.float()

@EnvStepCache()
def contact_force(
    env: GenesisEnv,
    entity_name: str = "robot",
    links: tuple[str, ...] | None = None,
) -> torch.Tensor:
    if isinstance(links, str):
        links = [links]

    link_indices = env.contact_manager.get_link_indices(list(links), entity_name, preserve_order=True)
    link_names = env.contact_manager.get_link_names(link_indices)

    # Get contact forces
    sensors = env.scene_manager.sensors[entity_name]
    forces = torch.stack([
        torch.norm(sensors[name]["ContactForceSensor"].read(), dim=-1)
        for name in link_names
    ], dim=1)  # (num_envs, num_feet)

    return forces


@EnvStepCache()
def links_acc(env: GenesisEnv, entity_name: str = "robot", links: tuple[str, ...] | None = None) -> torch.Tensor:
    entity = env.scene_manager[entity_name]
    if links is None:
        links_idx_local = None
    else:
        links_idx_local, _ = eu.find_links(entity, links, global_ids=False)

    links_acc = entity.get_links_acc(links_idx_local=links_idx_local)
    return links_acc.reshape(env.num_envs, -1)


@EnvStepCache()
def dof_force(
    env: GenesisEnv,
    entity_name: str = "robot",
    dof_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """
    Output: [num_envs, n_dofs]
    """
    entity = env.scene_manager[entity_name]
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
    return env.robot.get_dofs_force(
        dofs_idx_local=env.act_manager.actuated_dof_ids
    )
