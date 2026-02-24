from __future__ import annotations

import torch
from genesis.utils.geom import transform_by_quat, inv_quat
from rlworld.rl.envs.utils import EnvStepCache
from rlworld.rl.utils import entity_utils as eu
from rlworld.rl.envs.mdp.observations.genesis import state
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv, LocomotionEnv


@EnvStepCache()
def projected_gravity(env: GenesisEnv) -> torch.Tensor:
    """
    :return: [num_envs, 1]
    """
    base_quat = env.robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    gravity = torch.tensor((0.0, 0.0, -9.81), device=env.device, dtype=inv_base_quat.dtype)
    gravity = gravity.repeat(env.num_envs, 1)
    return transform_by_quat(gravity, inv_base_quat)


@EnvStepCache()
def dof_pos(env: GenesisEnv, dofs_idx_local: torch.Tensor | None = None) -> torch.Tensor:
    """Get DOF positions for the robot.

        Args:
            env: Genesis environment.
            dofs_idx_local: DOF indices to query. If None, uses actuated DOF indices.

        Returns:
            DOF positions of shape (num_envs, num_dofs).
            When dofs_idx_local is None, ordering matches env.act_manager._actuated_joint_names.
    """
    if dofs_idx_local is None:
        dofs_idx_local = env.act_manager.actuated_dof_ids       # To remove base 6-dof joint
    return env.robot.get_dofs_position(dofs_idx_local)


@EnvStepCache()
def dof_pos_nominal_difference(env: GenesisEnv) -> torch.Tensor:
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def dof_vel(env: GenesisEnv, entity_name: str = "robot", dofs_idx_local: torch.Tensor | None = None) -> torch.Tensor:
    if dofs_idx_local is None:
        dofs_idx_local = env.act_manager.actuated_dof_ids       # To remove base 6-dof joint
    return env.scene_manager[entity_name].get_dofs_velocity(dofs_idx_local)


@EnvStepCache()
def raw_actions(env: GenesisEnv):
    return env.act_manager.raw_actions


@EnvStepCache()
def prev_processed_actions(env: GenesisEnv):
    return env.act_manager.processed_actions.clone()


@EnvStepCache()
def relative_links_pos(
    env: GenesisEnv,
    entity_name: str = "robot",
    base_name: str = "base",
    links: tuple[str] = None
):
    """Get link positions relative to base in body frame.

    Args:
        env: Genesis environment.
        entity_name: Name of the robot entity.
        base_name: Name of the base link.
        links: Link names to query.

    Returns:
        Tensor of shape (num_envs, num_links * 3).
    """
    entity = env.scene_manager[entity_name]

    if links is None:
        links_pos = entity.get_links_pos(links)
    else:
        links_ids, _ = eu.find_links(entity, links, global_ids=False)
        links_pos = entity.get_links_pos(links_ids)

    base_link = entity.get_link(base_name)
    base_pos = base_link.get_pos().unsqueeze(1)  # (num_envs, 1, 3)
    base_quat = state.base_quat(env)  # (num_envs, 4)

    # Relative position in world frame
    rel_pos_world = links_pos - base_pos  # (num_envs, num_links, 3)

    # Transform to body frame
    rel_pos_body = transform_by_quat(rel_pos_world, inv_quat(base_quat).unsqueeze(1))

    return rel_pos_body.reshape(env.num_envs, -1)


@EnvStepCache()
def imu(env: GenesisEnv, entity_name: str = "robot", base_name: str = "base"):
    sensor = env.scene_manager.sensors[entity_name][base_name]["IMUSensor"]

    value = sensor.read()
    lin_acc = value.lin_acc
    ang_vel = value.ang_vel
    return torch.concat((lin_acc, ang_vel), dim=-1)


@EnvStepCache()
def imu_lin_acc(env: GenesisEnv, entity_name: str = "robot", base_name: str = "base"):
    imu_val = imu(env, entity_name, base_name)
    return imu_val[..., :3]


@EnvStepCache()
def imu_ang_vel(env: GenesisEnv, entity_name: str = "robot", base_name: str = "base"):
    imu_val = imu(env, entity_name, base_name)
    return imu_val[..., 3:]


@EnvStepCache()
def gait_phase_encoding(env: LocomotionEnv):
    return env.gait_manager.get_phase_encoding()