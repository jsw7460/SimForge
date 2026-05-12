from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.utils import EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv, LocomotionEnv


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


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
        dofs_idx_local = env.act_manager.actuated_dof_ids  # To remove base 6-dof joint
    return env.robot.get_dofs_position(dofs_idx_local)


@EnvStepCache()
def dof_pos_nominal_difference(env: GenesisEnv) -> torch.Tensor:
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def dof_vel(
    env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, dofs_idx_local: torch.Tensor | None = None
) -> torch.Tensor:
    if dofs_idx_local is None:
        dofs_idx_local = env.act_manager.actuated_dof_ids  # To remove base 6-dof joint
    return env.scene_manager[asset_cfg.name].get_dofs_velocity(dofs_idx_local)


@EnvStepCache()
def raw_actions(env: GenesisEnv):
    return env.act_manager.raw_actions


@EnvStepCache()
def prev_processed_actions(env: GenesisEnv):
    return env.act_manager.processed_actions.clone()


@EnvStepCache()
def imu(env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, base_name: str = "base"):
    sensor = env.scene_manager.sensors[asset_cfg.name][base_name]["IMUSensor"]

    value = sensor.read()
    lin_acc = value.lin_acc
    ang_vel = value.ang_vel
    return torch.concat((lin_acc, ang_vel), dim=-1)


@EnvStepCache()
def imu_lin_acc(env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, base_name: str = "base"):
    imu_val = imu(env, asset_cfg, base_name)
    return imu_val[..., :3]


@EnvStepCache()
def imu_ang_vel(env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, base_name: str = "base"):
    imu_val = imu(env, asset_cfg, base_name)
    return imu_val[..., 3:]


@EnvStepCache()
def gait_phase_encoding(env: LocomotionEnv):
    return env.gait_manager.get_phase_encoding()
