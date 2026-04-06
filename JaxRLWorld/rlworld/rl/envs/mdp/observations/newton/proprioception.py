"""Newton proprioception observation functions.

These functions extract proprioceptive information from Newton environments,
including gravity projection, joint positions/velocities, and actions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.utils import EnvStepCache
from .state import base_quat, _quat_rotate_inverse
from .body_utils import get_bodies_pos

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv, NewtonLocomotionEnv


def projected_gravity(env: "NewtonEnv") -> torch.Tensor:
    quat = base_quat(env)
    if not hasattr(env, '_gravity_world_cache'):
        env._gravity_world_cache = torch.tensor(
            [[0.0, 0.0, -1.0]],
            device=env.device,
            dtype=torch.float32
        ).expand(env.num_envs, -1).contiguous()

    gravity_world = env._gravity_world_cache

    result = _quat_rotate_inverse(quat, gravity_world)
    return result


@EnvStepCache()
def dof_pos(env: "NewtonEnv") -> torch.Tensor:
    """Get actuated joint positions.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    accessor = env.scene_manager.robot_state
    dof_q = accessor.dof_positions(env.scene_manager.state)
    return dof_q[:, env.act_manager.actuated_q_indices]


@EnvStepCache()
def dof_pos_nominal_difference(env: "NewtonEnv") -> torch.Tensor:
    """Get joint positions relative to nominal (default) positions.

    Returns:
        Tensor of shape [num_envs, num_joints]
    """
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def dof_vel(env: "NewtonEnv") -> torch.Tensor:
    """Get actuated joint velocities.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    accessor = env.scene_manager.robot_state
    dof_qd = accessor.dof_velocities(env.scene_manager.state)
    return dof_qd[:, env.act_manager.actuated_qd_indices]


@EnvStepCache()
def raw_actions(env: "NewtonEnv") -> torch.Tensor:
    """Get raw (unprocessed) actions from current step.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.raw_actions


@EnvStepCache()
def prev_processed_actions(env: "NewtonEnv") -> torch.Tensor:
    """Get processed actions from previous step.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.processed_actions.clone()


@EnvStepCache()
def dof_pos_nominal_difference(env: "NewtonEnv") -> torch.Tensor:
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def relative_bodies_pos(
    env: "NewtonEnv",
    bodies: str | list[str],
    base_body: str = "torso_link",
) -> torch.Tensor:
    """Get body positions relative to base in body frame.

    Args:
        env: Newton environment.
        bodies: Body name pattern(s).
        base_body: Name of the base body.

    Returns:
        Tensor of shape (num_envs, num_bodies * 3).
    """
    result = get_bodies_pos(env, bodies)
    bodies_pos = result.data  # (num_envs, num_bodies, 3)

    base_result = get_bodies_pos(env, base_body)
    base_pos = base_result.data[:, 0, :]  # (num_envs, 3)

    quat = base_quat(env)  # (num_envs, 4)

    # Relative position in world frame
    rel_pos_world = bodies_pos - base_pos.unsqueeze(1)  # (num_envs, num_bodies, 3)

    # Transform to body frame
    rel_pos_body = _quat_rotate_inverse(
        quat.unsqueeze(1),  # (num_envs, 1, 4)
        rel_pos_world  # (num_envs, num_bodies, 3)
    )  # (num_envs, num_bodies, 3)

    return rel_pos_body.reshape(env.num_envs, -1)


@EnvStepCache()
def gait_phase_encoding(env: "NewtonLocomotionEnv"):
    return env.gait_manager.get_phase_encoding()