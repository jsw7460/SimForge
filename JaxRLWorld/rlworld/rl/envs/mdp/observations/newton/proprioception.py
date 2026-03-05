"""Newton proprioception observation functions.

These functions extract proprioceptive information from Newton environments,
including gravity projection, joint positions/velocities, and actions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.utils import EnvStepCache
from .state import base_quat, _quat_rotate_inverse, _get_newton_state_tensors
from .body_utils import get_bodies_pos

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv, NewtonLocomotionEnv


def projected_gravity(env: "NewtonEnv") -> jax.Array:
    quat = base_quat(env)
    if not hasattr(env, '_gravity_world_cache_jax'):
        env._gravity_world_cache_jax = jnp.broadcast_to(
            jnp.array([[0.0, 0.0, -1.0]]),
            (env.num_envs, 3),
        )

    gravity_world = env._gravity_world_cache_jax
    result = _quat_rotate_inverse(quat, gravity_world)
    return result


@EnvStepCache()
def dof_pos(env: "NewtonEnv") -> jax.Array:
    """Get actuated joint positions.

    Returns:
        Array of shape [num_envs, num_actions]
    """
    joint_q, _ = _get_newton_state_tensors(env)
    return joint_q[:, env.act_manager.actuated_q_indices]


@EnvStepCache()
def dof_pos_nominal_difference(env: "NewtonEnv") -> jax.Array:
    """Get joint positions relative to nominal (default) positions.

    Returns:
        Array of shape [num_envs, num_joints]
    """
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def dof_vel(env: "NewtonEnv") -> jax.Array:
    """Get actuated joint velocities.

    Returns:
        Array of shape [num_envs, num_actions]
    """
    _, joint_qd = _get_newton_state_tensors(env)
    return joint_qd[:, env.act_manager.actuated_qd_indices]


@EnvStepCache()
def raw_actions(env: "NewtonEnv") -> jax.Array:
    """Get raw (unprocessed) actions from current step.

    Returns:
        Array of shape [num_envs, num_actions]
    """
    return env.act_manager.raw_actions


@EnvStepCache()
def prev_processed_actions(env: "NewtonEnv") -> jax.Array:
    """Get processed actions from previous step.

    Returns:
        Array of shape [num_envs, num_actions]
    """
    return jnp.array(env.act_manager.processed_actions)


@EnvStepCache()
def relative_bodies_pos(
    env: "NewtonEnv",
    bodies: str | list[str],
    base_body: str = "torso_link",
) -> jax.Array:
    """Get body positions relative to base in body frame.

    Args:
        env: Newton environment.
        bodies: Body name pattern(s).
        base_body: Name of the base body.

    Returns:
        Array of shape (num_envs, num_bodies * 3).
    """
    result = get_bodies_pos(env, bodies)
    bodies_pos = result.data  # (num_envs, num_bodies, 3)

    base_result = get_bodies_pos(env, base_body)
    base_pos = base_result.data[:, 0, :]  # (num_envs, 3)

    quat = base_quat(env)  # (num_envs, 4)

    # Relative position in world frame
    rel_pos_world = bodies_pos - jnp.expand_dims(base_pos, 1)  # (num_envs, num_bodies, 3)

    # Transform to body frame
    rel_pos_body = _quat_rotate_inverse(
        jnp.expand_dims(quat, 1),  # (num_envs, 1, 4)
        rel_pos_world  # (num_envs, num_bodies, 3)
    )  # (num_envs, num_bodies, 3)

    return rel_pos_body.reshape(env.num_envs, -1)


@EnvStepCache()
def gait_phase_encoding(env: "NewtonLocomotionEnv"):
    return env.gait_manager.get_phase_encoding()
