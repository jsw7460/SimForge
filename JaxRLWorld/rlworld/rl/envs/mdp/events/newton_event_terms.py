from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax, wp_from_jax

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def push_robot(
    env: "NewtonEnv",
    env_ids,
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Push robot by adding velocity perturbation."""
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_qd = wp_to_jax(state.joint_qd).reshape(num_worlds, dofs_per_world)

    n_envs = len(env_ids)
    key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    # Linear velocity perturbation (indices 0, 1, 2)
    axis_map = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    for axis_name, axis_idx in axis_map.items():
        if axis_name in velocity_range:
            key, subkey = jax.random.split(key)
            perturbation = jax.random.uniform(
                subkey, shape=(n_envs,),
                minval=velocity_range[axis_name][0],
                maxval=velocity_range[axis_name][1],
            )
            joint_qd = joint_qd.at[env_ids, axis_idx].add(perturbation)

    wp.copy(state.joint_qd, wp_from_jax(joint_qd.flatten(), dtype=wp.float32))
