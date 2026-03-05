"""Newton-specific state initialization/reset functions (JAX-native).

These functions follow the same signature:
    func(env: NewtonEnv, env_ids, **params)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import newton
from rlworld.rl.envs.mdp.observations.newton.body_utils import get_cache
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax, wp_from_jax

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def _env_ids_to_wp(env_ids) -> wp.array:
    """Convert env_ids (jax/numpy/list) to wp.array(int32)."""
    ids_np = np.array(env_ids, dtype=np.int32)
    return wp.array(ids_np, dtype=wp.int32)


def initialize_base_pose(
    env: "NewtonEnv",
    env_ids,
    base_init_pos: list[float],
    base_init_quat: list[float],
    zero_velocity: bool = True,
) -> None:
    """Initialize base position and orientation."""
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    coords_per_world = model.joint_coord_count // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_q = wp_to_jax(state.joint_q).reshape(num_worlds, coords_per_world)
    joint_qd = wp_to_jax(state.joint_qd).reshape(num_worlds, dofs_per_world)

    base_pos = jnp.array(base_init_pos)
    base_quat = jnp.array(base_init_quat)

    joint_q = joint_q.at[env_ids, 0:3].set(base_pos)
    joint_q = joint_q.at[env_ids, 3:7].set(base_quat)

    if zero_velocity:
        joint_qd = joint_qd.at[env_ids, 0:3].set(0.0)
        joint_qd = joint_qd.at[env_ids, 3:6].set(0.0)

    wp.copy(state.joint_q, wp_from_jax(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp_from_jax(joint_qd.flatten(), dtype=wp.float32))

    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=_env_ids_to_wp(env_ids))


def initialize_dof_pos(
    env: "NewtonEnv",
    env_ids,
    zero_velocity: bool = True,
    noise_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Initialize joint positions from action manager offset with optional noise."""
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    coords_per_world = model.joint_coord_count // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_q = wp_to_jax(state.joint_q).reshape(num_worlds, coords_per_world)
    joint_qd = wp_to_jax(state.joint_qd).reshape(num_worlds, dofs_per_world)

    default_dof_pos = jnp.array(env.act_manager.offset[env_ids])

    if noise_range != (0.0, 0.0):
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        noise = jax.random.uniform(
            key, shape=default_dof_pos.shape,
            minval=noise_range[0], maxval=noise_range[1]
        )
        default_dof_pos = default_dof_pos + noise

    # Build mapping from joint name to joint_q index
    joint_q_start = np.array(wp_to_jax(model.joint_q_start))
    joints_per_world = len(model.joint_label) // num_worlds
    all_joint_names = list(model.joint_label)[:joints_per_world]
    name_to_q_idx = {name: int(joint_q_start[i]) for i, name in enumerate(all_joint_names)}

    joint_qd_start = np.array(wp_to_jax(model.joint_qd_start))
    name_to_qd_idx = {name: int(joint_qd_start[i]) for i, name in enumerate(all_joint_names)}

    act_names = env.act_manager.actuated_joint_names
    for i, name in enumerate(act_names):
        q_idx = name_to_q_idx[name]
        joint_q = joint_q.at[env_ids, q_idx].set(default_dof_pos[:, i])

        if zero_velocity:
            qd_idx = name_to_qd_idx[name]
            joint_qd = joint_qd.at[env_ids, qd_idx].set(0.0)

    wp.copy(state.joint_q, wp_from_jax(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp_from_jax(joint_qd.flatten(), dtype=wp.float32))

    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=_env_ids_to_wp(env_ids))


def initialize_dof_pos_with_noise(
    env: "NewtonEnv",
    env_ids,
    position_noise_range: tuple[float, float] = (0.0, 0.0),
    velocity_noise_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    if len(env_ids) == 0:
        return

    scene_manager = env.scene_manager
    model = scene_manager.model
    state = scene_manager.state_0

    num_worlds = model.world_count
    coords_per_world = model.joint_coord_count // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    joint_q = wp_to_jax(state.joint_q).reshape(num_worlds, coords_per_world)
    joint_qd = wp_to_jax(state.joint_qd).reshape(num_worlds, dofs_per_world)

    dof_pos = jnp.array(env.act_manager.offset[env_ids])
    if position_noise_range != (0.0, 0.0):
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        noise = jax.random.uniform(
            key, shape=dof_pos.shape,
            minval=position_noise_range[0], maxval=position_noise_range[1]
        )
        dof_pos = dof_pos + noise

    # Cache indices (build once, reuse)
    if not hasattr(env, '_dof_init_q_indices_jax'):
        joint_q_start = np.array(wp_to_jax(model.joint_q_start))
        joint_qd_start = np.array(wp_to_jax(model.joint_qd_start))
        joints_per_world = len(model.joint_label) // num_worlds
        all_joint_names = list(model.joint_label)[:joints_per_world]
        name_to_q_idx = {name: int(joint_q_start[i]) for i, name in enumerate(all_joint_names)}
        name_to_qd_idx = {name: int(joint_qd_start[i]) for i, name in enumerate(all_joint_names)}

        act_names = env.act_manager.actuated_joint_names
        env._dof_init_q_indices_jax = jnp.array(
            [name_to_q_idx[name] for name in act_names], dtype=jnp.int32
        )
        env._dof_init_qd_indices_jax = jnp.array(
            [name_to_qd_idx[name] for name in act_names], dtype=jnp.int32
        )

    q_indices = env._dof_init_q_indices_jax
    qd_indices = env._dof_init_qd_indices_jax
    zero_velocity = velocity_noise_range == (0.0, 0.0)

    # Vectorized assignment using advanced indexing
    env_ids_arr = jnp.array(env_ids)
    joint_q = joint_q.at[jnp.expand_dims(env_ids_arr, 1), jnp.expand_dims(q_indices, 0)].set(dof_pos)

    if zero_velocity:
        joint_qd = joint_qd.at[jnp.expand_dims(env_ids_arr, 1), jnp.expand_dims(qd_indices, 0)].set(0.0)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        dof_vel = jax.random.uniform(
            key, shape=(len(env_ids), len(q_indices)),
            minval=velocity_noise_range[0], maxval=velocity_noise_range[1]
        )
        joint_qd = joint_qd.at[jnp.expand_dims(env_ids_arr, 1), jnp.expand_dims(qd_indices, 0)].set(dof_vel)

    wp.copy(state.joint_q, wp_from_jax(joint_q.flatten(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp_from_jax(joint_qd.flatten(), dtype=wp.float32))

    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=_env_ids_to_wp(env_ids))


from newton.solvers import SolverNotifyFlags

def randomize_body_mass(
    env: "NewtonEnv",
    env_ids,
    body_patterns: str | list[str],
    mass_ratio_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize body mass for specified environments."""
    if len(env_ids) == 0:
        return
    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_patterns)

    body_mass = wp_to_jax(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)
    original_mass = cache.original_body_mass[body_indices]

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    key = jax.random.PRNGKey(np.random.randint(0, 2**31))
    ratios = jax.random.uniform(
        key, shape=(n_envs, n_bodies),
        minval=mass_ratio_range[0], maxval=mass_ratio_range[1]
    )

    env_ids_arr = jnp.array(env_ids)
    body_indices_arr = jnp.array(body_indices)
    body_mass = body_mass.at[
        jnp.expand_dims(env_ids_arr, 1), jnp.expand_dims(body_indices_arr, 0)
    ].set(original_mass * ratios)

    wp.copy(model.body_mass, wp_from_jax(body_mass.flatten(), dtype=wp.float32))

    solver = env.scene_manager.solver
    solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)
