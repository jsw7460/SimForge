from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.envs.utils import EnvStepCache
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


# ============================================================
# Query Results
# ============================================================

@dataclass
class BodiesResult:
    """Result of body query."""
    data: jax.Array
    body_names: list[str]
    body_indices: list[int]


@dataclass
class BodiesWithContactResult:
    """Result of body query with contact information."""
    data: jax.Array
    body_names: list[str]
    body_indices: list[int]
    contact_indices: list[int]


# ============================================================
# General Body Queries (no contact requirement)
# ============================================================

@EnvStepCache()
def get_body_q(env: "NewtonEnv") -> jax.Array:
    """Get all body transforms as (num_envs, bodies_per_env, 7) array."""
    cache = get_cache(env)
    state = env.scene_manager.state
    return wp_to_jax(state.body_q).reshape(env.num_envs, cache.bodies_per_env, 7)


def get_bodies_pos(
    env: "NewtonEnv",
    body_patterns: str | list[str],
) -> BodiesResult:
    """Get world positions for bodies matching pattern.

    Returns:
        BodiesResult with data shape (num_envs, num_bodies, 3).
    """
    cache = get_cache(env)
    body_indices = cache.get_body_indices(body_patterns)
    body_names = [cache.body_names[i] for i in body_indices]

    body_q = get_body_q(env)
    data = body_q[:, body_indices, :3]

    return BodiesResult(data=data, body_names=body_names, body_indices=body_indices)


def get_bodies_quat(
    env: "NewtonEnv",
    body_patterns: str | list[str],
) -> BodiesResult:
    """Get world quaternions for bodies matching pattern.

    Returns:
        BodiesResult with data shape (num_envs, num_bodies, 4).
    """
    cache = get_cache(env)
    body_indices = cache.get_body_indices(body_patterns)
    body_names = [cache.body_names[i] for i in body_indices]

    body_q = get_body_q(env)
    data = body_q[:, body_indices, 3:]

    return BodiesResult(data=data, body_names=body_names, body_indices=body_indices)


def get_bodies_height(
    env: "NewtonEnv",
    body_patterns: str | list[str],
) -> BodiesResult:
    """Get z-coordinates for bodies matching pattern.

    Returns:
        BodiesResult with data shape (num_envs, num_bodies).
    """
    result = get_bodies_pos(env, body_patterns)
    return BodiesResult(
        data=result.data[..., 2],
        body_names=result.body_names,
        body_indices=result.body_indices,
    )


# ============================================================
# Body Queries with Contact (requires contact_manager tracking)
# ============================================================

def get_bodies_pos_with_contact(
    env: "NewtonEnv",
    body_patterns: str | list[str],
) -> BodiesWithContactResult:
    """Get world positions for bodies tracked by contact_manager.

    Returns:
        BodiesWithContactResult with data shape (num_envs, num_bodies, 3).
    """
    cache = get_cache(env)
    body_indices, contact_indices = cache.get_body_indices_with_contact(body_patterns)
    body_names = [cache.body_names[i] for i in body_indices]

    body_q = get_body_q(env)
    data = body_q[:, body_indices, :3]

    return BodiesWithContactResult(
        data=data,
        body_names=body_names,
        body_indices=body_indices,
        contact_indices=contact_indices,
    )


def get_bodies_height_with_contact(
    env: "NewtonEnv",
    body_patterns: str | list[str],
) -> BodiesWithContactResult:
    """Get z-coordinates for bodies tracked by contact_manager.

    Returns:
        BodiesWithContactResult with data shape (num_envs, num_bodies).
    """
    result = get_bodies_pos_with_contact(env, body_patterns)
    return BodiesWithContactResult(
        data=result.data[..., 2],       # z
        body_names=result.body_names,
        body_indices=result.body_indices,
        contact_indices=result.contact_indices,
    )
