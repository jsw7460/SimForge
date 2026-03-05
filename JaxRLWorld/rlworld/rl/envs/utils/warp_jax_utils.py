"""Warp ↔ JAX conversion utilities.

CRITICAL CONSTRAINTS (hard-won from debugging):

1. jnp.array() forced copy is MANDATORY for wp_to_jax().
   wp.to_jax() uses dlpack zero-copy. Warp reuses internal buffers across steps,
   so storing a zero-copy JAX view will silently get overwritten on the next step.
   This caused PPO rollout corruption (rewards_list[0] overwritten by later steps).

2. dlpack does NOT support boolean dtype.
   Boolean warp arrays must bypass dlpack and convert via numpy instead.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp


def wp_to_jax(wp_array: wp.array) -> jax.Array:
    """Convert wp.array → jax.Array with forced copy (dlpack safety)."""
    if wp_array.dtype == wp.bool:
        return jnp.array(wp_array.numpy())
    return jnp.array(wp.to_jax(wp_array))


def wp_from_jax(jax_array: jax.Array, dtype: type = wp.float32) -> wp.array:
    """Convert jax.Array → wp.array."""
    return wp.from_jax(jax_array, dtype=dtype)
