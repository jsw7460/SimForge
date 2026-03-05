"""JAX-native quaternion utilities for Newton environments."""
import jax
import jax.numpy as jnp


def quat_to_xyz(quat: jax.Array, rpy: bool = True, degrees: bool = False) -> jax.Array:
    """Convert wxyz quaternion to Euler angles.

    Drop-in replacement for genesis.utils.geom.quat_to_xyz that works with JAX arrays.
    Supports arbitrary batch dimensions (e.g., [N, 4], [N, M, 4]).

    Args:
        quat: Quaternion array with last dim = 4, in wxyz convention.
        rpy: If True, return (roll, pitch, yaw). If False, return (x, y, z) = (pitch, yaw, roll).
        degrees: If True, convert to degrees.

    Returns:
        Euler angles array with last dim = 3.
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    if rpy:
        result = jnp.stack([roll, pitch, yaw], axis=-1)
    else:
        result = jnp.stack([pitch, yaw, roll], axis=-1)

    if degrees:
        result = jnp.degrees(result)

    return result