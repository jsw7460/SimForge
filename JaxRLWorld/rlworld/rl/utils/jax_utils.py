import jax
import jax.dlpack as jdl
import torch
from jax import numpy as jnp


def torch_to_jax(x: torch.Tensor) -> jax.Array:
    """Convert torch.Tensor to jax.Array.

    IMPORTANT: jax.dlpack.from_dlpack() performs zero-copy conversion,
    meaning the returned JAX array shares the same memory buffer as the
    original torch tensor. If the torch tensor is reused/overwritten by
    the environment on subsequent steps (which is common for reward,
    obs, and done buffers), all previously stored JAX arrays pointing
    to that buffer will silently have their values corrupted.

    We wrap with jnp.array() to force a copy and ensure each JAX array
    owns its own memory. Discovered 2026-02-03 while debugging PPO
    rollout storage: rewards_list[0] was being overwritten by later
    steps due to shared dlpack memory, causing incorrect GAE computation.
    """
    return jnp.array(jdl.from_dlpack(x))


def jax_to_torch(x: jax.Array, device: torch.device) -> torch.Tensor:
    """Convert jax.Array to torch.Tensor."""
    return torch.from_dlpack(x)


def convert_infos_to_jax(infos: dict, device: torch.device) -> dict:
    """Convert info dict values from torch to jax where needed."""
    result = {}
    for k, v in infos.items():
        if isinstance(v, torch.Tensor):
            result[k] = torch_to_jax(v)
        elif isinstance(v, dict):
            result[k] = convert_infos_to_jax(v, device)
        else:
            result[k] = v
    return result
