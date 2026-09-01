import jax
import jax.dlpack as jdl
import torch
from jax import numpy as jnp


def torch_to_jax(x: torch.Tensor) -> jax.Array:
    """Convert a torch tensor into a JAX array that owns its memory.

    ``from_dlpack`` does not copy. It hands JAX a *view* of the buffer
    torch owns, and the DLPack capsule releases that buffer as soon as
    the view is dropped. ``jnp.array`` copies out of the view, but only
    *enqueues* that copy — so with nothing else in the way the order is:
    capsule released -> torch's caching allocator hands the block to the
    next allocation -> the copy finally runs and reads whatever
    overwrote it. The result is wrong and nothing raises.

    Two things prevent that, and both are load-bearing:

    - ``x`` stays bound as a parameter for the whole call, so the
      capsule's release cannot drop the storage's refcount to zero.
      Callers passing a temporary — ``torch_to_jax(t.to(torch.uint8))``
      is the common shape — depend on exactly this: their temporary
      dies at the call boundary, and the wait below happens inside it.
    - ``block_until_ready`` makes the copy actually run while ``x`` is
      still alive.

    Drop either and the corruption comes back silently. It arrived that
    way once already: the collection loop's per-step ``.cpu()`` calls
    had been serving as the wait, and removing them as an optimisation
    cost roughly half of PPO's return on GPU while every value that got
    compared still came out equal.

    Be warned about how weak the evidence available here is. The damage
    is only visible end to end — HalfCheetah PPO settles at ~770 without
    the wait and ~1560 with it, three seeds each, against SB3's ~1560 —
    and every attempt to provoke it synthetically has failed, including
    ``check_dlpack_bridge_lifetime``, which converts temporaries, forces
    torch to allocate over the released blocks, defers the readback, and
    still cannot make the pre-fix code produce a wrong number. So the
    lifetime account above is the explanation that fits every experiment,
    not something a unit test observed. Treat the benchmark as the gate:
    ``jaxrlworld/scripts/benchmark/sb3_compare/ppo_halfcheetah.bash``, three
    seeds, compared against SB3's runs in the same project. Removing this
    wait will not fail anything else.

    Converting several tensors at once? :func:`torch_to_jax_many` pays
    the wait once for the whole group instead of once per tensor.
    """
    out = jnp.array(jdl.from_dlpack(x))
    out.block_until_ready()
    return out


def torch_to_jax_many(sources: dict[str, torch.Tensor]) -> dict[str, jax.Array]:
    """Convert a group of torch tensors, waiting once for the whole group.

    Same guarantee as :func:`torch_to_jax` — every returned array owns
    its memory — at one synchronisation instead of one per tensor. What
    makes that safe is the ``sources`` mapping: it holds every input
    alive across the single wait.

    So temporaries have to be built *into the mapping*, never converted
    on their own alongside it::

        sources["terminated"] = terminated.to(torch.uint8)   # right
        terminated_jax = torch_to_jax(terminated.to(torch.uint8))  # separate wait

    Casting the results (``.astype(jnp.bool_)``) is done afterwards on
    the returned arrays, which JAX owns.
    """
    out = {name: jnp.array(jdl.from_dlpack(tensor)) for name, tensor in sources.items()}
    jax.block_until_ready(list(out.values()))
    return out


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
