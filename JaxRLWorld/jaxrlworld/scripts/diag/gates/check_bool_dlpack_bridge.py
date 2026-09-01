"""Adversarial stress test for the uint8 DLPack boolean bridge.

The collect loop moves ``terminated`` / ``truncated`` / ``bootstrap_mask``
from torch to JAX.  The historical path was ``.cpu().numpy()`` — correct
but a full pipeline stall per tensor per step — because putting a BOOL
tensor through DLPack directly once produced rare, random bit flips (a
week-long debugging scar).  The replacement keeps bool out of DLPack
entirely: cast to uint8 on device (fresh buffer, exact 0/1 semantics),
cross via the same DLPack path the float rewards already use, cast back
to bool inside JAX.

This diag is the adoption gate: it hunts the "rare random flip" failure
mode directly.  Every trial builds a random-density bool tensor, wedges
heavy async kernel traffic into the queue to open any race window
(uncompleted producer writes, stream reordering), converts through BOTH
paths, and bit-compares against the synchronous ground truth.  Adversarial
patterns (all-True, all-False, alternating, single-True) are mixed in.

Adoption criterion: ZERO mismatched elements over every trial.  A single
flipped bit anywhere fails the diag and the old path stays.

Run on the training box:
    jaxpy -m jaxrlworld.scripts.diag.gates.check_bool_dlpack_bridge
    jaxpy -m jaxrlworld.scripts.diag.gates.check_bool_dlpack_bridge --trials 50000
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxrlworld.rl.utils.jax_utils import torch_to_jax


def new_path(t: torch.Tensor) -> jax.Array:
    """The proposed bridge: bool never touches DLPack."""
    return torch_to_jax(t.to(torch.uint8)).astype(jnp.bool_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--num-envs", type=int, default=4096)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.num_envs
    print("=" * 78)
    print("  BOOL->UINT8 DLPACK BRIDGE — ADVERSARIAL BIT-EQUALITY STRESS")
    print(f"  device: {device}  jax backend: {jax.default_backend()}  " f"trials: {args.trials}  num_envs: {n}")
    print("=" * 78)

    gen = torch.Generator(device=device).manual_seed(1234)
    # Ballast tensors for async queue pressure between produce and convert.
    ballast_a = torch.randn(2048, 2048, device=device)
    ballast_b = torch.randn(2048, 2048, device=device)
    # JAX-side ballast so both frameworks' streams stay busy.
    jax_ballast = jnp.ones((1024, 1024))

    mismatches = 0
    checked = 0
    t0 = time.perf_counter()
    for trial in range(args.trials):
        kind = trial % 8
        if kind < 4:
            # Random density sweeping the realistic range (sparse dones ... dense).
            p = (0.001, 0.02, 0.5, 0.98)[kind]
            src = torch.rand(n, device=device, generator=gen) < p
        elif kind == 4:
            src = torch.zeros(n, dtype=torch.bool, device=device)
        elif kind == 5:
            src = torch.ones(n, dtype=torch.bool, device=device)
        elif kind == 6:
            src = (torch.arange(n, device=device) % 2).bool()
        else:
            src = torch.zeros(n, dtype=torch.bool, device=device)
            src[int(torch.randint(0, n, (1,), device=device, generator=gen))] = True

        # Open the race window: pile unfinished async work onto both queues
        # so a stream-ordering bug would read stale/garbage data.
        ballast_a = ballast_a @ ballast_b
        ballast_a = ballast_a / (ballast_a.norm() + 1.0)
        jax_ballast = jax_ballast @ jax_ballast / 1024.0

        converted = new_path(src)

        # More traffic AFTER the conversion but before the readback, so a
        # lifetime bug (freed/reused source buffer) would also surface.
        ballast_b = ballast_b + ballast_a
        src.logical_not_()  # mutate the source: the bridge must have copied

        ref = src.logical_not().cpu().numpy()  # ground truth (synchronous)
        got = np.asarray(converted)
        bad = int((ref != got).sum())
        mismatches += bad
        checked += n
        if bad:
            print(f"  [trial {trial}] MISMATCH: {bad} elements (pattern kind {kind})")

    dt = time.perf_counter() - t0
    print(f"  checked {checked:,} elements over {args.trials:,} trials in {dt:.1f}s")
    print("=" * 78)
    if mismatches:
        print(f"  FAIL — {mismatches} mismatched elements. DO NOT adopt the uint8 bridge.")
        print("=" * 78)
        return 1
    print("  PASS — zero mismatches. The uint8 bridge is bit-exact under async pressure.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
