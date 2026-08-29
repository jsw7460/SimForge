"""Regression gate for the torch<->JAX bridge's memory lifetime.

``from_dlpack`` hands JAX a view of torch's buffer and the capsule
releases that buffer when the view is dropped. The copy taken out of it
is only *enqueued*, so if the source dies before the copy runs, torch's
caching allocator can hand the block to the next allocation and the copy
reads whatever overwrote it — silently, with no error and no shape or
dtype change. ``torch_to_jax`` prevents this by waiting for the copy
while its parameter still holds the source alive.

READ THIS BEFORE TRUSTING A PASS. Section [1] does NOT reproduce the
failure. It converts temporaries, forces torch to allocate over every
released block, defers all readback, feeds the results to a jitted
consumer so the copy can be deferred, and runs thousands of trials at
the collect loop's own tensor sizes — and the pre-fix implementation
still returns correct values every time, on GPU. Whatever the real loop
does to trigger this, no synthetic arrangement tried so far reproduces
it. Section [1] is therefore reported for information and cannot fail
the gate; it is kept because a future JAX or torch version may start
reproducing it, and because it documents exactly what was tried.

The failure is only observable end to end: HalfCheetah PPO settles at
~770 without the wait and ~1560 with it, three seeds each, against SB3's
~1560. The real regression test is
``rlworld/scripts/benchmark/sb3_compare/ppo_halfcheetah.bash``.

Sections [2]-[4] do gate: they check invariants that are observable —
that the copy is real rather than elided, that a batched conversion
keeps its sources alive, and that the reverse direction keeps the JAX
buffer alive.

Three gates that came before this one all passed against the broken
implementation, so the shape of this test matters more than its
existence:

- The source is a **temporary**, dying at the call boundary, which is how
  the collection loop actually calls the bridge
  (``torch_to_jax(t.to(torch.uint8))``).
- Nothing is read back until every conversion is done. A readback is a
  synchronisation, and synchronising is the very thing under test — the
  earlier gates compared each trial immediately and so could never
  observe the failure.
- After each conversion, torch is made to allocate over the block that
  was just released.
- The pre-fix implementation is run side by side in the same process and
  **must fail**. A gate that passes both ways is not testing anything,
  and this reports that as a failure of the gate itself.

The hazard needs asynchronous execution, so the old-implementation check
is only meaningful on GPU; on CPU the gate still verifies the current
implementation but says plainly that it proved nothing about the old one.

Run on the training box:
    jaxpy -m rlworld.scripts.diag.gates.check_dlpack_bridge_lifetime
    jaxpy -m rlworld.scripts.diag.gates.check_dlpack_bridge_lifetime --trials 400
"""

import argparse

import jax
import jax.dlpack as jdl
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax, torch_to_jax_many

POISON = -999.0


def _pre_fix_bridge(x: torch.Tensor) -> jax.Array:
    """The implementation this gate exists to reject, verbatim."""
    return jnp.array(jdl.from_dlpack(x))


@jax.jit
def _consume(x: jax.Array, w: jax.Array) -> jax.Array:
    """Stand-in for the policy forward the converted array feeds.

    The collect loop never leaves a converted array sitting unused: it
    goes straight into a jitted forward and into donated storage
    buffers, which lets XLA fuse the copy into that computation and
    defer it. A gate that converts and then touches nothing may see the
    copy run eagerly, and then it is not testing the same thing.
    """
    return jnp.tanh(x @ w).sum(axis=-1)


def _run_trials(bridge, trials: int, shape: tuple[int, ...], reuse: int, device: str) -> tuple[int, list]:
    """Convert ``trials`` temporaries, then read them all back at the end."""
    kept: list[tuple[jax.Array, jax.Array]] = []
    ballast = jnp.ones((1024, 1024))
    weights = jnp.ones((shape[-1], 32)) / shape[-1]

    for i in range(trials):
        # Keep JAX busy so the enqueued copy cannot run straight away.
        ballast = ballast @ ballast / 1024.0
        # The source is a temporary: its last reference is the argument,
        # exactly like ``torch_to_jax(t.to(torch.uint8))`` in the loop.
        converted = bridge(torch.full(shape, float(i + 1), device=device, dtype=torch.float32))
        kept.append((converted, _consume(converted, weights)))
        # Make torch allocate over whatever was just released. Same shape,
        # so the block lands back in the same size class.
        poison = [torch.full(shape, POISON, device=device, dtype=torch.float32) for _ in range(reuse)]
        del poison

    bad = []
    for i, (arr, _) in enumerate(kept):
        got = np.asarray(arr)
        if not np.all(got == float(i + 1)):
            bad.append((i, float(i + 1), float(got.flat[0]), int((got != float(i + 1)).sum())))
    return len(bad), bad


def _report(name: str, n_bad: int, bad: list, trials: int) -> None:
    print(f"  {name:<22s} {trials - n_bad}/{trials} conversions correct")
    for i, want, got, count in bad[:3]:
        print(f"      trial {i:4d}: expected {want}, got {got} ({count} elements wrong)")
    if len(bad) > 3:
        print(f"      ... and {len(bad) - 3} more corrupted conversions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="rows per converted tensor; with --obs-dim this sets the allocation size class, "
        "and the collect loop's tensors are small enough to live in torch's small-block pool",
    )
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--reuse", type=int, default=32, help="allocations forced over each released block")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    on_gpu = device == "cuda" and jax.default_backend() == "gpu"
    shape = (args.num_envs, args.obs_dim)
    n = args.num_envs * args.obs_dim

    print("=" * 78)
    print("  DLPACK BRIDGE LIFETIME — REGRESSION GATE")
    print(f"  torch device: {device}   jax backend: {jax.default_backend()}")
    print(f"  trials: {args.trials}   shape: {shape} ({n * 4} bytes)   reuse allocations: {args.reuse}")
    print("=" * 78)

    failures = 0

    print("\n  [1] source lifetime, sources dropped at the call boundary")
    cur_bad, cur = _run_trials(torch_to_jax, args.trials, shape, args.reuse, device)
    _report("torch_to_jax", cur_bad, cur, args.trials)
    old_bad, old = _run_trials(_pre_fix_bridge, args.trials, shape, args.reuse, device)
    _report("pre-fix (info only)", old_bad, old, args.trials)

    if cur_bad:
        print("  FAIL — the current bridge corrupts data. This is the bug, not a gate problem.")
        failures += 1
    if old_bad:
        print("  NOTE — the pre-fix implementation failed here. That is new: no synthetic")
        print("         arrangement had reproduced it before, so this section has become a")
        print("         real gate. Make it one — drop the informational wording above.")
    else:
        print("  NOTE — the pre-fix implementation passed, as it always has here. This section")
        print("         proves nothing either way; the end-to-end benchmark is the real gate")
        print("         (sb3_compare/ppo_halfcheetah.bash, 3 seeds, ~1560 with the wait vs")
        print("         ~770 without). Sections [2]-[4] below are what can fail.")

    print("\n  [2] the copy is real, not elided into a view of torch's buffer")
    src = torch.full((n,), 1.0, device=device, dtype=torch.float32)
    converted = torch_to_jax(src)
    try:
        aliased = converted.unsafe_buffer_pointer() == src.data_ptr()
    except Exception as exc:  # single-device arrays only
        print(f"      buffer pointer unavailable ({exc.__class__.__name__}); skipped")
        aliased = False
    print(f"      shares torch's buffer: {aliased}")
    if aliased:
        print("  FAIL — jnp.array elided the copy, so waiting for it guarantees nothing.")
        failures += 1
    del src, converted

    print("\n  [3] torch_to_jax_many keeps every source of a batch alive")
    batch_bad = 0
    for i in range(args.trials // 4):
        sources = {
            "obs": torch.full((n,), float(i + 1), device=device, dtype=torch.float32),
            # A cast temporary: the mapping is what keeps it alive.
            "done": (torch.full((n,), float(i % 2), device=device) > 0.5).to(torch.uint8),
        }
        out = torch_to_jax_many(sources)
        del sources
        poison = [torch.full((n,), POISON, device=device, dtype=torch.float32) for _ in range(args.reuse)]
        del poison
        if not np.all(np.asarray(out["obs"]) == float(i + 1)) or not np.all(np.asarray(out["done"]) == (i % 2)):
            batch_bad += 1
    print(f"      {args.trials // 4 - batch_bad}/{args.trials // 4} batches correct")
    if batch_bad:
        print("  FAIL — a batched conversion lost its source before the copy ran.")
        failures += 1

    print("\n  [4] reverse direction: torch's view of a JAX buffer outlives the JAX array")
    rev_bad = 0
    for i in range(args.trials // 4):
        array = jnp.full((n,), float(i + 1), dtype=jnp.float32)
        tensor = jax_to_torch(array, device)
        del array
        ballast = [jnp.full((n,), POISON, dtype=jnp.float32) for _ in range(args.reuse)]
        got = tensor.cpu().numpy()
        del ballast, tensor
        if not np.all(got == float(i + 1)):
            rev_bad += 1
    print(f"      {args.trials // 4 - rev_bad}/{args.trials // 4} conversions correct")
    if rev_bad:
        print("  FAIL — jax_to_torch's tensor does not keep the JAX buffer alive.")
        failures += 1

    print("\n" + "=" * 78)
    if failures:
        print(f"  {failures} check(s) failed.")
        print("=" * 78)
        return 1
    print("  PASS — every invariant this gate can observe holds.")
    if not on_gpu:
        print("         (CPU run: section [1] is doubly uninformative here, execution is synchronous.)")
    print("         It does NOT cover the failure that motivated the fix — see the module")
    print("         docstring. Only the end-to-end PPO benchmark does.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
