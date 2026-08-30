"""What the device replay buffer is worth, at the shape that matters.

The gym benchmark is the wrong place to judge this. There the
transitions are born on the host — a CPU MuJoCo env — so a host buffer
is already where the data is, and moving the buffer to the accelerator
mostly moves the copy rather than removing it.

The shape worth measuring is the one a GPU simulator produces, which is
what ``go2/newton/sac`` does: 8192 environments, ``num_steps_per_env=24``
collecting into the buffer, then ``num_gradient_steps=200`` sampling
batches of 8192 back out. There every transition is born on the device,
so a host buffer makes it cross down and back up for nothing.

This times ``store_parallel`` and ``sample_batch`` alone, both backends,
at those shapes — no environment, no algorithm, nothing else to
attribute. The per-iteration totals at the end are what a training
iteration of that preset actually spends moving data.

Run on the training box:
    jaxpy -m rlworld.scripts.diag.perf.replay_buffer_throughput
    jaxpy -m rlworld.scripts.diag.perf.replay_buffer_throughput --num-envs 4096 --obs-dim 100
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from rlworld.rl.storages import make_replay_buffer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--obs-dim", type=int, default=48)
    ap.add_argument("--act-dim", type=int, default=12)
    ap.add_argument("--buffer-size", type=int, default=5_000_000)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--stores-per-iter", type=int, default=24, help="num_steps_per_env")
    ap.add_argument("--samples-per-iter", type=int, default=200, help="num_gradient_steps")
    ap.add_argument("--reps", type=int, default=200, help="timed calls per measurement")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    size_per_env = args.buffer_size // args.num_envs
    floats = 2 * args.obs_dim + 2 * args.obs_dim + args.act_dim + 3
    bytes_total = args.num_envs * size_per_env * floats * 4
    per_store = args.num_envs * floats * 4
    per_sample = args.batch_size * floats * 4

    print("=" * 78)
    print("  REPLAY BUFFER THROUGHPUT — HOST VS DEVICE")
    print(
        f"  backend {jax.default_backend()}   envs {args.num_envs:,}   ring {size_per_env:,}   n_steps {args.n_steps}"
    )
    print(f"  batch {args.batch_size:,}   buffer holds {bytes_total / 2**30:.2f} GiB")
    print(f"  one store moves {per_store / 2**20:.2f} MiB, one sample {per_sample / 2**20:.2f} MiB")
    print("=" * 78)

    rng = np.random.default_rng(0)
    fields = [
        jnp.asarray(rng.normal(size=(args.num_envs, args.obs_dim)), jnp.float32),
        jnp.asarray(rng.normal(size=(args.num_envs, args.obs_dim)), jnp.float32),
        jnp.asarray(rng.normal(size=(args.num_envs, args.act_dim)), jnp.float32),
        jnp.asarray(rng.normal(size=(args.num_envs,)), jnp.float32),
        jnp.asarray(rng.normal(size=(args.num_envs, args.obs_dim)), jnp.float32),
        jnp.asarray(rng.normal(size=(args.num_envs, args.obs_dim)), jnp.float32),
        jnp.asarray(rng.random(args.num_envs) < 0.02),
        jnp.asarray(rng.random(args.num_envs) < 0.02),
    ]

    results = {}
    for device in ("host", "device"):
        buf = make_replay_buffer(
            device,
            num_envs=args.num_envs,
            actor_obs_dim=args.obs_dim,
            critic_obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            size_per_env=size_per_env,
            n_steps=args.n_steps,
            gamma=0.97,
            seed=0,
        )
        for _ in range(max(args.n_steps + 1, args.warmup)):
            buf.store_parallel(*fields)

        for _ in range(args.warmup):
            buf.store_parallel(*fields)
        jax.block_until_ready(buf.sample_batch(args.batch_size))
        t0 = time.perf_counter()
        for _ in range(args.reps):
            buf.store_parallel(*fields)
        jax.block_until_ready(buf.sample_batch(args.batch_size))
        t_store = (time.perf_counter() - t0) / args.reps * 1e3

        for _ in range(args.warmup):
            batch = buf.sample_batch(args.batch_size)
        jax.block_until_ready(batch)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            batch = buf.sample_batch(args.batch_size)
        jax.block_until_ready(batch)
        t_sample = (time.perf_counter() - t0) / args.reps * 1e3

        results[device] = (t_store, t_sample)
        print(f"  {device:<7s} store {t_store:7.3f} ms/call    sample {t_sample:7.3f} ms/call")
        del buf

    print("\n" + "-" * 78)
    print(f"  per training iteration ({args.stores_per_iter} stores + {args.samples_per_iter} samples)")
    totals = {}
    for device, (t_store, t_sample) in results.items():
        total = t_store * args.stores_per_iter + t_sample * args.samples_per_iter
        totals[device] = total
        print(
            f"    {device:<7s} {t_store * args.stores_per_iter:8.1f} ms storing"
            f" + {t_sample * args.samples_per_iter:8.1f} ms sampling"
            f" = {total:8.1f} ms"
        )
    saved = totals["host"] - totals["device"]
    print(f"    saved   {saved:8.1f} ms per iteration  ({100 * saved / totals['host']:+.0f}%)")
    print("-" * 78)
    print(f"  cost: {bytes_total / 2**30:.2f} GiB of device memory the simulator no longer has.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
