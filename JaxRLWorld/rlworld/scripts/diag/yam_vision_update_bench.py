"""Real-workload benchmark: PPO update time and device memory for the
yam_lift VISION task.

Measures what the rollout-storage minibatch refactor changes: the old
path materialized ``num_epochs`` full shuffled copies of the pixel
observations on device (431 ms / update and a 7.5 GiB OOM at 8192 envs
on the 4096-env baseline of 2026-08-27); the refactored path gathers one
minibatch per scan step.  Run before and after syncing the refactor and
compare.

    jaxpy -m rlworld.scripts.diag.yam_vision_update_bench
    jaxpy -m rlworld.scripts.diag.yam_vision_update_bench --num-envs 8192
"""

import argparse
import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import numpy as np

from rlworld.rl.configs.presets.yam_lift.vision import YamLiftVisionConfig
from rlworld.rl.runners import BaseRunner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="mujoco", help="vision path is wired for mjlab")
    ap.add_argument("--num-envs", type=int, default=None, help="default: preset value")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    overrides = {"num_envs": args.num_envs} if args.num_envs else {}
    cfgs = YamLiftVisionConfig(sim_type=args.sim, **overrides).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    alg = runner.alg

    print("=" * 78)
    print("  YAM LIFT VISION — PPO update bench")
    print(f"  backend: {jax.default_backend()}  num_envs: {runner.env.num_envs}")
    print("=" * 78)

    # One real rollout; then keep it (clear becomes a no-op) so update()
    # can run on identical data every rep.  compute_returns lives in
    # _run_training_iteration, between collect and update — mirror that.
    runner.it = 0
    obs = runner._get_initial_obs()
    t0 = time.perf_counter()
    data = runner._collect_experience(obs=obs, ep_infos=[])
    alg.compute_returns(data["last_obs"]["critic_obs"])
    print(f"  rollout collected in {time.perf_counter() - t0:.2f}s")
    alg.storage.clear = lambda: None

    for _ in range(args.warmup):
        alg.update()
    jax.block_until_ready(jax.tree_util.tree_leaves(alg.train_state.model)[0])

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        alg.update()
        jax.block_until_ready(jax.tree_util.tree_leaves(alg.train_state.model)[0])
        times.append(time.perf_counter() - t0)

    print(
        f"  update(): mean {np.mean(times) * 1e3:.1f} ms  (median {np.median(times) * 1e3:.1f}, "
        f"min {min(times) * 1e3:.1f}, {args.reps} reps)"
    )
    stats = jax.local_devices()[0].memory_stats() or {}
    for k in ("peak_bytes_in_use", "bytes_in_use", "largest_alloc_size"):
        if k in stats:
            print(f"  {k}: {stats[k] / 2**30:.2f} GiB")
    if not stats:
        print("  (device memory_stats unavailable on this backend)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
