"""Head-to-head: mjlab's native G1 flat env vs our parity preset, one process.

The uv-venv mjlab train reports 33.0 ms/step of collection at 16384 envs
while our physics-matched preset spends 45.9 — and the ledger only adds
up if mjlab's mjwarp portion is cheaper than ours. This bench removes
every remaining variable by running BOTH stacks in the same interpreter
(the conda env, same mujoco_warp/warp), stepping each env with the same
cadence, and dumping the compiled model + live contact counts:

1. mjlab native: ``unitree_g1_flat_env_cfg()`` + ``ManagerBasedRlEnv``,
   timed ``env.step(zero_actions)`` — their managers, their model.
2. ours: the speed-parity preset, timed ``env.step(zero_actions)``.

If the two engines see different ncon/nefc or model sizes, the gap is a
scene/model difference; if those match and mjlab still steps faster, the
difference is in the stacks' host-side work.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.perf.g1_mjlab_native_bench --num-envs 16384 --steps 96
"""

from __future__ import annotations

import argparse
import time

import torch


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_stats(mj_model, wp_data) -> dict:
    import warp as wp

    ncon_arr = getattr(wp_data, "ncon", None)
    if ncon_arr is None:
        ncon = -1
    elif isinstance(ncon_arr, torch.Tensor):
        ncon = int(ncon_arr.sum().item())
    else:
        ncon = int(wp.to_torch(ncon_arr).sum().item())
    stats = {
        "nq": mj_model.nq,
        "nv": mj_model.nv,
        "nu": mj_model.nu,
        "ngeom": mj_model.ngeom,
        "npair": mj_model.npair,
        "nsensor": mj_model.nsensor,
        "naconmax": getattr(wp_data, "naconmax", -1),
        "njmax": getattr(wp_data, "njmax", -1),
        "live ncon (total)": ncon,
    }
    names = [mj_model.sensor(i).name for i in range(mj_model.nsensor)]
    stats["sensor names"] = ", ".join(names)
    return stats


def _time_steps(step_fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        step_fn()
    _cuda_sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    _cuda_sync()
    return (time.perf_counter() - t0) * 1e3 / steps


def _bench_mjlab_native(num_envs: int, warmup: int, steps: int) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    cfg = unitree_g1_flat_env_cfg()
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    actions = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)

    # Bind the method so the timing lambda holds no reference to ``env``
    # after the ``del`` below (ruff F821 flags a deleted name in a closure).
    step_fn = env.step
    ms = _time_steps(lambda: step_fn(actions), warmup, steps)
    print(f"\n[mjlab native] env.step: {ms:.3f} ms/step ({num_envs} envs)")
    stats = _model_stats(env.sim.mj_model, env.sim._wp_data)
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")
    del env
    torch.cuda.empty_cache()


def _bench_ours(num_envs: int, warmup: int, steps: int) -> None:
    from jaxrlworld.rl.runners import BaseRunner
    from jaxrlworld.scripts.diag.perf.g1_mjlab_speed_parity import G1FlatMjlabSpeedParityConfig

    cfgs = G1FlatMjlabSpeedParityConfig(sim_type="mujoco", num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env
    env.reset()
    # Stagger like training so the reset branch runs, same as mjlab above
    # (mjlab's episodes terminate naturally during the bench too).
    env.termination_manager.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length)
    )
    actions = torch.zeros(num_envs, env.num_actions, device=env.device)

    ms = _time_steps(lambda: env.step(actions), warmup, steps)
    print(f"\n[ours parity ] env.step: {ms:.3f} ms/step ({num_envs} envs)")
    sim = env.scene_manager._sim
    stats = _model_stats(sim.mj_model, sim._wp_data)
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=24)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--skip-mjlab", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    args = ap.parse_args()

    if not args.skip_mjlab:
        _bench_mjlab_native(args.num_envs, args.warmup, args.steps)
    if not args.skip_ours:
        _bench_ours(args.num_envs, args.warmup, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
