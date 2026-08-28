"""Bisect the mjlab reset path: which operation corrupts CUDA state at scale.

Context: at num_envs>=256 the g1_29dof mujoco env crashes inside
``env.reset()`` with ``CUDA_ERROR_ILLEGAL_ADDRESS`` surfacing at
``recompute_constants -> smooth.kinematics``.  A pure-mjlab repro of that
exact call chain (expand + TorchArray writes + recompute_constants) PASSES
at 256, so the faulting operation is something ELSE in our reset sequence
whose asynchronous error only surfaces at the next kernel launch.

This diag builds the same env (no JAX), then replays the reset event
sequence ONE TERM AT A TIME with ``wp.synchronize_device()`` +
``torch.cuda.synchronize()`` after every call, so the first faulting
operation raises at its OWN stage marker instead of at a later launch.

Usage (GPU box):
    CUDA_LAUNCH_BLOCKING=1 python -m rlworld.scripts.diag.dr.mjlab_dr_bisect_diag
    CUDA_LAUNCH_BLOCKING=1 python -m rlworld.scripts.diag.dr.mjlab_dr_bisect_diag --preset rough
"""

from __future__ import annotations

import argparse

import torch
import warp as wp


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


def _sync() -> None:
    wp.synchronize_device("cuda:0")
    torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=("flat", "rough"), default="flat")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = args.num_envs
    torch.manual_seed(args.seed)

    if args.preset == "flat":
        from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig

        cfgs = G1FlatConfig(sim_type="mujoco", num_envs=n, seed=args.seed).build()
    else:
        from rlworld.rl.configs.presets.g1_29dof.mujoco.rough import G1RoughMujocoConfig

        cfgs = G1RoughMujocoConfig().build()
        cfgs.apply_overrides(**{"env": {"num_envs": n, "seed": args.seed}})

    from rlworld.rl.evals.sim_initializers.mjlab import MujocoInitializer

    env = MujocoInitializer().init_environment(cfgs)
    _sync()
    _stage(f"env built (num_envs={n}, preset={args.preset})")

    from rlworld.rl.configs.base_config import iter_terms
    from rlworld.rl.configs.events.event_term_config import EventTermConfig

    all_ids = torch.arange(n, device=env.device, dtype=torch.long)

    # Replay the reset-path event terms one by one, synchronizing after
    # each so an async fault raises at its own marker.
    for mode in ("reset", "reset_dr"):
        for name, term in iter_terms(env.event_cfg, EventTermConfig).items():
            if term.mode != mode:
                continue
            fn_name = getattr(term.func, "__name__", str(term.func))
            env.event_manager._call_event_fn(name, term, env_ids=all_ids)
            _sync()
            _stage(f"[{mode}] term '{name}' ({fn_name}) done")

    _stage("manual per-term replay complete — now the full env.reset()")

    env.reset()
    _sync()
    _stage("full env.reset() done")

    env.reset()
    _sync()
    _stage("second full env.reset() done")

    print(f"\nOVERALL: PASS (num_envs={n}, preset={args.preset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
