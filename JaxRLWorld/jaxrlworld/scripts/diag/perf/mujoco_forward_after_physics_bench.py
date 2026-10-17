"""Step-time cost of MujocoSceneConfig.forward_after_physics.

The switch adds one ``sim.forward()`` after the decimation loop so that
termination and reward read the integrated state, as on Newton and Genesis.
This times the MuJoCo env of each preset with the switch off and on, at the
training env count, under the same random actions (fresh uniform actions in
[-1, 1] every step, so robots fall and reset as in early training):

    env.step  ms/step, off vs on, and the difference
    forward   ms per bare ``scene_manager.forward()`` call, timed alone

Each preset/switch pair is built in its own subprocess, so CUDA graph pools
and caches do not carry over between cells.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.perf.mujoco_forward_after_physics_bench
    jaxpy -m jaxrlworld.scripts.diag.perf.mujoco_forward_after_physics_bench --presets g1_29dof --num-envs 8192
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time

# Shorthand -> "module:Class"; any other "module:Class" is taken as given.
# The class must accept ``sim_type`` and ``num_envs``.
_PRESETS = {
    "g1_29dof": "jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig",
    "go2": "jaxrlworld.rl.configs.presets.go2.base:Go2FlatConfig",
    "k1_velocity": "jaxrlworld.rl.configs.presets.k1_velocity.base:K1VelocityConfig",
    "t1_getup": "jaxrlworld.rl.configs.presets.t1_getup.base:T1GetupConfig",
}
_DEFAULT_PRESETS = ("g1_29dof", "go2", "k1_velocity")


def _cell(spec: str, num_envs: int, forward: bool, warmup: int, steps: int, forward_calls: int, seed: int) -> dict:
    import torch

    from jaxrlworld.rl.runners import BaseRunner

    mod_path, cls_name = spec.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="mujoco", num_envs=num_envs).build()
    cfgs.scene.forward_after_physics = forward
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    gen = torch.Generator(device=env.device).manual_seed(seed)
    env.reset()

    def action() -> torch.Tensor:
        return torch.rand(env.num_envs, env.num_actions, device=env.device, generator=gen) * 2.0 - 1.0

    for _ in range(warmup):
        env.step(action())
    torch.cuda.synchronize()

    resets = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(action())
        resets += int((terminated | truncated).sum())
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / steps * 1e3

    env.scene_manager.forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(forward_calls):
        env.scene_manager.forward()
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) / forward_calls * 1e3

    return {"step_ms": step_ms, "forward_ms": forward_ms, "resets_per_step": resets / steps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", default=",".join(_DEFAULT_PRESETS), help=f"Comma list; shorthands {sorted(_PRESETS)}")
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--forward-calls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--_cell", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._cell is not None:
        spec, forward = args._cell.rsplit("@", 1)
        result = _cell(spec, args.num_envs, forward == "on", args.warmup, args.steps, args.forward_calls, args.seed)
        print("CELL_RESULT " + json.dumps(result), flush=True)
        return 0

    rows = []
    for name in args.presets.split(","):
        spec = _PRESETS.get(name, name)
        cells = {}
        for forward in ("off", "on"):
            cmd = [
                sys.executable,
                "-m",
                "jaxrlworld.scripts.diag.perf.mujoco_forward_after_physics_bench",
                "--_cell",
                f"{spec}@{forward}",
                "--num-envs",
                str(args.num_envs),
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--forward-calls",
                str(args.forward_calls),
                "--seed",
                str(args.seed),
            ]
            print(f"=== {name} forward_after_physics={forward}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("CELL_RESULT ")]
            if proc.returncode != 0 or not lines:
                print(proc.stdout[-3000:])
                print(proc.stderr[-3000:])
                raise SystemExit(f"cell {name}@{forward} failed (exit {proc.returncode})")
            cells[forward] = json.loads(lines[-1].removeprefix("CELL_RESULT "))
            print(f"    {cells[forward]}", flush=True)
        rows.append((name, cells))

    print(f"\nmujoco, num_envs={args.num_envs}, {args.steps} timed steps of uniform random actions")
    header = f"{'preset':14s} {'step off':>10s} {'step on':>10s} {'delta':>10s} {'delta %':>8s} {'forward':>10s} {'resets/step':>12s}"
    print(header)
    print("-" * len(header))
    for name, cells in rows:
        off, on = cells["off"], cells["on"]
        delta = on["step_ms"] - off["step_ms"]
        print(
            f"{name:14s} {off['step_ms']:9.3f}ms {on['step_ms']:9.3f}ms {delta:+9.3f}ms "
            f"{delta / off['step_ms'] * 100:+7.1f}% {off['forward_ms']:9.3f}ms {off['resets_per_step']:12.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
