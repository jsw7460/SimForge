"""Step-speed baseline, for before/after comparison across a refactor.

A standalone measurement, deliberately not framework instrumentation: the
env-var-gated profilers that used to live inside the managers were removed
on purpose, and nothing here runs unless this script is invoked.

Prints milliseconds per control step and per reset for a preset, so a
refactor that claims "no performance change" can be held to a number. Run
it before the change, keep the output, run it again after.

    python -m rlworld.scripts.diag.step_speed_baseline --num-envs 4096

or one backend:

    python -m rlworld.scripts.diag.step_speed_baseline --sim newton --num-envs 4096
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.runners import BaseRunner

_SIMS = ("genesis", "newton", "mujoco")


def _sync() -> None:
    """Block until the GPU has actually finished the queued work.

    Without this every launch is timed as ~0 and the total lands on
    whichever call happens to synchronize.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_single(sim: str, num_envs: int, warmup: int, steps: int, resets: int) -> dict:
    cfgs = Go2FlatConfig(sim_type=sim, num_envs=num_envs).build()
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()

    n_act = env.act_manager.num_actions
    torch.manual_seed(0)
    action = torch.empty(env.num_envs, n_act, device=env.device).uniform_(-0.3, 0.3)

    for _ in range(warmup):
        env.step(action)
    _sync()

    # Per-step samples rather than one total: a single mean hides a
    # periodic stall, which is exactly what a manager-loop regression
    # would look like.
    per_step: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        env.step(action)
        _sync()
        per_step.append((time.perf_counter() - t0) * 1e3)

    all_ids = torch.arange(env.num_envs, device=env.device)
    per_reset: list[float] = []
    for _ in range(resets):
        t0 = time.perf_counter()
        env._reset_idx(all_ids)
        _sync()
        per_reset.append((time.perf_counter() - t0) * 1e3)

    result = {
        "sim": sim,
        "num_envs": num_envs,
        "step_mean_ms": round(statistics.mean(per_step), 4),
        "step_median_ms": round(statistics.median(per_step), 4),
        "step_p95_ms": round(sorted(per_step)[int(0.95 * len(per_step))], 4),
        "step_max_ms": round(max(per_step), 4),
        "reset_mean_ms": round(statistics.mean(per_reset), 4),
        "num_actions": n_act,
    }

    print("=" * 78)
    print(f"STEP SPEED  [sim={sim}  num_envs={num_envs}  actions={n_act}]")
    print("=" * 78)
    print(f"  step   mean {result['step_mean_ms']:8.3f} ms   median {result['step_median_ms']:8.3f} ms")
    print(f"         p95  {result['step_p95_ms']:8.3f} ms   max    {result['step_max_ms']:8.3f} ms")
    print(f"  reset  mean {result['reset_mean_ms']:8.3f} ms   over {resets} full resets")
    print("=" * 78)
    return result


def run_all(num_envs: int, warmup: int, steps: int, resets: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="step_speed_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "rlworld.scripts.diag.step_speed_baseline",
            "--sim",
            sim,
            "--result-json",
            str(path),
            "--num-envs",
            str(num_envs),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--resets",
            str(resets),
        ]
        print()
        print("#" * 78)
        print(f"# $ {' '.join(cmd)}")
        print("#" * 78)
        subprocess.run(cmd, env=env_vars, check=False)
        if path.exists():
            out[sim] = json.loads(path.read_text())

    if not out:
        print("No backend produced a result.")
        return 1

    print()
    print("=" * 78)
    print(f"BASELINE SUMMARY  (num_envs={num_envs})")
    print("=" * 78)
    print(f"{'metric':<20}" + "".join(f"{s:>14}" for s in _SIMS))
    print("-" * 78)
    for key in ("step_mean_ms", "step_median_ms", "step_p95_ms", "step_max_ms", "reset_mean_ms"):
        row = f"{key:<20}"
        for s in _SIMS:
            v = out.get(s, {}).get(key)
            row += f"{'—' if v is None else f'{v:.3f}':>14}"
        print(row)
    print("=" * 78)
    print("Keep this table. Re-run after the change and compare.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=list(_SIMS), default=None, help="Run one backend. Omit to run all three.")
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--resets", type=int, default=20)
    args = ap.parse_args()

    if args.sim is None:
        return run_all(args.num_envs, args.warmup, args.steps, args.resets)

    result = run_single(args.sim, args.num_envs, args.warmup, args.steps, args.resets)
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
