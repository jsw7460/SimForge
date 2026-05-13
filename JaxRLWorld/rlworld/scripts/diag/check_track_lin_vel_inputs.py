"""Decompose ``track_lin_vel`` divergence across simulators.

``track_lin_vel = exp(-(||cmd - v_b_xy||² + v_b_z²) / std²)``. If the value
diverges across sims while ``v_b`` is unified (we verified that), then
**only the command** differs. This script dumps the raw inputs per sim:

  command       = (lin_vel_x, lin_vel_y, ang_vel)   sampled at reset
  measured v_b  = root_link_lin_vel_b                read after one zero-action step
  manual reward = exp(-(||cmd_xy - v_b_xy||² + v_b_z²) / std²)

for the first ``--num-envs`` envs, per sim. Same ``torch.manual_seed``
before each ``env.reset()`` so RNG entering reset is identical. Any
sim-to-sim divergence in the printed commands localizes the cause to
sim-specific reset-path RNG consumption (DR / event terms / ...).

Usage:
    python -m rlworld.scripts.diag.check_track_lin_vel_inputs
    python -m rlworld.scripts.diag.check_track_lin_vel_inputs --num-envs 4 --std 0.5
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import importlib

import numpy as np

_SIMS = ("genesis", "newton", "mujoco")


def _build_env(sim: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    mod = importlib.import_module("rlworld.rl.configs.presets.g1_29dof.base")
    cfg_cls = mod.G1FlatConfig
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _read(env) -> dict[str, np.ndarray]:
    cm = env.command_manager
    rd = env.get_robot_data()

    def _np(t):
        return t.detach().cpu().numpy()

    return {
        "lin_vel_x": _np(cm.lin_vel_x),
        "lin_vel_y": _np(cm.lin_vel_y),
        "ang_vel": _np(cm.ang_vel),
        "v_b": _np(rd.root_link_lin_vel_b),
        "v_w": _np(rd.root_link_lin_vel_w),
    }


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std", type=float, default=0.5)
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    results: dict[str, dict[str, np.ndarray]] = {}

    for sim in sims:
        print(f"\n{'=' * 64}\nBuilding [{sim}] g1_29dof (num_envs={args.num_envs}) ...")
        env = _build_env(sim, args.num_envs)
        torch.manual_seed(args.seed)
        env.reset()
        # one zero-action step (matches check_reward_parity step 0)
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(zero)
        results[sim] = _read(env)
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-env dump.
    for env_i in range(args.num_envs):
        print(f"\n{'─' * 64}\n[env {env_i}]")
        print(f"  {'sim':<10s} {'cmd (vx, vy, wz)':<30s} {'v_b (x, y, z)':<30s} {'reward':<10s}")
        for sim in sims:
            r = results[sim]
            cmd_xy = np.array([r["lin_vel_x"][env_i], r["lin_vel_y"][env_i]])
            cmd_wz = r["ang_vel"][env_i]
            v_b = r["v_b"][env_i]
            err_xy = np.sum((cmd_xy - v_b[:2]) ** 2)
            err_z = v_b[2] ** 2
            reward = float(np.exp(-(err_xy + err_z) / args.std**2))
            cmd_str = f"({cmd_xy[0]:+.3f}, {cmd_xy[1]:+.3f}, {cmd_wz:+.3f})"
            v_str = f"({v_b[0]:+.4f}, {v_b[1]:+.4f}, {v_b[2]:+.4f})"
            print(f"  {sim:<10s} {cmd_str:<30s} {v_str:<30s} {reward:.5f}")

    # Cross-sim check: do commands match between sims?
    if len(sims) >= 2:
        print(f"\n{'=' * 64}\nCROSS-SIM CHECK: do commands match across sims?")
        ref = sims[0]
        for sim in sims[1:]:
            d_x = np.max(np.abs(results[sim]["lin_vel_x"] - results[ref]["lin_vel_x"]))
            d_y = np.max(np.abs(results[sim]["lin_vel_y"] - results[ref]["lin_vel_y"]))
            d_w = np.max(np.abs(results[sim]["ang_vel"] - results[ref]["ang_vel"]))
            verdict = "IDENTICAL" if max(d_x, d_y, d_w) < 1e-6 else "DIFFERENT"
            print(f"  {sim} vs {ref}: max|Δlin_x|={d_x:.4g}  max|Δlin_y|={d_y:.4g}  max|Δang|={d_w:.4g}  →  {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
