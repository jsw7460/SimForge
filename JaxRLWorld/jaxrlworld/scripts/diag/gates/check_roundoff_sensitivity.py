"""How far does a roundoff-sized nudge carry in one preset's rollout?

When two builds of a simulator (before and after a pull, say) produce
trajectories that agree to float32 roundoff for a few steps and then
part, the question is whether the builds compute different physics or
the same physics with operations in another order. A rollout that
amplifies a perturbation of a few ulps into a visible difference within
the same number of steps cannot tell those apart -- and that is the
property this diag measures.

It runs ``check_all_presets``'s cell twice with the CURRENT code: once
as the sweep does, once with the first step's actions scaled by
``1 + eps`` (``eps`` defaults to two float32 ulps). Both runs are
otherwise identical. It prints the per-step relative difference of the
reward and observation fingerprints, and, given the two sweep
directories of a before/after comparison (``--before`` / ``--after``),
the per-step before/after difference of the same cell next to it.

Reading it: if the nudge grows to the before/after difference on the
same schedule, the before/after difference is explained by roundoff
amplified by the rollout; if the before/after difference is far larger
at the steps where the nudge is still at roundoff, the builds compute
different physics.

    python -m jaxrlworld.scripts.diag.gates.check_roundoff_sensitivity --preset g1_flat --sim genesis \\
        --before pull_before --after pull_after
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _apply_mode, _finite, _load, _obs_vector

KEYS = ("rew", "obs", "obs_abs")


def run(preset: str, sim: str, num_envs: int, steps: int, seed: int, eps: float) -> dict:
    loader = next(loader for label, loader, sims in _PUBLIC if label == preset and sim in sims)
    cfgs = _load(loader, sim, num_envs)
    _apply_mode(cfgs, "eager", sim)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    env.reset()
    gen = torch.Generator().manual_seed(seed)
    fp = {"rew": [], "rew_sq": [], "done": [], "obs": [], "obs_abs": []}
    for k in range(steps):
        actions = torch.randn((env.num_envs, env.num_actions), generator=gen, device="cpu") * 0.5
        if k == 0:
            actions = actions * (1.0 + eps)
        obs, rewards, terminated, truncated, _ = env.step(actions.to(env.device))
        if not _finite(rewards) or not _finite(obs):
            raise RuntimeError(f"non-finite reward or observation at step {k}")
        vec = _obs_vector(obs).double()
        fp["rew"].append(float(rewards.double().sum()))
        fp["rew_sq"].append(float((rewards.double() ** 2).sum()))
        fp["done"].append(int((terminated | truncated).sum()))
        fp["obs"].append(float(vec.sum()))
        fp["obs_abs"].append(float(vec.abs().sum()))
    return fp


def per_step_rel(a: dict, b: dict, steps: int) -> list[float]:
    return [max(abs(a[key][k] - b[key][k]) / max(abs(b[key][k]), 1e-6) for key in KEYS) for k in range(steps)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=2.0**-22, help="relative nudge of the first step's actions")
    ap.add_argument("--before", default=None, help="sweep --out directory of the build before the change")
    ap.add_argument("--after", default=None, help="sweep --out directory of the build after the change")
    ap.add_argument("--child-eps", type=float, default=None, help="internal")
    ap.add_argument("--child-out", default=None, help="internal")
    args = ap.parse_args()

    if args.child_out is not None:
        fp = run(args.preset, args.sim, args.num_envs, args.steps, args.seed, args.child_eps)
        Path(args.child_out).write_text(json.dumps(fp))
        return 0

    # Each run in its own process, as the sweep's cells are: a simulator
    # may hold process-wide state that a second build would inherit.
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, eps in (("plain", 0.0), ("nudged", args.eps)):
            out = Path(tmp) / f"{name}.json"
            cmd = [
                sys.executable,
                "-m",
                "jaxrlworld.scripts.diag.gates.check_roundoff_sensitivity",
                "--preset",
                args.preset,
                "--sim",
                args.sim,
                "--num-envs",
                str(args.num_envs),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--child-eps",
                repr(eps),
                "--child-out",
                str(out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-3000:], proc.stderr[-3000:])
                raise SystemExit(f"{name} run failed (rc={proc.returncode})")
            runs[name] = json.loads(out.read_text())

    nudge = per_step_rel(runs["nudged"], runs["plain"], args.steps)
    pull = None
    if args.before and args.after:
        cell = f"{args.preset}__{args.sim}.json"
        before = json.loads((Path(args.before) / cell).read_text())["fingerprint"]
        after = json.loads((Path(args.after) / cell).read_text())["fingerprint"]
        n = min(args.steps, len(before["rew"]), len(after["rew"]))
        pull = per_step_rel(after, before, n)
        # The sweep's own run of the current code must be the plain run here.
        same = per_step_rel(after, runs["plain"], n)
        print(
            f"sanity: this diag's plain run vs the --after sweep cell, max rel {max(same):.1e} (0 or run-to-run noise)"
        )

    print(f"\n{args.preset}:{args.sim}  num_envs={args.num_envs}  eps={args.eps:.2e} on step-0 actions")
    print(f"{'step':>4}  {'nudge (same code)':>18}" + (f"  {'before vs after':>16}" if pull else ""))
    for k in range(args.steps):
        line = f"{k:>4}  {nudge[k]:>18.1e}"
        if pull is not None and k < len(pull):
            line += f"  {pull[k]:>16.1e}"
        print(line)
    if pull is not None:
        n = len(pull)
        print(f"\nmax over {n} steps: nudge {max(nudge[:n]):.1e}, before vs after {max(pull):.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
