"""How often does the MuJoCo Warp solver hit its iteration caps on a preset?

MuJoCo Warp keeps, per world, a sticky overflow bitmask on its ``Data``
(``mujoco_warp.OverflowType``). Two of its bits are solver budgets rather
than buffer sizes: ``ITERATIONS`` means the Newton loop stopped at
``iterations`` before reaching ``tolerance``, and ``LS_ITERATIONS`` means a
line search stopped at ``ls_iterations`` before meeting ``ls_tolerance``.
The solver then continues from where it stopped, so a hit is not an error:
it means that substep was integrated from an unconverged solve. The engine
only prints a warning when a budget is hit; this diag counts the hits per
control step and per world under random actions, reports every other
overflow bit it sees (those are buffer budgets, see ``newton_g1_dr_nan_diag``)
and times the step, so a budget change can be judged on convergence and
cost together.

Both MuJoCo Warp backends: ``--sim mujoco`` (mjlab) and ``--sim newton``
(``SolverMuJoCo``). The preset's own solver settings are used unless
``--iterations`` / ``--ls-iterations`` override them, through the same
config fields the presets set. The bitmask is cleared before every control
step, so a hit is attributed to the step whose substeps produced it. The
engine's warning print is switched off through the scene config's
``warn_overflow`` before the env is built: the kernels bake the flag in
when they are first built and captured, so a flag set later is ignored on
the GPU, and the print inside the captured graph would dominate the timing.

    python -m jaxrlworld.scripts.diag.perf.check_solver_convergence --preset g1_flat --sim mujoco
    python -m jaxrlworld.scripts.diag.perf.check_solver_convergence --preset g1_flat --sim mujoco --ls-iterations 50
    python -m jaxrlworld.scripts.diag.perf.check_solver_convergence --preset k1_velocity --sim newton --num-envs 8192
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import warp as wp
from mujoco_warp import OverflowType

from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _load

SOLVER_BITS = {"ITERATIONS": OverflowType.ITERATIONS, "LS_ITERATIONS": OverflowType.LS_ITERATIONS}
BUFFER_BITS = {
    name: value for name, value in OverflowType.__members__.items() if name not in ("NONE", "ALL", *SOLVER_BITS)
}


def _override(cfgs, sim: str, iterations: int | None, ls_iterations: int | None) -> None:
    if sim == "mujoco":
        cfgs.scene.warn_overflow = False
        if iterations is not None:
            cfgs.scene.solver_iterations = iterations
        if ls_iterations is not None:
            cfgs.scene.solver_ls_iterations = ls_iterations
    else:
        cfgs.scene.solver_cfg.warn_overflow = False
        if iterations is not None:
            cfgs.scene.solver_cfg.iterations = iterations
        if ls_iterations is not None:
            cfgs.scene.solver_cfg.ls_iterations = ls_iterations


def _mjw(env, sim: str):
    """The MuJoCo Warp ``Model`` and ``Data`` behind the env."""
    if sim == "mujoco":
        return env.scene_manager.sim.wp_model, env.scene_manager.sim.wp_data
    return env.scene_manager.solver.mjw_model, env.scene_manager.solver.mjw_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True, help="label from check_all_presets' table")
    ap.add_argument("--sim", required=True, choices=("mujoco", "newton"))
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--action-scale", type=float, default=0.5, help="std of the random normal actions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=None, help="override the preset's solver iterations")
    ap.add_argument("--ls-iterations", type=int, default=None, help="override the preset's line-search iterations")
    args = ap.parse_args()

    loader = next(loader for label, loader, sims in _PUBLIC if label == args.preset and args.sim in sims)
    cfgs = _load(loader, args.sim, args.num_envs)
    _override(cfgs, args.sim, args.iterations, args.ls_iterations)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    model, data = _mjw(env, args.sim)
    if model.opt.warn_overflow != 0:
        raise RuntimeError(f"warn_overflow is {model.opt.warn_overflow}; the scene config did not reach the model")
    env.reset()
    overflow = wp.to_torch(data.overflow)
    print(
        f"{args.preset}:{args.sim}  num_envs={env.num_envs}  steps={args.steps}  action std={args.action_scale}\n"
        f"solver budgets in effect: iterations={model.opt.iterations}  ls_iterations={model.opt.ls_iterations}  "
        f"tolerance={float(wp.to_torch(model.opt.tolerance)[0]):.1e}  "
        f"ls_tolerance={float(wp.to_torch(model.opt.ls_tolerance)[0]):.1e}"
    )

    gen = torch.Generator().manual_seed(args.seed)
    hits = {name: torch.zeros(env.num_envs, dtype=torch.int64) for name in SOLVER_BITS}
    per_step = {name: [] for name in SOLVER_BITS}
    buffer_seen = {name: 0 for name in BUFFER_BITS}
    times = []
    for _ in range(args.steps):
        actions = (torch.randn((env.num_envs, env.num_actions), generator=gen, device="cpu") * args.action_scale).to(
            env.device
        )
        overflow.zero_()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        bits = overflow.cpu()
        for name, bit in SOLVER_BITS.items():
            hit = (bits & int(bit)) != 0
            hits[name] += hit
            per_step[name].append(int(hit.sum()))
        for name, bit in BUFFER_BITS.items():
            buffer_seen[name] += int(((bits & int(bit)) != 0).sum())

    times.sort()
    total = args.steps * env.num_envs
    print(f"\nstep time: median {times[len(times) // 2]:.2f} ms, p90 {times[int(len(times) * 0.9)]:.2f} ms")
    print(
        f"{'bit':<14}{'(step,world) hit':>18}{'steps w/ hit':>14}{'worlds ever':>13}{'max/step':>10}{'mean/step':>11}"
    )
    for name in SOLVER_BITS:
        counts = per_step[name]
        print(
            f"{name:<14}{sum(counts) / total:>17.2%} {sum(c > 0 for c in counts):>8}/{args.steps:<5}"
            f"{int((hits[name] > 0).sum()):>8}/{env.num_envs:<5}{max(counts):>9}{sum(counts) / len(counts):>11.1f}"
        )
    flagged = {name: n for name, n in buffer_seen.items() if n}
    print(f"\nbuffer overflow bits seen (world-steps): {flagged if flagged else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
