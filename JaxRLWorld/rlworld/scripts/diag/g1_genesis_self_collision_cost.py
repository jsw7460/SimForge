"""How much of Genesis's G1 engine step is self-collision?

G1 at 4096 envs, measured with ``collect_breakdown``: the engine's own
substep costs 4.55 ms on Genesis against 2.59 on mjlab, and that single
line is 55% of the whole 14.2 ms gap between the two backends' env.step.

Five configuration hypotheses (cone, impratio, solver, iterations,
self-collision) were tested against that gap on K1 and the yam arm and
every one came back null, which is recorded as "do not retry". But G1 is
not those robots. It is the only family that runs
``enable_self_collision=True``, and it carries 34 collision geoms, so
the pair count it hands the broad phase is unlike anything the earlier
sweep covered. That makes this one hypothesis untested rather than
disproved, and it is the only untested one left on a 55% target.

Answering it does not require deciding anything: self-collision is
load-bearing for G1 (a reward reads the ``self_collision`` sensor), so
this is not a proposal to turn it off. It is a question about where the
time is, and the answer opens or closes a direction -- if most of the
engine's cost is self-collision, narrowing it to the links that can
actually meet is worth designing; if it is not, the engine cost is the
engine and the search ends here.

Reports, for each setting: candidate collision pairs the collider built,
the contact-list width that follows from them, and the wall time of the
engine step alone.

Usage:
    python -m rlworld.scripts.diag.g1_genesis_self_collision_cost
    python -m rlworld.scripts.diag.g1_genesis_self_collision_cost --num-envs 8192
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import torch  # noqa: E402


def build(num_envs: int, self_collision: bool):
    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.runners import BaseRunner

    cfgs = G1FlatConfig(sim_type="genesis", num_envs=num_envs, seed=0).build()
    cfgs.scene.rigid_options.enable_self_collision = self_collision
    return BaseRunner._create_env_from_config(cfgs)


def measure(env, steps: int, warmup: int) -> dict:
    """Time the engine's substep on its own, drained on both sides."""
    zero = torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device)
    for _ in range(warmup):
        env.step(zero)
    torch.cuda.synchronize()

    solver = env.scene_manager.scene.sim.rigid_solver
    pairs = int(solver.collider._n_possible_pairs)
    width = int(solver.collider._collider_state.contact_data.geom_a.shape[0])

    # The engine alone, with the substep loop's other work left out.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.scene_manager.step()
    torch.cuda.synchronize()
    engine = (time.perf_counter() - t0) / steps * 1e3

    # And the whole control step, so the engine share is visible.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(zero)
    torch.cuda.synchronize()
    full = (time.perf_counter() - t0) / steps * 1e3

    return {"pairs": pairs, "width": width, "engine_ms": engine, "full_ms": full}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    print("=" * 78)
    print("  HOW MUCH OF GENESIS'S G1 ENGINE STEP IS SELF-COLLISION")
    print("=" * 78)
    print(f"  {args.num_envs} envs, {args.steps} steps after {args.warmup} warmup")
    print("  reference from collect_breakdown: engine substep 4.55 ms genesis, 2.59 mjlab")

    out = {}
    for flag in (True, False):
        env = build(args.num_envs, flag)
        out[flag] = measure(env, args.steps, args.warmup)
        del env

    print()
    print(f"  {'':<30}{'self-collision ON':>20}{'OFF':>14}")
    for key, label in (
        ("pairs", "candidate collision pairs"),
        ("width", "contact-list width"),
        ("engine_ms", "engine substep (ms)"),
        ("full_ms", "whole control step (ms)"),
    ):
        a, b = out[True][key], out[False][key]
        fmt = "{:>20.3f}{:>14.3f}" if isinstance(a, float) else "{:>20}{:>14}"
        print(f"  {label:<30}" + fmt.format(a, b))

    saved = out[True]["engine_ms"] - out[False]["engine_ms"]
    gap = 4.55 - 2.59
    print()
    print(f"  self-collision costs {saved:.3f} ms of the engine substep")
    print(f"  the genesis-vs-mjlab engine gap is {gap:.2f} ms, so it explains {100 * saved / gap:.0f}% of it")
    print()
    if saved / max(gap, 1e-9) > 0.4:
        print("  Worth designing around: narrow self-collision to the links that can")
        print("  actually meet, rather than every pair of 34 geoms.")
    else:
        print("  Not the lever. The engine's cost is the engine, and the five")
        print("  hypotheses already ruled out on K1 and the arm now cover G1 too.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
