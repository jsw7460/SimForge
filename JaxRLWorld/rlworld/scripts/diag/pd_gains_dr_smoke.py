"""Verify randomize_pd_gains reaches the EXPLICIT actuator gains on every backend.

Explicit actuators (IdealPD / DelayedPD / ...) compute torque in Python from
their own stiffness/damping tensors, and the sim-side PD store is zeroed or
unused. ``randomize_pd_gains`` therefore must mutate the actuator tensors —
previously only the Newton backend did, and on Genesis/mjlab the term silently
randomized the unused sim store (a no-op for every shipping preset).

For each requested sim this builds Go2 (explicit PD actuators, 4 envs) and
checks, printing everything:

  1. baseline per-env mean stiffness/damping of every explicit actuator,
  2. after ``randomize_pd_gains(env_ids=[0,1], scale 0.5-1.5)``:
     envs 0/1 changed, envs 2/3 untouched,
  3. after a second call: values stay within base*range (scale is applied to
     the CAPTURED base, not compounded onto the previous sample).

Run (GPU box):
    jaxpy rlworld/scripts/diag/pd_gains_dr_smoke.py --sim genesis
    jaxpy rlworld/scripts/diag/pd_gains_dr_smoke.py --sim newton
    jaxpy rlworld/scripts/diag/pd_gains_dr_smoke.py --sim mujoco
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.envs.mdp.events.dr.unified import randomize_pd_gains
from rlworld.rl.runners import BaseRunner

KP_RANGE = (0.5, 1.5)
KD_RANGE = (0.5, 1.5)


def gain_snapshot(env) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """[(actuator_name, per-env mean stiffness, per-env mean damping)]"""
    out = []
    for actuator, _joint_idx in env.act_manager.actuators:
        out.append(
            (type(actuator).__name__, actuator.stiffness.mean(dim=1).clone(), actuator.damping.mean(dim=1).clone())
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"], default="genesis")
    args = ap.parse_args()

    cfg = Go2FlatConfig(sim_type=args.sim, num_envs=4)
    env = BaseRunner.create_with_env(cfg.build()).env
    env.reset()

    print("=" * 74)
    print(f"PD-GAINS DR SMOKE  [sim={args.sim}]  has_explicit={env.act_manager.has_explicit_actuators}")
    print("=" * 74)
    if not env.act_manager.has_explicit_actuators:
        print("FAIL: preset unexpectedly has no explicit actuators — test is vacuous")
        return 1

    results: dict[str, bool] = {}
    dr_ids = torch.tensor([0, 1], device=env.device)

    base = gain_snapshot(env)
    for name, kp, kd in base:
        print(f"[base]   {name:<18} kp/env={kp.tolist()}  kd/env={kd.tolist()}")

    randomize_pd_gains(env, dr_ids, kp_range=KP_RANGE, kd_range=KD_RANGE, operation="scale")
    after1 = gain_snapshot(env)
    changed = untouched = True
    for (name, kp0, kd0), (_, kp1, kd1) in zip(base, after1):
        print(f"[call#1] {name:<18} kp/env={kp1.tolist()}  kd/env={kd1.tolist()}")
        changed &= bool((kp1[:2] != kp0[:2]).all() and (kd1[:2] != kd0[:2]).all())
        untouched &= bool(torch.equal(kp1[2:], kp0[2:]) and torch.equal(kd1[2:], kd0[2:]))
    results["dr_envs_changed"] = changed
    results["other_envs_untouched"] = untouched

    randomize_pd_gains(env, dr_ids, kp_range=KP_RANGE, kd_range=KD_RANGE, operation="scale")
    after2 = gain_snapshot(env)
    in_band = True
    for (name, kp0, _), (_, kp2, _) in zip(base, after2):
        lo, hi = kp0[:2] * KP_RANGE[0], kp0[:2] * KP_RANGE[1]
        print(f"[call#2] {name:<18} kp/env={kp2.tolist()}  band=[{lo.tolist()}, {hi.tolist()}]")
        in_band &= bool(((kp2[:2] >= lo - 1e-5) & (kp2[:2] <= hi + 1e-5)).all())
    results["no_compounding"] = in_band

    print("-" * 74)
    for k, v in results.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"  OVERALL               : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
