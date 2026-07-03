"""Verify RobotData.applied_torque is live for EXPLICIT actuators on every backend.

Explicit actuators (Go2 preset: IdealPD/DelayedPD on all sims) compute torque in
Python. Per backend the torque reaches the sim differently and so does the
readback:

  * Genesis: control_dofs_force -> get_dofs_control_force        (sim readback)
  * mjlab:   set_joint_effort_target -> motor -> qfrc_actuator   (sim readback)
  * Newton:  control.joint_f (NO mjwarp actuator, qfrc_actuator ~0)
             -> RobotData now returns act_manager.applied_torque (the exact
             tensor written to joint_f). Previously this read ~0 and silently
             disabled the energy termination / power-penalty reward on Newton.

This diag builds Go2 (explicit PD), steps with a small sinusoid action, and
prints per-sim: |applied_torque| stats from RobotData, the Python-side
act_manager.applied_torque, their max abs difference, and the mechanical power
sum(|tau * qd|). PASS if RobotData torque is non-trivial and (for Newton)
matches the Python tensor exactly.

Run (GPU box):
    jaxpy rlworld/scripts/diag/applied_torque_smoke.py --sim genesis
    jaxpy rlworld/scripts/diag/applied_torque_smoke.py --sim newton
    jaxpy rlworld/scripts/diag/applied_torque_smoke.py --sim mujoco
"""

from __future__ import annotations

import argparse
import math

import torch

from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.runners import BaseRunner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"], default="newton")
    args = ap.parse_args()

    cfg = Go2FlatConfig(sim_type=args.sim, num_envs=2)
    env = BaseRunner.create_with_env(cfg.build()).env
    env.reset()

    print("=" * 74)
    print(f"APPLIED-TORQUE SMOKE  [sim={args.sim}]  has_explicit={env.act_manager.has_explicit_actuators}")
    print("=" * 74)
    if not env.act_manager.has_explicit_actuators:
        print("FAIL: Go2 preset unexpectedly has no explicit actuators")
        return 1

    # Step with a small sinusoid so PD errors (and thus torques) are non-trivial.
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for step in range(30):
        actions.fill_(0.3 * math.sin(step / 5.0))
        env.step(actions)

    rd = env.get_robot_data("robot")
    rd_tau = rd.applied_torque
    py_tau = env.act_manager.applied_torque
    qd = rd.joint_vel

    print(f"[RobotData]   applied_torque |mean|={rd_tau.abs().mean():.4f}  max={rd_tau.abs().max():.4f}")
    print(f"[ActManager]  applied_torque |mean|={py_tau.abs().mean():.4f}  max={py_tau.abs().max():.4f}")
    max_diff = (rd_tau - py_tau).abs().max()
    print(f"[compare]     max |RobotData - ActManager| = {max_diff:.6f}")
    power = (rd_tau * qd).abs().sum(dim=1)
    print(f"[power]       sum|tau*qd| per env = {power.tolist()}")

    results = {
        # The old Newton bug read ~0 here; any healthy PD run has |mean| >> 0.01.
        "torque_nonzero": bool(rd_tau.abs().mean() > 1e-2),
        "power_nonzero": bool((power > 1e-2).all()),
    }
    if args.sim == "newton":
        # Newton explicit returns the act_manager tensor itself — exact match.
        results["matches_python_exactly"] = bool(max_diff == 0.0)
    else:
        # Genesis/mjlab read back from the sim; allow clipping/timing tolerance
        # but they must be in the same ballpark as the commanded torque.
        results["matches_python_approx"] = bool(max_diff < 0.5 * max(py_tau.abs().max().item(), 1.0))

    print("-" * 74)
    for k, v in results.items():
        print(f"  {k:24s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"  OVERALL                 : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
