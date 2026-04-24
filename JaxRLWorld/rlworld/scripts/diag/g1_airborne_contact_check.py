"""Airborne contact check: verify mjwarp reports ncon=0 and zero base-DOF
constraint force when the robot is in free fall.

NOTE: this script does NOT override the initial height at runtime (that path
was fragile and could hang due to warp/sim state desync). Instead, set the
robot's initial height in the preset BEFORE running this script:

    # JaxRLWorld/rlworld/rl/configs/robots/g1_29dof.py
    base_init_height: float = 10.0   # was 0.793

Then run::

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_airborne_contact_check.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim mujoco --n_steps 5

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_airborne_contact_check.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim newton --n_steps 5

Compare side by side.

What it checks:
  - ``ncon``              — mjwarp's active contact count
  - ``|qfrc_constraint[0:6]|`` — base 6 DOF (free joint) constraint force
  - ``|qfrc_constraint[6:]|``  — 29 hinge DOF constraint forces

If the robot is truly in the air (z=10), ncon must be 0, base must be 0,
and hinge values must stay within ±frictionloss (0.3 for G1). Any deviation
means the sim is generating phantom contacts even in free fall.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rlworld.rl.evals.evaluator import PolicyEvaluator


def _np(x):
    if x is None:
        return None
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)


def _get_mjw_data(env, sim_type: str):
    sm = env.scene_manager
    if sim_type == "mujoco":
        return sm.sim.data
    if sim_type == "newton":
        return sm.solver.mjw_data
    raise ValueError(sim_type)


def _scalar_int(x) -> int:
    arr = _np(x)
    if arr is None:
        return -1
    if arr.ndim == 0:
        return int(arr)
    return int(arr.flat[0])


def _first_row(x) -> np.ndarray:
    arr = _np(x)
    if arr is None:
        return np.array([])
    if arr.ndim == 2:
        return arr[0]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--eval_sim", required=True, choices=("mujoco", "newton"))
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument(
        "--frictionloss",
        type=float,
        default=0.3,
        help="Hinge frictionloss limit (default matches G1 preset 0.3)",
    )
    args = parser.parse_args()

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        extra_overrides={"env": {"num_envs": 1}},
        record_video=False,
    )
    env = evaluator.env
    env.reset()

    num_actions = env.act_manager.num_actions
    zero_action = torch.zeros(1, num_actions, device=env.device)

    mjw = _get_mjw_data(env, args.eval_sim)

    print(f"=== {args.eval_sim} airborne contact check ===")
    print(f"  n_steps={args.n_steps}  frictionloss_limit={args.frictionloss}")
    print(f"  (initial root_z taken from preset; edit base_init_height to drop from high)")
    print("")

    hdr = (
        f"{'step':>4}  {'ncon':>5}  {'root_z':>8}  "
        f"{'|qcons[0:6]|max':>16}  {'|qcons[6:]|max':>16}  "
        f"{'hinge_over_fl':>14}  {'VERDICT':<10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for step in range(args.n_steps):
        ncon = _scalar_int(getattr(mjw, "ncon", None))
        qfrc = _first_row(mjw.qfrc_constraint)
        qpos = _first_row(mjw.qpos)
        root_z = float(qpos[2]) if qpos.size >= 3 else float("nan")

        base_max = float(np.abs(qfrc[0:6]).max()) if qfrc.size >= 6 else float("nan")
        hinge = np.abs(qfrc[6:]) if qfrc.size > 6 else np.array([])
        hinge_max = float(hinge.max()) if hinge.size else float("nan")
        hinge_over = int((hinge > args.frictionloss + 1e-4).sum()) if hinge.size else -1

        clean = (base_max < 1e-4) and (hinge_over == 0) and (ncon == 0 or ncon == -1)
        verdict = "CLEAN" if clean else "DIRTY"

        print(
            f"{step:>4}  {ncon:>5}  {root_z:>+8.3f}  "
            f"{base_max:>+16.6f}  {hinge_max:>+16.6f}  "
            f"{hinge_over:>14}  {verdict:<10}"
        )

        env.step(zero_action)

    print("")
    print("Interpretation:")
    print("  CLEAN  → no phantom contact force (free fall as expected).")
    print("  DIRTY  → sim is generating spurious constraint force in the air.")


if __name__ == "__main__":
    main()
