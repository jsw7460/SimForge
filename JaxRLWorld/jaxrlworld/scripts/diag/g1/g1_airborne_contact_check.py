"""Airborne contact check: verify mjwarp reports nacon=0 and zero base-DOF
constraint force when the robot is in free fall.

NOTE: this script does NOT override the initial height at runtime (that path
was fragile and could hang due to warp/sim state desync). Instead, set the
robot's initial height in the preset BEFORE running this script:

    # JaxRLWorld/jaxrlworld/rl/configs/robots/g1_29dof.py
    base_init_height: float = 10.0   # was 0.793

Then run::

    jaxpy JaxRLWorld/jaxrlworld/scripts/diag/g1_airborne_contact_check.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim mujoco --n_steps 5

    jaxpy JaxRLWorld/jaxrlworld/scripts/diag/g1_airborne_contact_check.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim newton --n_steps 5

Compare side by side.

What it checks:
  - ``nacon``             — mjwarp's active contact count (``Data.nacon``,
                              summed over worlds; this script runs one world)
  - ``|qfrc_constraint[0:6]|`` — base 6 DOF (free joint) constraint force
  - ``|qfrc_constraint[6:]|``  — 29 hinge DOF constraint forces

If the robot is truly in the air (z=10), nacon must be 0, base must be 0,
and hinge values must stay within ±frictionloss (0.3 for G1). Any deviation
means the sim is generating phantom contacts even in free fall.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jaxrlworld.rl.evals.evaluator import PolicyEvaluator


def _np(x) -> np.ndarray:
    """Host copy of a raw mujoco_warp ``wp.array`` field."""
    return x.numpy()


def _get_mjw_data(env, sim_type: str):
    """Raw ``mujoco_warp.Data`` of the scene (mjlab's torch bridge unwrapped)."""
    sm = env.scene_manager
    if sim_type == "mujoco":
        return sm.sim.data.struct
    if sim_type == "newton":
        return sm.solver.mjw_data
    raise ValueError(sim_type)


def _first_row(x) -> np.ndarray:
    """World 0 of a ``(nworld, n)`` mujoco_warp field."""
    return _np(x)[0]


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
    print("  (initial root_z taken from preset; edit base_init_height to drop from high)")
    print("")

    hdr = (
        f"{'step':>4}  {'nacon':>5}  {'root_z':>8}  "
        f"{'|qcons[0:6]|max':>16}  {'|qcons[6:]|max':>16}  "
        f"{'hinge_over_fl':>14}  {'VERDICT':<10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for step in range(args.n_steps):
        # ``nacon`` is a 1-element array: the contact count over all
        # worlds, which is world 0's count with num_envs=1.
        nacon = int(_np(mjw.nacon)[0])
        qfrc = _first_row(mjw.qfrc_constraint)
        qpos = _first_row(mjw.qpos)
        root_z = float(qpos[2])

        base_max = float(np.abs(qfrc[0:6]).max())
        hinge = np.abs(qfrc[6:])
        hinge_max = float(hinge.max())
        hinge_over = int((hinge > args.frictionloss + 1e-4).sum())

        clean = (base_max < 1e-4) and (hinge_over == 0) and (nacon == 0)
        verdict = "CLEAN" if clean else "DIRTY"

        print(
            f"{step:>4}  {nacon:>5}  {root_z:>+8.3f}  "
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
