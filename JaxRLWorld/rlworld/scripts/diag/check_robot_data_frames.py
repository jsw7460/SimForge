"""Verify ``RobotData`` link-origin vs center-of-mass pose/velocity.

For each requested simulator this:

  1. builds a preset (small ``num_envs`` — we only read env 0), resets, and
     steps a few times with a fixed joint-target perturbation so the robot is
     actually moving;
  2. reads ``env.get_robot_data()`` and prints, for env 0, the root link- and
     CoM-referenced pose / linear velocity (world and body frame);
  3. runs **internal-consistency** checks that don't need cross-sim state
     matching — these are the real test of the CoM<->origin transform + sign:

       offset            := root_com_pos_w - root_link_pos_w        (== R @ c)
       transform_residual := ‖ (root_com_lin_vel_w - root_link_lin_vel_w)
                                - cross(root_link_ang_vel_w, offset) ‖   → ~0
       *_b_rotation_resid := ‖ root_*_lin_vel_b - R(quat)^T @ root_*_lin_vel_w ‖ → ~0

  4. compares ``offset`` across simulators — it's a model property (the root
     body's CoM offset), so it must match for the same robot.

Bit-exact cross-sim velocity comparison after stepping isn't possible (the
solvers diverge), but ``offset`` matching + every per-sim residual being ~0
proves the implementation is consistent.

Usage::

    python -m rlworld.scripts.diag.check_robot_data_frames                  # go2_flat, all 3 sims
    python -m rlworld.scripts.diag.check_robot_data_frames --preset g1_29dof
    python -m rlworld.scripts.diag.check_robot_data_frames --sim genesis    # one sim only
    python -m rlworld.scripts.diag.check_robot_data_frames --num-envs 4 --steps 60
"""

from __future__ import annotations

import argparse

import numpy as np

# preset key -> (module path, config class name)
_PRESETS: dict[str, tuple[str, str]] = {
    "go2_flat": ("rlworld.rl.configs.presets.go2_flat.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")

_FIELDS = (
    "root_link_pos_w",
    "root_link_quat_w",
    "root_link_ang_vel_w",
    "root_link_lin_vel_w",
    "root_link_lin_vel_b",
    "root_com_pos_w",
    "root_com_lin_vel_w",
    "root_com_lin_vel_b",
)


def _qrot_inv_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)^T @ v — mirrors quat_utils.quat_rotate_inverse_wxyz (a - b + c)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qv = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (qv @ v) * 2.0
    return a - b + c


def _build_env(preset: str, sim: str, num_envs: int):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _probe(env, settle: int, steps: int, perturb: float) -> dict[str, np.ndarray]:
    import torch

    env.reset()
    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    for _ in range(settle):
        env.step(zero)
    pert = perturb * torch.ones(env.num_envs, n_act, device=env.device)
    for _ in range(steps):
        env.step(pert)
    rd = env.get_robot_data()
    return {name: getattr(rd, name)[0].detach().cpu().double().numpy() for name in _FIELDS}


def _checks(d: dict[str, np.ndarray]) -> dict[str, float]:
    offset = d["root_com_pos_w"] - d["root_link_pos_w"]  # == R @ c (world)
    pred = np.cross(d["root_link_ang_vel_w"], offset)  # ω × (R@c)
    got = d["root_com_lin_vel_w"] - d["root_link_lin_vel_w"]  # should equal `pred`
    q = d["root_link_quat_w"]
    return {
        "com_offset_norm": float(np.linalg.norm(offset)),
        "ang_vel_norm": float(np.linalg.norm(d["root_link_ang_vel_w"])),
        "link_lin_vel_norm": float(np.linalg.norm(d["root_link_lin_vel_w"])),
        "transform_residual": float(np.linalg.norm(got - pred)),
        "link_b_rotation_residual": float(
            np.linalg.norm(d["root_link_lin_vel_b"] - _qrot_inv_wxyz(q, d["root_link_lin_vel_w"]))
        ),
        "com_b_rotation_residual": float(
            np.linalg.norm(d["root_com_lin_vel_b"] - _qrot_inv_wxyz(q, d["root_com_lin_vel_w"]))
        ),
    }


def _arr(v: np.ndarray) -> str:
    return np.array2string(v, precision=4, suppress_small=True, sign=" ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="go2_flat")
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--settle", type=int, default=5, help="zero-action steps before perturbing")
    ap.add_argument("--steps", type=int, default=30, help="perturbed steps before reading state")
    ap.add_argument("--perturb", type=float, default=0.4, help="constant action applied during the perturbed steps")
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    results: dict[str, tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    bad = False

    for sim in sims:
        print(f"\n{'=' * 64}\n[{sim}] building {args.preset!r} (num_envs={args.num_envs}) ...")
        env = _build_env(args.preset, sim, args.num_envs)
        d = _probe(env, args.settle, args.steps, args.perturb)
        c = _checks(d)
        results[sim] = (d, c)

        print(f"[{sim}] env 0 RobotData (after {args.settle} zero + {args.steps} × {args.perturb} steps):")
        for name in _FIELDS:
            print(f"    {name:<22s} {_arr(d[name])}")
        # tolerance: residuals must be tiny relative to the velocity magnitude in play.
        scale = max(c["link_lin_vel_norm"], c["ang_vel_norm"] * max(c["com_offset_norm"], 1e-3), 1e-2)
        tol = 1e-3 * scale + 1e-5
        print(f"[{sim}] consistency (tol ≈ {tol:.3g}):")
        for name, val in c.items():
            note = ""
            if name in ("transform_residual", "link_b_rotation_residual", "com_b_rotation_residual"):
                ok = val < tol
                note = "  OK" if ok else "  ⚠ FAIL"
                bad = bad or not ok
            print(f"    {name:<28s} {val:.6f}{note}")
        if c["com_offset_norm"] < 1e-4:
            print(
                f"    NOTE: root CoM ≈ link frame origin for {args.preset!r} — the split is a near no-op here; "
                "try --preset g1_29dof (larger pelvis CoM offset) for a more telling test."
            )

    if len(results) > 1:
        print(f"\n{'=' * 64}\nCROSS-SIM (env 0):")
        print("  offset = root_com_pos_w - root_link_pos_w  — a model property; must match across sims:")
        ref = None
        for sim in sims:
            off = results[sim][0]["root_com_pos_w"] - results[sim][0]["root_link_pos_w"]
            print(f"    {sim:<10s} {_arr(off)}")
            if ref is None:
                ref = off
            elif np.linalg.norm(off - ref) > 1e-3 + 1e-3 * np.linalg.norm(ref):
                bad = True
        print(
            "  root_link_lin_vel_b / root_com_lin_vel_b  — diverge after stepping (different solvers), "
            "should be the same ballpark / direction:"
        )
        for sim in sims:
            d = results[sim][0]
            print(f"    {sim:<10s} link={_arr(d['root_link_lin_vel_b'])}   com={_arr(d['root_com_lin_vel_b'])}")

    print(
        "\n→ Every per-sim transform_residual / *_b_rotation_residual should be ~0, and `offset` "
        "should match across sims. If a residual FAILs, the CoM<->origin transform (sign or layout) "
        "is wrong in that sim's robot_data.py — report the numbers."
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
