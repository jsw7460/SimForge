"""Systematic sim2sim divergence trace for T1 getup.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_trace.py --sim newton
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_trace.py --sim mujoco
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_trace.py --sim genesis

What it does (num_envs=1, deterministic, no DR, no fallen branch):
    1. Build env with ``fallen_prob=0`` so reset always produces the
       *standing* branch — identical deterministic initial state on
       every sim.
    2. Reset with fixed seed. Dump: root_pos, root_quat (wxyz), root
       linear + angular velocity (world + body frame), per-DOF
       joint_pos / joint_vel in canonical order,
       projected_gravity_b. Also dump actor-group obs tensor and the
       per-term breakdown so obs-layout drift surfaces immediately.
    3. Step 5 times with a fixed deterministic action vector (ramp
       [0, 1/N, 2/N, ...]). Dump the same quantities after each step.
       With IdealPDActuator the torque is Python-computed — we dump
       it too so any gain/armature mismatch shows up at step 1.

Then diff the three sim outputs. The first mismatching field is
precisely the bug.
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def _fmt(t: torch.Tensor, prec: int = 6) -> str:
    """Single-line string representation for (1, D) or (D,) tensor."""
    v = t.detach().flatten().cpu().numpy()
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in v) + "]"


def _dump_state(env, tag: str) -> None:
    rd = env.robot_data
    am = env.act_manager

    print(f"\n--- {tag} ---")
    print(f"root_pos_w         = {_fmt(rd.root_link_pos_w)}")
    print(f"root_quat_w (wxyz) = {_fmt(rd.root_link_quat_w)}")
    print(f"root_lin_vel_w     = {_fmt(rd.root_link_lin_vel_w)}")
    print(f"root_ang_vel_w     = {_fmt(rd.root_link_ang_vel_w)}")
    if hasattr(rd, "root_link_lin_vel_b"):
        print(f"root_lin_vel_b     = {_fmt(rd.root_link_lin_vel_b)}")
    if hasattr(rd, "root_link_ang_vel_b"):
        print(f"root_ang_vel_b     = {_fmt(rd.root_link_ang_vel_b)}")
    if hasattr(rd, "projected_gravity_b"):
        print(f"projected_gravity_b= {_fmt(rd.projected_gravity_b)}")
    print(f"joint_pos          = {_fmt(rd.joint_pos)}")
    print(f"joint_vel          = {_fmt(rd.joint_vel)}")

    # Full actor obs tensor + per-term breakdown (obs ordering audit)
    try:
        obs_mgr = env.obs_manager
        for group_name in ("actor", "critic"):
            group = obs_mgr.get_group(group_name) if hasattr(obs_mgr, "get_group") else None
            if group is None:
                continue
            print(f"\n[{group_name}] per-term:")
            offset = 0
            for name in group.term_names:
                t = group.terms[name].last_obs
                if t is None:
                    continue
                dim = t.shape[-1]
                print(f"  {offset:>3}..{offset + dim:<3} {name:<28} {_fmt(t, prec=4)}")
                offset += dim
    except Exception as e:
        print(f"[obs dump skipped: {e}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("newton", "genesis", "mujoco"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--action-scale", type=float, default=0.0, help="0 = zero action; positive = small ramp to exercise torque."
    )
    args = parser.parse_args()

    # fallen_prob=0 forces the standing branch — identical starting pose
    cfg = T1GetupConfig(
        sim_type=args.sim,
        num_envs=1,
        seed=args.seed,
        fallen_prob=0.0,
        fall_velocity_range=(0.0, 0.0),
        standing_z_offset=0.0,
    )
    cfgs = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    # Zero action; or small deterministic ramp.
    action_dim = env.act_manager.total_action_dim
    if args.action_scale > 0:
        ramp = torch.linspace(0, args.action_scale, action_dim, device=env.device)
        action = ramp.unsqueeze(0).expand(env.num_envs, -1).contiguous()
    else:
        action = torch.zeros(env.num_envs, action_dim, device=env.device)

    # Reset
    env.reset()
    _dump_state(env, f"[{args.sim}] post-reset (t=0)")

    # Step trace
    for t in range(1, args.steps + 1):
        env.step(action)
        _dump_state(env, f"[{args.sim}] after step {t}  (action={_fmt(action[:1], prec=3)})")


if __name__ == "__main__":
    main()
