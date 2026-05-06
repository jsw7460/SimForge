"""Deterministic step-by-step state/torque trace for T1 getup sim2sim audit.

Design:
  1. Force a KNOWN fallen pose via the robot state writer after reset so
     both sims start from byte-identical state. Eliminates reset-random
     divergence.
  2. Either use zero action, a fixed deterministic pattern, or a loaded
     policy checkpoint. Same action → same ideal-PD torque math → any
     state divergence is purely physics.
  3. Every step dumps in fixed-width columns: joint_pos, joint_vel,
     applied_torque (from IdealPDActuator), raw_action, processed_action,
     root_pos, root_quat, body-frame velocities, projected_gravity,
     plus a header showing the canonical joint name order and per-joint
     PD gains / scale / offset.
  4. Output is deliberately long and machine-diffable; use
     ``diff -u a.txt b.txt`` to spot first divergence.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_trace_v2.py \\
        --sim newton --steps 40 > /tmp/trace_newton.txt 2>&1
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_trace_v2.py \\
        --sim mujoco --steps 40 > /tmp/trace_mujoco.txt 2>&1
    diff -u /tmp/trace_mujoco.txt /tmp/trace_newton.txt | head -200
"""

from __future__ import annotations

import argparse
import math

import torch

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner

# ─── Hardcoded deterministic fallen pose (wxyz, z=0.8, face-down tilt) ──

# Root position: dropped from 0.8m
KNOWN_ROOT_POS = (0.0, 0.0, 0.80)
# Root quaternion (wxyz): pitch 60° forward  → tilted face-down
_PITCH = math.radians(60.0)
KNOWN_ROOT_QUAT = (math.cos(_PITCH / 2), 0.0, math.sin(_PITCH / 2), 0.0)
# All velocities zero
KNOWN_ROOT_LIN_VEL = (0.0, 0.0, 0.0)
KNOWN_ROOT_ANG_VEL = (0.0, 0.0, 0.0)


def _fmt_row(values, prec: int = 5) -> str:
    return " ".join(f"{float(v):+.{prec}f}" for v in values)


def _tensor_to_list(t):
    if isinstance(t, torch.Tensor):
        return t.detach().flatten().cpu().tolist()
    return list(t)


def _print_header(env) -> None:
    am = env.act_manager
    rd = env.robot_data

    print("=" * 100)
    print(f"num_envs = {env.num_envs}")
    print(f"device   = {env.device}")
    print(f"sim_type = {getattr(env, 'sim_type', '?')}")
    print(f"total_action_dim = {am.total_action_dim}")

    # Canonical joint order — the single source of truth for all
    # per-joint obs/action indexing.
    print("\n[canonical joint order]")
    print("idx  joint_name                        sim_index  scale       offset")
    scales = _tensor_to_list(am._scale)
    offsets = _tensor_to_list(am._offset[0])
    sim_ids = _tensor_to_list(am._indexing.sim_indices)
    for i, name in enumerate(am.actuated_joint_names):
        print(f"{i:<4} {name:<32} {int(sim_ids[i]):<9}  {scales[i]:+.6f}  {offsets[i]:+.6f}")

    # Soft joint limits (in canonical order)
    try:
        lo = _tensor_to_list(am._indexing.joint_limits_lower)
        hi = _tensor_to_list(am._indexing.joint_limits_upper)
        print("\n[joint limits (canonical order)]")
        print("idx  joint_name                        lower       upper")
        for i, name in enumerate(am.actuated_joint_names):
            print(f"{i:<4} {name:<32} {lo[i]:+.5f}   {hi[i]:+.5f}")
    except Exception as e:
        print(f"[joint limits dump skipped: {e}]")

    # IdealPD gains if present (look through _actuators list)
    print("\n[actuator PD gains (if IdealPD-style)]")
    if not am._actuators:
        print("  (no explicit actuators — Implicit mode, PD is sim-side)")
    else:

        def _first_attr(obj, *names):
            for n in names:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
            return None

        def _fmt_maybe_tensor(v, prec=4):
            if v is None:
                return "None"
            if torch.is_tensor(v):
                flat = v.detach().flatten().cpu().tolist()
                return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in flat) + "]"
            if hasattr(v, "__len__") and not isinstance(v, str):
                return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in v) + "]"
            return str(v)

        for idx, (inst, joint_ids) in enumerate(am._actuators):
            kp = _first_attr(inst, "stiffness", "_stiffness", "kp", "_kp")
            kd = _first_attr(inst, "damping", "_damping", "kd", "_kd")
            ids = _tensor_to_list(joint_ids) if torch.is_tensor(joint_ids) else list(joint_ids)
            print(f"  actuator[{idx}] type={type(inst).__name__} joint_ids={ids}")
            print(f"    kp = {_fmt_maybe_tensor(kp)}")
            print(f"    kd = {_fmt_maybe_tensor(kd)}")
    print("=" * 100)


def _force_state(env, pos, quat_wxyz, lin_vel, ang_vel, joint_pos_vec):
    """Overwrite all envs' root + joint state to a known deterministic pose."""
    writer = env.get_robot_state_writer("robot")
    env_ids = torch.arange(env.num_envs, device=env.device)
    N = env.num_envs

    p = torch.tensor(pos, dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    q = torch.tensor(quat_wxyz, dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    lv = torch.tensor(lin_vel, dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    av = torch.tensor(ang_vel, dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    jp = joint_pos_vec.to(env.device).unsqueeze(0).expand(N, -1).contiguous()
    jv = torch.zeros_like(jp)

    writer.set_root_pose(p, q, env_ids=env_ids)
    writer.set_root_velocity(lv, av, env_ids=env_ids)
    writer.set_dof_positions(jp, env_ids=env_ids)
    writer.set_dof_velocities(jv, env_ids=env_ids)
    if hasattr(writer, "eval_fk"):
        writer.eval_fk(env_ids=env_ids)


def _dump_step(env, step_idx: int, raw_action) -> None:
    am = env.act_manager
    rd = env.robot_data

    # Take env 0 for single-line prints
    def g(x):
        if isinstance(x, torch.Tensor):
            return x[0].detach().cpu().tolist() if x.dim() > 1 else x.detach().cpu().tolist()
        return list(x)

    print(f"\n── step {step_idx} ──")
    print(f"raw_action       [{len(_tensor_to_list(raw_action[0])):2d}] = {_fmt_row(g(raw_action))}")

    if hasattr(am, "_processed_action_history") and am._processed_action_history:
        pa = am._processed_action_history[0]
        print(f"processed_action [{pa.shape[-1]:2d}] = {_fmt_row(g(pa))}")

    if hasattr(am, "applied_torque"):
        tq = am.applied_torque
        print(f"applied_torque   [{tq.shape[-1]:2d}] = {_fmt_row(g(tq))}")
    elif hasattr(am, "_applied_torque"):
        tq = am._applied_torque
        print(f"applied_torque   [{tq.shape[-1]:2d}] = {_fmt_row(g(tq))}")

    print(f"joint_pos        [{rd.joint_pos.shape[-1]:2d}] = {_fmt_row(g(rd.joint_pos))}")
    print(f"joint_vel        [{rd.joint_vel.shape[-1]:2d}] = {_fmt_row(g(rd.joint_vel))}")

    print(f"root_pos_w       [ 3] = {_fmt_row(g(rd.root_link_pos_w))}")
    print(f"root_quat_w wxyz [ 4] = {_fmt_row(g(rd.root_link_quat_w))}")
    print(f"root_lin_vel_w   [ 3] = {_fmt_row(g(rd.root_link_lin_vel_w))}")
    print(f"root_ang_vel_w   [ 3] = {_fmt_row(g(rd.root_link_ang_vel_w))}")
    if hasattr(rd, "root_link_lin_vel_b"):
        print(f"root_lin_vel_b   [ 3] = {_fmt_row(g(rd.root_link_lin_vel_b))}")
    if hasattr(rd, "root_link_ang_vel_b"):
        print(f"root_ang_vel_b   [ 3] = {_fmt_row(g(rd.root_link_ang_vel_b))}")
    if hasattr(rd, "projected_gravity_b"):
        print(f"proj_gravity_b   [ 3] = {_fmt_row(g(rd.projected_gravity_b))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("newton", "genesis", "mujoco"), required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action", choices=("zero", "ramp"), default="zero")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path. Defaults to ./t1_trace_<sim>.txt in the "
        "current working directory. Pass 'stdout' to print to console.",
    )
    args = parser.parse_args()

    # Redirect all prints to a file (unless --out stdout)
    out_path = args.out or f"t1_trace_{args.sim}.txt"
    if out_path != "stdout":
        import sys as _sys

        _log_fh = open(out_path, "w")
        _orig_stdout = _sys.stdout
        _sys.stdout = _log_fh
        print(f"# t1_trace_v2 sim={args.sim} steps={args.steps} seed={args.seed} action={args.action}")
    else:
        _log_fh = None

    torch.manual_seed(args.seed)

    cfg = T1GetupConfig(
        sim_type=args.sim,
        num_envs=1,
        seed=args.seed,
        fallen_prob=0.0,  # avoid random branch — we'll overwrite anyway
        fall_velocity_range=(0.0, 0.0),
        standing_z_offset=0.0,
    )
    cfgs = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    _print_header(env)

    # Reset first (initialises counters, fires startup DR, etc.)
    env.reset()

    # Build canonical joint_pos vector from T1Config.default_joint_angles
    # using the same pattern-resolve as everywhere else.
    from rlworld.rl.utils import string as string_utils

    matched_idx, _, matched_vals = string_utils.resolve_matching_names_values(
        cfg.robot.default_joint_angles, list(env.act_manager.actuated_joint_names)
    )
    known_jp = torch.zeros(env.act_manager.total_action_dim, dtype=torch.float32)
    for i, v in zip(matched_idx, matched_vals):
        known_jp[i] = float(v)

    # Force known deterministic state AFTER reset so both sims are byte-identical.
    _force_state(
        env,
        pos=KNOWN_ROOT_POS,
        quat_wxyz=KNOWN_ROOT_QUAT,
        lin_vel=KNOWN_ROOT_LIN_VEL,
        ang_vel=KNOWN_ROOT_ANG_VEL,
        joint_pos_vec=known_jp,
    )

    # Also reset episode counter so settle_steps logic starts fresh
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf[:] = 0
    if hasattr(env.act_manager, "_raw_action_history"):
        for h in env.act_manager._raw_action_history:
            h.zero_()
    if hasattr(env.act_manager, "_processed_action_history"):
        for h in env.act_manager._processed_action_history:
            h.zero_()

    # Action tensor
    action_dim = env.act_manager.total_action_dim
    if args.action == "ramp":
        base = torch.linspace(-0.3, 0.3, action_dim, device=env.device)
        action = base.unsqueeze(0).expand(env.num_envs, -1).contiguous()
    else:
        action = torch.zeros(env.num_envs, action_dim, device=env.device)

    print("\n[forced initial state]")
    _dump_step(env, step_idx=0, raw_action=action)

    # Main loop
    for t in range(1, args.steps + 1):
        env.step(action)
        _dump_step(env, step_idx=t, raw_action=action)

    if _log_fh is not None:
        import sys as _sys

        _sys.stdout = _orig_stdout
        _log_fh.close()
        print(f"Trace written to: {out_path}")


if __name__ == "__main__":
    main()
