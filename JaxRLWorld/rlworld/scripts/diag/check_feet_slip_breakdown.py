"""Decompose ``feet_slip`` divergence across simulators.

``feet_slip = -sum_feet(vel_xy² × is_contact) × command_active``. The reward
divergence between sims at training start is driven by some combination
of these inputs. This script breaks down each per (env, foot):

    foot_frame_pos_w        # dummy welded body world position
    foot_frame_vel_w        # dummy welded body world velocity
    ankle_roll_pos_w        # parent body world position (contact source)
    ankle_roll_vel_w        # parent body world velocity
    vel_xy_norm_sq          # foot_frame xy velocity squared norm
    is_contact              # contact_manager.is_contact result
    cost_per_foot           # vel_xy² × is_contact
    command_active          # command magnitude > threshold

Builds each sim with the same preset, seeds torch identically, resets,
and runs a few steps with a small random action so feet have some real
motion. Then dumps everything to a single .txt file plus a short stdout
summary, and computes per-sim aggregates + cross-sim deltas.

The .txt is written to ``./feet_slip_breakdown.txt`` (i.e. the current
working directory of the invoker).

Usage:
    python -m rlworld.scripts.diag.check_feet_slip_breakdown
    python -m rlworld.scripts.diag.check_feet_slip_breakdown --num-envs 16 --settle 10 --steps 40 --action-scale 0.4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# This diag builds multiple sim backends sequentially in one process — bypass
# the single-backend guard in BaseRunner.create_with_env.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import importlib

import numpy as np

_SIMS = ("genesis", "newton", "mujoco")

_FEET_FRAME_BODIES = ("left_foot_frame", "right_foot_frame")
_ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_COMMAND_THRESHOLD = 0.05  # matches g1_29dof feet_slip preset


def _build_env(sim: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    mod = importlib.import_module("rlworld.rl.configs.presets.g1_29dof.base")
    cfg_cls = mod.G1FlatConfig
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _np(t):
    return t.detach().cpu().numpy()


def _probe(env, settle: int, steps: int, action_scale: float, seed: int) -> dict[str, np.ndarray]:
    """Reset env, settle with zero action, then run random-action steps. Capture state."""
    import torch

    torch.manual_seed(seed)
    env.reset()
    n_envs, n_act = env.num_envs, env.num_actions
    zero = torch.zeros(n_envs, n_act, device=env.device)
    for _ in range(settle):
        env.step(zero)
    # Random small action so feet have actual motion.
    for _ in range(steps):
        action = (torch.rand(n_envs, n_act, device=env.device) * 2.0 - 1.0) * action_scale
        env.step(action)
    rd = env.get_robot_data()
    cm = env.command_manager
    contact_mgr = env.contact_manager

    # Resolve body indices for foot_frame and ankle_roll bodies.
    foot_ids = [rd.find_body_index(n) for n in _FEET_FRAME_BODIES]
    ankle_ids = [rd.find_body_index(n) for n in _ANKLE_BODIES]
    foot_ids_t = torch.tensor(foot_ids, device=env.device, dtype=torch.long)
    ankle_ids_t = torch.tensor(ankle_ids, device=env.device, dtype=torch.long)

    foot_pos = _np(rd.body_pos_w_by_ids(foot_ids_t))  # (W, 2, 3)
    foot_vel = _np(rd.body_lin_vel_w_by_ids(foot_ids_t))  # (W, 2, 3)
    ankle_pos = _np(rd.body_pos_w_by_ids(ankle_ids_t))
    ankle_vel = _np(rd.body_lin_vel_w_by_ids(ankle_ids_t))
    ankle_ang = _np(rd.body_ang_vel_w_all)[:, ankle_ids, :]  # (W, 2, 3)

    is_contact = _np(contact_mgr.is_contact("feet_ground_contact", order=list(_ANKLE_BODIES))).astype(bool)
    # (W, 2)

    cmd_x = _np(cm.lin_vel_x)
    cmd_y = _np(cm.lin_vel_y)
    cmd_w = _np(cm.ang_vel)
    cmd_norm = np.sqrt(cmd_x**2 + cmd_y**2) + np.abs(cmd_w)
    command_active = (cmd_norm > _COMMAND_THRESHOLD).astype(np.float32)

    vel_xy_norm_sq = foot_vel[..., 0] ** 2 + foot_vel[..., 1] ** 2  # (W, 2)
    cost_per_foot = vel_xy_norm_sq * is_contact.astype(np.float32)  # (W, 2)
    cost_per_env = cost_per_foot.sum(axis=1)  # (W,)
    reward_per_env = -cost_per_env * command_active  # (W,)

    return {
        "foot_pos": foot_pos,
        "foot_vel": foot_vel,
        "ankle_pos": ankle_pos,
        "ankle_vel": ankle_vel,
        "ankle_ang": ankle_ang,
        "is_contact": is_contact,
        "vel_xy_norm_sq": vel_xy_norm_sq,
        "cost_per_foot": cost_per_foot,
        "cost_per_env": cost_per_env,
        "command_active": command_active,
        "cmd_x": cmd_x,
        "cmd_y": cmd_y,
        "cmd_w": cmd_w,
        "reward_per_env": reward_per_env,
    }


def _fmt_arr(arr: np.ndarray, fmt: str = "+.4f") -> str:
    return "[" + ", ".join(f"{x:{fmt}}" for x in arr) + "]"


def _section(out: list[str], title: str) -> None:
    out.append("\n" + "=" * 72)
    out.append(title)
    out.append("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--settle", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--action-scale", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--output", type=str, default="feet_slip_breakdown.txt")
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    results: dict[str, dict[str, np.ndarray]] = {}
    out: list[str] = []
    out.append(
        f"feet_slip cross-sim breakdown — preset=g1_29dof num_envs={args.num_envs} "
        f"settle={args.settle} steps={args.steps} action_scale={args.action_scale} seed={args.seed}"
    )
    out.append("")
    out.append("Feet_slip formula: reward = -sum_feet(vel_xy² × is_contact) × command_active")
    out.append(f"command_threshold = {_COMMAND_THRESHOLD}  (sum of |cmd_xy| + |ang_vel| > threshold → active)")
    out.append("")
    out.append("Dummy body 'left_foot_frame' / 'right_foot_frame' is welded to ankle_roll_link")
    out.append("at +0.04m fore, -0.037m sole. Contacts come from ankle_roll (frame body has no geom),")
    out.append("so 'is_contact' is sampled at ankle_roll while 'foot_vel' is at the welded frame body.")

    for sim in sims:
        import torch  # local; torch imported inside _probe too

        print(f"Building [{sim}] ...", flush=True)
        env = _build_env(sim, args.num_envs)
        d = _probe(env, args.settle, args.steps, args.action_scale, args.seed)
        results[sim] = d
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ───────────────────────────────────────────────────────────
    # Per-env dump (every env, every foot).
    # ───────────────────────────────────────────────────────────
    _section(out, "PER-ENV DUMP")
    for env_i in range(args.num_envs):
        out.append("")
        out.append(f"─── env {env_i} ───")
        for sim in sims:
            r = results[sim]
            for foot_i, fname in enumerate(_FEET_FRAME_BODIES):
                fp = r["foot_pos"][env_i, foot_i]
                fv = r["foot_vel"][env_i, foot_i]
                ap_ = r["ankle_pos"][env_i, foot_i]
                av = r["ankle_vel"][env_i, foot_i]
                aw = r["ankle_ang"][env_i, foot_i]
                ic = bool(r["is_contact"][env_i, foot_i])
                vxy2 = float(r["vel_xy_norm_sq"][env_i, foot_i])
                cpf = float(r["cost_per_foot"][env_i, foot_i])
                out.append(
                    f"  {sim:<8s} {fname:<18s}"
                    f"  foot_pos={_fmt_arr(fp)}  foot_vel={_fmt_arr(fv)}\n"
                    f"  {'':<8s} {'(ankle)':<18s}"
                    f"  ankle_pos={_fmt_arr(ap_)}  ankle_vel={_fmt_arr(av)}  ankle_ang={_fmt_arr(aw)}\n"
                    f"  {'':<8s} {'':<18s}"
                    f"  is_contact={ic}  vel_xy²={vxy2:.6f}  cost={cpf:.6f}"
                )
            cpe = float(r["cost_per_env"][env_i])
            ca = float(r["command_active"][env_i])
            rew = float(r["reward_per_env"][env_i])
            out.append(
                f"  {sim:<8s} {'(env total)':<18s}"
                f"  cmd=({r['cmd_x'][env_i]:+.3f}, {r['cmd_y'][env_i]:+.3f}, {r['cmd_w'][env_i]:+.3f})"
                f"  cost_env={cpe:.6f}  cmd_active={ca:.0f}  REWARD={rew:+.6f}"
            )

    # ───────────────────────────────────────────────────────────
    # Per-sim aggregates.
    # ───────────────────────────────────────────────────────────
    _section(out, "PER-SIM AGGREGATES")
    out.append("")
    header = (
        f"  {'sim':<10s}  {'fc_left':<8s}  {'fc_right':<8s}  {'fc_both':<8s}  "
        f"{'<|fv_xy|>':<12s}  {'<vel²>':<12s}  {'<cost>':<12s}  {'<reward>':<14s}"
    )
    out.append(header)
    out.append("  " + "─" * (len(header) - 2))
    for sim in sims:
        r = results[sim]
        ic = r["is_contact"]
        fc_left = ic[:, 0].mean()
        fc_right = ic[:, 1].mean()
        fc_both = (ic[:, 0] & ic[:, 1]).mean()
        fv = r["foot_vel"]
        fv_xy_norm = np.sqrt(fv[..., 0] ** 2 + fv[..., 1] ** 2)  # (W, 2)
        mean_fv = fv_xy_norm.mean()
        mean_vel2 = r["vel_xy_norm_sq"].mean()
        mean_cost = r["cost_per_env"].mean()
        mean_reward = r["reward_per_env"].mean()
        out.append(
            f"  {sim:<10s}  {fc_left:<8.3f}  {fc_right:<8.3f}  {fc_both:<8.3f}  "
            f"{mean_fv:<12.6f}  {mean_vel2:<12.6f}  {mean_cost:<12.6f}  {mean_reward:<+14.6f}"
        )

    # ───────────────────────────────────────────────────────────
    # Cross-sim deltas (vs first sim as reference).
    # ───────────────────────────────────────────────────────────
    if len(sims) >= 2:
        _section(out, "CROSS-SIM DELTAS (vs first sim)")
        ref = sims[0]
        rref = results[ref]
        out.append(f"\nReference: {ref}")
        out.append("")
        out.append(
            f"  {'sim':<10s}  {'max|Δfoot_pos|':<15s}  {'max|Δfoot_vel|':<15s}  "
            f"{'Δfc_both':<10s}  {'reward_ratio':<14s}"
        )
        out.append("  " + "─" * 80)
        for sim in sims:
            r = results[sim]
            d_fp = float(np.max(np.abs(r["foot_pos"] - rref["foot_pos"])))
            d_fv = float(np.max(np.abs(r["foot_vel"] - rref["foot_vel"])))
            d_fcb = float(
                (r["is_contact"][:, 0] & r["is_contact"][:, 1]).mean()
                - (rref["is_contact"][:, 0] & rref["is_contact"][:, 1]).mean()
            )
            rew = r["reward_per_env"].mean()
            rew_ref = rref["reward_per_env"].mean()
            ratio = rew / rew_ref if abs(rew_ref) > 1e-9 else float("nan")
            out.append(f"  {sim:<10s}  {d_fp:<15.6f}  {d_fv:<15.6f}  " f"{d_fcb:<+10.3f}  {ratio:<+14.3f}")

    # ───────────────────────────────────────────────────────────
    # Welded-body kinematic consistency check.
    # ───────────────────────────────────────────────────────────
    # The foot_frame is welded to ankle_roll at local offset (0.04, 0, -0.037).
    # In world frame: foot_frame_pos = ankle_pos + R(ankle_quat) @ (0.04, 0, -0.037)
    # foot_frame_vel = ankle_vel + ankle_ang × (foot_frame_pos - ankle_pos)
    # If the sim's body_lin_vel_w computation respects weld, residuals should be ~0.
    _section(out, "WELDED-BODY KINEMATIC CONSISTENCY (foot_vel vs ankle_vel + ω × Δpos)")
    out.append("")
    out.append("Expected (if weld kinematics is exact): foot_vel = ankle_vel + ankle_ang × (foot_pos − ankle_pos)")
    out.append("")
    out.append(f"  {'sim':<10s}  {'mean|residual|':<16s}  {'max|residual|':<16s}")
    out.append("  " + "─" * 50)
    for sim in sims:
        r = results[sim]
        delta_pos = r["foot_pos"] - r["ankle_pos"]  # (W, 2, 3)
        pred_vel = r["ankle_vel"] + np.cross(r["ankle_ang"], delta_pos, axis=-1)  # (W, 2, 3)
        residual = r["foot_vel"] - pred_vel  # (W, 2, 3)
        residual_norm = np.linalg.norm(residual, axis=-1)  # (W, 2)
        out.append(f"  {sim:<10s}  {residual_norm.mean():<16.6e}  {residual_norm.max():<16.6e}")

    # ───────────────────────────────────────────────────────────
    # Contact pattern breakdown.
    # ───────────────────────────────────────────────────────────
    _section(out, "CONTACT PATTERN BREAKDOWN")
    out.append("")
    out.append(f"  {'sim':<10s}  {'both':<7s}  {'left only':<10s}  {'right only':<11s}  {'neither':<8s}")
    out.append("  " + "─" * 60)
    for sim in sims:
        ic = results[sim]["is_contact"]
        both = (ic[:, 0] & ic[:, 1]).sum()
        lonly = (ic[:, 0] & ~ic[:, 1]).sum()
        ronly = (~ic[:, 0] & ic[:, 1]).sum()
        neither = (~ic[:, 0] & ~ic[:, 1]).sum()
        out.append(f"  {sim:<10s}  {both:<7d}  {lonly:<10d}  {ronly:<11d}  {neither:<8d}")

    # ───────────────────────────────────────────────────────────
    # Conclusion hint.
    # ───────────────────────────────────────────────────────────
    _section(out, "WHAT TO LOOK FOR")
    out.append("""
The reward is product of (vel²) × (is_contact) × (command_active), so a 2x
divergence on mujoco can come from any one of:

  (A) FOOT VELOCITY — mjlab's foot_vel larger or smaller than Newton/Genesis
      for the welded dummy body. Look at the 'mean|residual|' in WELDED-BODY
      KINEMATIC CONSISTENCY: if mjlab's residual ≈ 0 (clean transfer) but
      Newton/Genesis have nonzero residuals, the latter under/over-estimate
      foot velocity → different cost.

  (B) CONTACT FRACTION — mjlab detects 'in contact' more (or less) than
      Newton/Genesis. Look at CONTACT PATTERN BREAKDOWN: if mjlab's 'both'
      count is ~2x Newton/Genesis, the cost summed across feet × envs is
      ~2x larger on mjlab even with identical vel² — directly explains
      the reported "mujoco half" (-2x = half reward).

  (C) FOOT VS ANKLE — if foot_pos / foot_vel agree across sims but the
      WELDED-BODY CONSISTENCY residual is ~0 for all sims, then the
      discrepancy is purely in the contact detection layer ((B) dominates).

  (D) COMMAND ACTIVE — should match across sims after the encoder_bias fix.
      If <reward> column scales with command_active count differences, the
      RNG-state parity check failed somewhere (unlikely).
""")

    # Write to file + brief stdout summary.
    out_path = Path(args.output).resolve()
    out_path.write_text("\n".join(out), encoding="utf-8")
    print()
    print(f"Wrote breakdown to: {out_path}")
    print(f"Lines: {len(out)}, KB: {out_path.stat().st_size / 1024:.1f}")
    # Brief stdout (last sections only):
    print("\n=== AGGREGATE SUMMARY (see file for full per-env dump) ===")
    for line in out:
        if line.startswith("  ") and len(line) < 200:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
