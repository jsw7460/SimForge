"""Load trained policy from wandb and dump per-joint torque across the
settle boundary for T1 getup sim2sim audit.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_policy_trace.py \
        --sim newton --scenario standing \
        --wandb jsw7460/T1_Getup/ihdt0ykj

Two scenarios, identical initial state enforced via the robot state
writer so all three sims start from byte-identical conditions:

    --scenario standing  : root at (0, 0, 0.665), quat identity,
                           joint_pos = default_joint_angles, zero vel
    --scenario fallen    : root at (0, 0, 0.80),
                           quat pitched 60 deg forward,
                           joint_pos = default_joint_angles,
                           zero vel (drop onto ground)

Per-step dump covers steps ``[settle_steps - 5, settle_steps + 20]`` so
we see the last few settle-masked steps (policy output ignored,
target held to current pose) and the first 20 policy-driven steps.
"""
from __future__ import annotations

import argparse
import math

import torch

from rlworld.rl.evals import PolicyEvaluator


# ── Deterministic init poses ────────────────────────────────────────

# Standing: canonical T1 standing pose
STANDING = {
    "pos":      (0.0, 0.0, 0.665),
    "quat":     (1.0, 0.0, 0.0, 0.0),
    "lin_vel":  (0.0, 0.0, 0.0),
    "ang_vel":  (0.0, 0.0, 0.0),
}

# Fallen: dropped from 0.8 m, 60 deg forward pitch
_PITCH = math.radians(60.0)
FALLEN = {
    "pos":      (0.0, 0.0, 0.80),
    "quat":     (math.cos(_PITCH / 2), 0.0, math.sin(_PITCH / 2), 0.0),
    "lin_vel":  (0.0, 0.0, 0.0),
    "ang_vel":  (0.0, 0.0, 0.0),
}


def _fmt_row(values, prec=5):
    return " ".join(f"{float(v):+.{prec}f}" for v in values)


def _force_state(env, pose_dict, joint_pos_vec):
    """Overwrite root + joint state, then FK."""
    writer = env.get_robot_state_writer("robot")
    env_ids = torch.arange(env.num_envs, device=env.device)
    N = env.num_envs
    p  = torch.tensor(pose_dict["pos"],     dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    q  = torch.tensor(pose_dict["quat"],    dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    lv = torch.tensor(pose_dict["lin_vel"], dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    av = torch.tensor(pose_dict["ang_vel"], dtype=torch.float32, device=env.device).expand(N, -1).contiguous()
    jp = joint_pos_vec.to(env.device).unsqueeze(0).expand(N, -1).contiguous()
    jv = torch.zeros_like(jp)

    writer.set_root_pose(p, q, env_ids=env_ids)
    writer.set_root_velocity(lv, av, env_ids=env_ids)
    writer.set_dof_positions(jp, env_ids=env_ids)
    writer.set_dof_velocities(jv, env_ids=env_ids)
    if hasattr(writer, "eval_fk"):
        writer.eval_fk(env_ids=env_ids)


def _build_default_joint_pos(env, robot_cfg):
    """Regex-resolve cfg.robot.default_joint_angles against canonical joint order."""
    from rlworld.rl.utils import string as string_utils
    all_names = list(env.act_manager.actuated_joint_names)
    matched_idx, _, matched_vals = string_utils.resolve_matching_names_values(
        robot_cfg.default_joint_angles, all_names
    )
    jp = torch.zeros(len(all_names), dtype=torch.float32)
    for i, v in zip(matched_idx, matched_vals):
        jp[i] = float(v)
    return jp


def _dump_step(env, step_idx, raw_action, settle_steps):
    am = env.act_manager
    rd = env.robot_data
    def g(x):
        if isinstance(x, torch.Tensor):
            return x[0].detach().cpu().tolist() if x.dim() > 1 else x.detach().cpu().tolist()
        return list(x)

    ep_len = int(env.episode_length_buf[0].item()) if hasattr(env, "episode_length_buf") else -1
    in_settle = "SETTLE" if ep_len < settle_steps else "POLICY"

    print(f"\n── step {step_idx}  (episode_length_buf={ep_len}, mode={in_settle}) ──")
    print(f"raw_action      = {_fmt_row(g(raw_action))}")

    if hasattr(am, "_processed_action_history") and am._processed_action_history:
        pa = am._processed_action_history[0]
        print(f"processed_action= {_fmt_row(g(pa))}")

    tq = getattr(am, "applied_torque", None)
    if tq is None:
        tq = getattr(am, "_applied_torque", None)
    if tq is not None:
        print(f"applied_torque  = {_fmt_row(g(tq))}")

    print(f"joint_pos       = {_fmt_row(g(rd.joint_pos))}")
    print(f"joint_vel       = {_fmt_row(g(rd.joint_vel))}")
    print(f"root_pos_w      = {_fmt_row(g(rd.root_link_pos_w))}")
    print(f"root_quat_wxyz  = {_fmt_row(g(rd.root_link_quat_w))}")
    if hasattr(rd, "projected_gravity_b"):
        print(f"proj_gravity_b  = {_fmt_row(g(rd.projected_gravity_b))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("newton", "genesis", "mujoco"), required=True)
    parser.add_argument("--scenario", choices=("standing", "fallen"), required=True)
    parser.add_argument("--wandb", type=str, required=True, help="wandb run path, e.g. user/project/run_id")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--pre-settle", type=int, default=5, help="Steps before settle end to start dumping")
    parser.add_argument("--post-settle", type=int, default=20, help="Steps after settle end to dump")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or f"t1_policy_trace_{args.sim}_{args.scenario}.txt"
    import sys as _sys
    _log_fh = open(out_path, "w")
    _sys.stdout = _log_fh
    print(f"# t1_policy_trace sim={args.sim} scenario={args.scenario} "
          f"wandb={args.wandb} settle={args.settle_steps}")

    torch.manual_seed(args.seed)

    # Load policy + env via PolicyEvaluator (handles wandb download + cross-sim resolve)
    evaluator = PolicyEvaluator(
        wandb_run_path=args.wandb,
        eval_target=args.sim,
        num_evals=1,
        seed=args.seed,
        record_video=False,
        use_logging=False,
        save_data=False,
        use_rich_display=False,
    )
    env = evaluator.env
    policy = evaluator.policy

    # Force num_envs = 1 usage
    assert env.num_envs == 1 or env.num_envs > 0  # we'll only read env 0

    # Build default joint pos vector
    robot_cfg = evaluator.eval_cfgs.scene.robot_cfg \
        if hasattr(evaluator.eval_cfgs.scene, "robot_cfg") \
        else evaluator.eval_cfgs.scene.entities["robot"]
    # Some configs put robot on eval_cfgs.scene.robot_cfg; others on entities["robot"].init_state.joint_pos etc.
    # Fall back to env.robot_data.default_joint_pos
    default_jp = env.robot_data.default_joint_pos.clone().detach().cpu()
    if default_jp.dim() > 1:
        default_jp = default_jp[0]

    # Reset and force deterministic state
    obs, _ = env.reset()
    pose = STANDING if args.scenario == "standing" else FALLEN
    _force_state(env, pose, default_jp)

    # Zero out action history and episode counter so settle_steps logic
    # starts from step 0
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf.zero_()
    if hasattr(env.act_manager, "_raw_action_history"):
        for h in env.act_manager._raw_action_history:
            h.zero_()
    if hasattr(env.act_manager, "_processed_action_history"):
        for h in env.act_manager._processed_action_history:
            h.zero_()

    # Canonical joint order header
    print("\n[canonical joint order]")
    for i, name in enumerate(env.act_manager.actuated_joint_names):
        print(f"  {i:<3} {name}")

    # Recompute observation after state forcing (reset inner machinery that
    # cached initial obs).  Rather than calling a private API, we step
    # once with zero action and treat that as step 0 (it fires settle
    # logic that holds current pos → no policy action).
    # For simplicity we just step with policy and dump around settle.

    # Dump window
    dump_lo = max(0, args.settle_steps - args.pre_settle)
    dump_hi = args.settle_steps + args.post_settle

    # Initial obs / robot_states after state forcing — same pattern as
    # evaluator.evaluate(): compute obs manually, then loop.
    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state() if hasattr(env, "get_robot_state") else None

    # Step loop
    for t in range(dump_hi + 1):
        raw_action = policy.get_action(obs, robot_states)
        obs, _, _, _, _ = env.step(raw_action)
        if hasattr(env, "get_robot_state"):
            robot_states = env.get_robot_state()

        if dump_lo <= t <= dump_hi:
            _dump_step(env, t, raw_action, args.settle_steps)

    _log_fh.close()
    _sys.stdout = _sys.__stdout__
    print(f"Trace written to: {out_path}")


if __name__ == "__main__":
    main()
