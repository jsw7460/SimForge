"""K1 standing-posture diag: does the trained policy lean forward IN SIM?

The real robot stands with a persistent +1.5~3.5 deg forward pitch under
zero command, and the measured static ankle torque matches the XML CoM
*at that leaned pose* — so the model CoM is not the suspect. This diag
answers the remaining branch question: is the lean the policy's own
equilibrium (then sim shows it too), or a hardware-side bias (encoder
zero / IMU mounting — then sim stands level)?

Rolls out a checkpoint in its training sim with the velocity command
locked to zero (``set_command`` external-control lock, re-applied every
step so episode resets cannot unlock it). Eval defaults already disable
obs noise, pushes and reset-DR, so the world is the nominal plant.

Reports, over the post-settle window (mean ± std across envs x time):
  - root roll/pitch (deg) and projected gravity — the policy's own view
  - leg sagittal applied torques (hip_pitch / knee / ankle_pitch, L/R)
    and the L+R ankle-pitch sum (the CoP statics number)
  - base height, mean |joint_vel| (how static the stance really is)
  - fall count (terminations) — falls make the posture stats meaningless

Real-robot reference (highfoot dataset, first 1 s of each file):
pitch +2.3 deg mean (+0.4 .. +3.5 across files), roll |<=1.3| deg,
ankle_pitch torque L+R ~ +7.5 N*m, gravity ~ (+0.04, ~0, -0.999).

Run (GPU box)::

    jaxpy -m jaxrlworld.scripts.diag.k1.k1_standing_pitch_diag \
        --wandb_run_path jsw7460/K1_Joystick/s1noc0b3
"""

from __future__ import annotations

import argparse

import torch

from jaxrlworld.rl.evals import PolicyEvaluator

# Sagittal leg joints (canonical act-manager names).
_REPORT_JOINTS = (
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
)


def _roll_pitch_deg(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll/pitch in degrees from a (N, 4) wxyz quaternion batch."""
    w, x, y, z = quat_wxyz.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return torch.rad2deg(roll), torch.rad2deg(pitch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb_run_path",
        default="jsw7460/K1_Joystick/s1noc0b3",
        help="wandb run whose latest checkpoint to evaluate",
    )
    parser.add_argument("--checkpoint", default=None, help="local checkpoint dir (overrides --wandb_run_path)")
    parser.add_argument("--sim", default=None, help="cross-sim eval target (default: training sim)")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500, help="rollout length (control steps)")
    parser.add_argument("--settle_steps", type=int, default=100, help="steps discarded before collecting stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluator = PolicyEvaluator(
        policy_path=args.checkpoint,
        wandb_run_path=None if args.checkpoint else args.wandb_run_path,
        eval_target=args.sim,
        seed=args.seed,
        use_logging=False,
        record_video=False,
        save_data=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    env = evaluator.env
    policy = evaluator.policy
    dev = env.device

    rd = env.get_robot_data("robot")
    vel_term = env.command_manager.get_term("velocity")
    all_ids = torch.arange(env.num_envs, device=dev)
    zero_cmd = torch.zeros(env.num_envs, 3, device=dev)

    joint_names = list(env.act_manager.actuated_joint_names)
    jidx = {name: joint_names.index(name) for name in _REPORT_JOINTS}

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()
    vel_term.set_command(all_ids, zero_cmd)

    n_collect = args.steps - args.settle_steps
    if n_collect <= 0:
        raise ValueError("--steps must exceed --settle_steps")
    roll_buf = torch.zeros(n_collect, env.num_envs, device=dev)
    pitch_buf = torch.zeros_like(roll_buf)
    grav_buf = torch.zeros(n_collect, env.num_envs, 3, device=dev)
    tau_buf = torch.zeros(n_collect, env.num_envs, len(_REPORT_JOINTS), device=dev)
    height_buf = torch.zeros_like(roll_buf)
    speed_buf = torch.zeros_like(roll_buf)
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    tau_cols = torch.tensor([jidx[n] for n in _REPORT_JOINTS], device=dev)

    with torch.no_grad():
        for step in range(args.steps):
            # Re-lock every step: an episode reset clears external control.
            vel_term.set_command(all_ids, zero_cmd)
            action = policy.get_action(obs, robot_states)
            obs, _, terminated, truncated, _ = env.step(action)
            robot_states = env.get_robot_state()
            done = terminated | truncated
            if done.any():
                fallen |= terminated
                policy.notify_reset(done.cpu().numpy())

            if step >= args.settle_steps:
                i = step - args.settle_steps
                roll, pitch = _roll_pitch_deg(rd.root_link_quat_w)
                roll_buf[i] = roll
                pitch_buf[i] = pitch
                grav_buf[i] = rd.projected_gravity_b
                tau_buf[i] = rd.applied_torque[:, tau_cols]
                height_buf[i] = rd.root_link_pos_w[:, 2]
                speed_buf[i] = rd.joint_vel.abs().mean(dim=1)

    cmd_norm = float(vel_term.command.norm(dim=1).max())
    n_fallen = int(fallen.sum())

    def _stat(buf: torch.Tensor) -> tuple[float, float]:
        return float(buf.mean()), float(buf.std())

    print("\n=== K1 standing-posture diag (zero command) ===")
    print(f"checkpoint      : {evaluator.policy_path}")
    print(
        f"sim             : {evaluator.sim_type}   envs={env.num_envs}  "
        f"steps={args.steps} (settle {args.settle_steps})  seed={args.seed}"
    )
    print(f"cmd norm (max)  : {cmd_norm:.2e}  (must be ~0)")
    print(f"fallen envs     : {n_fallen}/{env.num_envs}" + ("  !! posture stats contaminated" if n_fallen else ""))

    rm, rs = _stat(roll_buf)
    pm, ps = _stat(pitch_buf)
    g = grav_buf.reshape(-1, 3).mean(dim=0)
    hm, hs = _stat(height_buf)
    sm, _ = _stat(speed_buf)
    print("\n-- posture (post-settle window) --")
    print(f"roll  : {rm:+6.2f} ± {rs:.2f} deg      (real: |<=1.3|)")
    print(f"pitch : {pm:+6.2f} ± {ps:.2f} deg      (real: +2.3 mean, +0.4..+3.5)")
    print(
        f"gravity_b       : ({float(g[0]):+.3f}, {float(g[1]):+.3f}, {float(g[2]):+.3f})"
        "   (real: (+0.04, ~0, -0.999))"
    )
    print(f"base height     : {hm:.3f} ± {hs:.3f} m")
    print(f"mean |qdot|     : {sm:.3f} rad/s")

    print("\n-- sagittal leg applied torque (N*m, mean ± std) --")
    tau_flat = tau_buf.reshape(-1, len(_REPORT_JOINTS))
    for k, name in enumerate(_REPORT_JOINTS):
        tm = float(tau_flat[:, k].mean())
        ts = float(tau_flat[:, k].std())
        print(f"{name:18s} {tm:+7.2f} ± {ts:5.2f}")
    ankle_sum = float(
        tau_flat[:, _REPORT_JOINTS.index("Left_Ankle_Pitch")].mean()
        + tau_flat[:, _REPORT_JOINTS.index("Right_Ankle_Pitch")].mean()
    )
    print(f"{'ankle_pitch L+R':18s} {ankle_sum:+7.2f}        (real: +7.5;" " upright-CoM statics: ~+4.0)")

    print("\n-- verdict guide --")
    print(
        "sim pitch ~ +2deg  → the lean is the policy's equilibrium"
        " (reward-side fix: flat_orientation std tightening)"
    )
    print("sim pitch ~ 0deg   → real-only bias: ankle/hip encoder zero" " or IMU mounting pitch — CoM is not the cause")

    evaluator.cleanup()


if __name__ == "__main__":
    main()
