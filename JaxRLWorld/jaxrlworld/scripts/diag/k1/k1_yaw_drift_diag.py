"""K1 lateral-command yaw-drift diagnostic (sim).

Question: when told to walk PURELY sideways, does the policy spin (yaw) on its
own in sim? We pin a fixed lateral command (vy=+/-, vx=0, wz=0) — so the yaw-rate
command is exactly 0, i.e. "do NOT rotate" — and measure the ACHIEVED yaw rate.
A nonzero achieved yaw rate under wz=0 means the policy rotates by itself;
integrated over a 20 s episode it becomes a large heading rotation (the "keeps
turning while going sideways" the user sees). Reports LEFT vs RIGHT so mirror
asymmetry shows up.

Command is fixed by writing the velocity term's buffer every step (the same
mechanism the viser command panel uses — it does NOT break the gait), so this is
a clean held-command probe, not a random-command average.

Verdict:
  - yaw rate clearly != 0 in sim  -> the spin is a policy/training issue
    (track_ang_vel too weak to hold wz=0, and/or left/right asymmetry).
  - ~0 in sim but seen on real    -> sim2real (COM / latency), not training.

Run (JAX -> jaxpy):
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_yaw_drift_diag \\
        --wandb-run-path jsw7460/K1_Joystick/x7sjv3jp --sim newton \\
        --vy 0.5 --num-envs 128 --steps 1000
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    import numpy as np
    import torch

    from jaxrlworld.rl.evals import PolicyEvaluator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint", required=True, help="local checkpoint dir OR wandb run path (entity/project/run_id)"
    )
    ap.add_argument("--sim", default="newton", choices=("newton", "mujoco", "genesis"))
    ap.add_argument("--vy", type=float, default=0.5, help="lateral speed magnitude (LEFT=+vy, RIGHT=-vy)")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--heading",
        choices=("on", "off"),
        default="on",
        help="on: measure WITH heading control (heading-trained policy, holds heading); "
        "off: rate-only baseline (wz=0 command, exposes raw drift)",
    )
    ap.add_argument(
        "--target-rate",
        type=float,
        default=0.0,
        help="rotate heading_target at this rate (rad/s); >0 tests TURNING: does the robot "
        "follow (achieved yaw rate ~ target-rate)? 0 = hold heading (drift test)",
    )
    args = ap.parse_args()

    # local dir -> policy_path; else treat as a wandb run path
    loader = {"policy_path": args.checkpoint} if os.path.isdir(args.checkpoint) else {"wandb_run_path": args.checkpoint}
    print(f"[yaw-drift] loading {args.checkpoint} on {args.sim} (num_envs={args.num_envs})")
    ev = PolicyEvaluator(
        **loader,
        eval_target=args.sim,
        num_evals=1,
        seed=args.seed,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    env, policy = ev.env, ev.policy
    n = env.num_envs
    half = n // 2
    device = env.device
    left_ids = torch.arange(0, half, device=device)
    right_ids = torch.arange(half, n, device=device)

    # Velocity command term buffer: [vx, vy, wz], body frame. Held fixed each step.
    vt = env.command_manager._terms["velocity"]
    vt.cfg.rel_standing_envs = 0.0  # never resample an env to standing (would zero our held command)
    # Force heading control on/off at eval so we can measure a true rate-only
    # baseline (off) vs the heading-trained behavior (on), independent of what the
    # rebuilt preset defaults to.
    heading_on = args.heading == "on"
    vt.cfg.heading_command = heading_on

    control_dt = float(env.control_dt)
    turn = torch.zeros(n, device=device)  # accumulated heading target (rad) for --target-rate

    def _set_cmd() -> None:
        vt._command[:, 0] = 0.0  # vx = 0
        vt._command[left_ids, 1] = args.vy
        vt._command[right_ids, 1] = -args.vy
        vt._command[:, 2] = 0.0
        if heading_on:
            if args.target_rate != 0.0:
                turn.add_(args.target_rate * control_dt)  # rotate the heading target over time
                vt.heading_target[:] = turn
            else:
                vt.heading_target[:] = 0.0  # hold heading (drift test)
            vt.is_heading_env[:] = True

    rs = env.get_robot_state()

    WZ = []  # achieved body-frame yaw rate per step
    with torch.no_grad():
        for t in range(args.steps):
            _set_cmd()
            obs = env.obs_manager.get_observation()
            action = policy.get_action(obs, rs)
            obs, _rew, term, trunc, _extras = env.step(action)
            rs = env.get_robot_state()
            done = term | trunc
            if done.any():
                policy.notify_reset(done.cpu().numpy())
            rd = env.get_robot_data()
            WZ.append(rd.root_link_ang_vel_b[:, 2].cpu().numpy())  # (N,)
            if t % 250 == 0:
                print(f"  step {t}/{args.steps}")

    wz = np.stack(WZ)  # (steps, N)
    left_wz = float(wz[:, :half].mean())
    right_wz = float(wz[:, half:].mean())
    left_abs = float(np.abs(wz[:, :half]).mean())
    right_abs = float(np.abs(wz[:, half:]).mean())

    def deg(r: float) -> float:
        return r * 180.0 / np.pi

    print(f"\n=== lateral-command yaw drift ({args.sim}, vy=±{args.vy}, wz cmd = 0) ===")
    print(f"  control_dt={control_dt * 1000:.1f} ms, rollout={args.steps * control_dt:.1f} s")
    print(
        f"  LEFT  (vy=+{args.vy}): mean yaw rate = {left_wz:+.4f} rad/s = {deg(left_wz):+.2f} deg/s"
        f"   -> {deg(left_wz) * 20:+.0f} deg over a 20 s episode   (|rate| avg {deg(left_abs):.2f} deg/s)"
    )
    print(
        f"  RIGHT (vy=-{args.vy}): mean yaw rate = {right_wz:+.4f} rad/s = {deg(right_wz):+.2f} deg/s"
        f"   -> {deg(right_wz) * 20:+.0f} deg over a 20 s episode   (|rate| avg {deg(right_abs):.2f} deg/s)"
    )
    print(f"  left/right asymmetry = {abs(left_wz - right_wz):.4f} rad/s  (mirror should shrink this)")

    if args.target_rate != 0.0:
        all_wz = float(wz.mean())
        ratio = all_wz / args.target_rate
        print(
            f"\n[Turning test]  heading_target rotated at {args.target_rate:+.3f} rad/s = {deg(args.target_rate):+.1f} deg/s"
        )
        print(f"  achieved yaw rate (all) = {all_wz:+.4f} rad/s = {deg(all_wz):+.1f} deg/s")
        print(f"  follow ratio            = {ratio:.2f}   (1.0 = perfect turn tracking)")
        if abs(ratio) > 0.5:
            print("  -> robot FOLLOWS the target: TURNING IS LEARNED in sim.")
            print("     If it does not turn on deploy/sim2sim: a DEPLOY issue (joystick->desired_heading,")
            print("     use_heading_command flag, or reset not clearing _desired_heading).")
        else:
            print("  -> robot does NOT follow: TURNING WAS NOT LEARNED in sim (more training / heading setup).")
        return 0

    print("\n[Verdict]")
    thr = 0.02  # ~1.1 deg/s ; over 20 s that is >20 deg of heading rotation
    for wzv, name in ((left_wz, "LEFT"), (right_wz, "RIGHT")):
        if abs(wzv) > thr:
            print(f"  {name}: yaw drift PRESENT in sim ({deg(wzv):+.1f} deg/s under wz=0) -> visible spin.")
        else:
            print(f"  {name}: yaw rate ~0 in sim (holds heading).")
    print("  -> present in sim  => policy/training (track_ang_vel too weak / left-right asymmetry).")
    print("     clean in sim, only on real => sim2real (COM / latency).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
