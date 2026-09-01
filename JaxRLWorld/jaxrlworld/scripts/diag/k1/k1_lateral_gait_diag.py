"""Does the lateral-walking "glide -> rapid stepping" switch happen IN SIM?

The real K1 shows a two-regime lateral gait (seen in the d6 logs): after
a vy command it first slides laterally WITHOUT stepping (knee swing
amplitude ~0) for 0.6-1.2 s, then switches abruptly into sustained
small-amplitude high-cadence stepping. The plant-side fixes tried so
far (calibrated armature/lag, kp boost, action-delay DR 30-60 ms) did
not change it on hardware.

This diag answers the branch question: roll the SAME policy in sim,
stand 2 s at zero command, then step the command to (0, +/-vy, 0) and
profile the gait onset exactly like the real-data analysis —

  * sim ALSO shows glide -> abrupt-switch -> small/fast stepping
      -> the behavior is the POLICY's learned lateral gait (reward
         shaping / gait-basin issue); no plant parameter will fix it.
  * sim steps cleanly from the start
      -> the switch is real-only; remaining suspects are the
         floor/sole contact (friction, compliance) and per-session
         encoder bias — not actuator dynamics.

Half the envs get +vy, half -vy (left/right in one run). Reported per
0.5 s window after command onset (mean over surviving envs):
knee-pitch swing amplitude [rad], hip-roll amplitude, step cadence
[Hz] from knee-velocity zero crossings — the same estimators used on
the real d6 files, so the two tables compare 1:1.

Real d6 reference (left): amp ~0 until t0+0.6 s, jump to ~0.1 rad by
t0+1.2 s, steady 0.09-0.11 rad at 2-3 Hz. (right): false start at
+0.2 s, pause, restart at +1.1 s, steady 0.065-0.08 rad.

Run (GPU box)::

    jaxpy -m jaxrlworld.scripts.diag.k1.k1_lateral_gait_diag \
        --wandb_run_path jsw7460/K1_Joystick/x7sjv3jp
"""

from __future__ import annotations

import argparse

import torch

from jaxrlworld.rl.evals import PolicyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb_run_path", default="jsw7460/K1_Joystick/x7sjv3jp")
    parser.add_argument("--checkpoint", default=None, help="local checkpoint dir (overrides wandb)")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--vy", type=float, default=0.4)
    parser.add_argument("--stand_steps", type=int, default=100, help="zero-command settle (2 s)")
    parser.add_argument("--walk_steps", type=int, default=200, help="profiled steps after onset (4 s)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluator = PolicyEvaluator(
        policy_path=args.checkpoint,
        wandb_run_path=None if args.checkpoint else args.wandb_run_path,
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
    joint_names = list(env.act_manager.actuated_joint_names)
    kneeL = joint_names.index("Left_Knee_Pitch")
    hiprL = joint_names.index("Left_Hip_Roll")

    # +vy for the first half, -vy for the second (left/right in one run).
    cmd = torch.zeros(env.num_envs, 3, device=dev)
    half = env.num_envs // 2
    walk_cmd = cmd.clone()
    walk_cmd[:half, 1] = args.vy
    walk_cmd[half:, 1] = -args.vy

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    knee_pos = torch.zeros(args.walk_steps, env.num_envs, device=dev)
    knee_vel = torch.zeros_like(knee_pos)
    hipr_pos = torch.zeros_like(knee_pos)
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)

    with torch.no_grad():
        for phase, steps, c in (("stand", args.stand_steps, cmd), ("walk", args.walk_steps, walk_cmd)):
            for step in range(steps):
                vel_term.set_command(all_ids, c)
                action = policy.get_action(obs, robot_states)
                obs, _, terminated, truncated, _ = env.step(action)
                robot_states = env.get_robot_state()
                done = terminated | truncated
                if done.any():
                    fallen |= terminated
                    policy.notify_reset(done.cpu().numpy())
                if phase == "walk":
                    knee_pos[step] = rd.joint_pos[:, kneeL]
                    knee_vel[step] = rd.joint_vel[:, kneeL]
                    hipr_pos[step] = rd.joint_pos[:, hiprL]

    ok = ~fallen
    n_fallen = int(fallen.sum())
    print("\n=== K1 lateral gait-onset diag (sim) ===")
    print(f"checkpoint : {evaluator.policy_path}")
    print(
        f"sim        : {evaluator.sim_type}  envs={env.num_envs} (+vy {half} / -vy {env.num_envs - half})"
        f"  vy={args.vy}  seed={args.seed}"
    )
    print(f"fallen     : {n_fallen}/{env.num_envs} (excluded from stats)")

    W, S = 25, 5  # 0.5 s window, 0.1 s stride — same as the real analysis
    print(
        f"\n{'t-t0[s]':>8s}  {'+vy: freq[Hz]':>13s} {'amp[rad]':>9s} {'hipR amp':>9s}"
        f"   {'-vy: freq[Hz]':>13s} {'amp[rad]':>9s} {'hipR amp':>9s}"
    )
    for w0 in range(0, args.walk_steps - W + 1, S):
        row = [f"{w0 * 0.02:8.2f}"]
        for sel in (ok[:half].nonzero().flatten(), (ok[half:].nonzero().flatten() + half)):
            kp = knee_pos[w0 : w0 + W, sel]
            kv = knee_vel[w0 : w0 + W, sel]
            hp = hipr_pos[w0 : w0 + W, sel]
            zc = (torch.diff(torch.sign(kv), dim=0) != 0).sum(dim=0).float()
            freq = float((zc / 2 / (W * 0.02)).mean())
            row.append(f"{freq:13.2f} {float(kp.std(dim=0).mean()):9.3f} " f"{float(hp.std(dim=0).mean()):9.3f}")
        print("  ".join(row))

    print("\n-- verdict guide --")
    print("sim amp ~0 for 0.5-1 s then abrupt jump to ~0.1 rad small/fast steps")
    print("   -> the two-regime gait is the policy's own lateral style (reward/gait-basin;")
    print("      plant fixes will keep changing nothing)")
    print("sim steps cleanly from onset (amp ramps immediately, cadence steady)")
    print("   -> real-only trigger: floor friction / sole compliance / encoder-session bias")

    evaluator.cleanup()


if __name__ == "__main__":
    main()
