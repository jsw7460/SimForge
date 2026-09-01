"""Log a trained K1 policy's joints in sim, eval-style (UNMODIFIED command).

The policy just walks under its own random velocity commands — exactly the eval
path that already works — and we log per-step joint_pos / joint_torque / command
/ root_quat for ALL envs. Gait left/right (a)symmetry per direction is then
sliced OFFLINE from the logged command (no command pinning, no extra reset, no
settle — those only broke the walk).

Run (server, from SimForge root; jaxpy for the JAX policy):
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_sim_deploy_rollout \\
        --wandb-run-path jsw7460/K1_Joystick/wdx6erdb --sim mujoco \\
        --out ~/workspace/JaxRLWorld-private/deploy/booster_k1/data/sim_wdx6erdb_mujoco.npz
"""

from __future__ import annotations

import argparse


def main() -> int:
    import numpy as np
    import torch

    from jaxrlworld.rl.evals import PolicyEvaluator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-run-path", required=True)
    ap.add_argument("--sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[rollout] loading {args.wandb_run_path} on {args.sim} (num_envs={args.num_envs})")
    evaluator = PolicyEvaluator(
        wandb_run_path=args.wandb_run_path,
        eval_target=args.sim,
        num_evals=1,
        seed=args.seed,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    env = evaluator.env
    policy = evaluator.policy
    jnames = list(env.act_manager.actuated_joint_names)

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    JP, JT, CMD, RQ = [], [], [], []
    with torch.no_grad():
        for step in range(args.steps):
            action = policy.get_action(obs, robot_states)
            obs, _rew, term, trunc, _extras = env.step(action)
            robot_states = env.get_robot_state()
            reset_now = term | trunc
            if reset_now.any():
                policy.notify_reset(reset_now.cpu().numpy())
            rd = env.get_robot_data()
            JP.append(rd.joint_pos.cpu().numpy())  # (N, 22)
            JT.append(rd.applied_torque.cpu().numpy())  # (N, 22)
            CMD.append(env.command_manager.get_commands_tensor().cpu().numpy())  # (N, 3)
            RQ.append(rd.root_link_quat_w.cpu().numpy())  # (N, 4) wxyz
            if step % 500 == 0:
                print(f"  step {step}/{args.steps}")

    np.savez(
        args.out,
        joint_pos=np.asarray(JP),  # (steps, N, 22)
        joint_torque=np.asarray(JT),
        command=np.asarray(CMD),  # (steps, N, 3)
        root_quat=np.asarray(RQ),  # (steps, N, 4)
        joint_names=np.asarray(jnames),
        control_dt=float(env.control_dt),
    )
    print(f"[rollout] saved {args.out}  shape joint_pos={np.asarray(JP).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
