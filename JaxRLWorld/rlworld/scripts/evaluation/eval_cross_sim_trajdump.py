"""Cross-simulator trajectory dump.

Roll out the same checkpoint on Newton and MuJoCo for the same number
of deterministic steps, dump per-step joint pos/vel/torque + root state
to numpy arrays, then print per-step ``Δ`` between the two sims.

The point: confirm whether ``joint_vel`` / ``applied_torque`` actually
diverge between Newton and MuJoCo when the policy walks well in both
— if they do, SysID's large per-step error is physical (sim model
gap); if they don't, the gap lives in the SysID pipeline.

Examples:
    python -m rlworld.scripts.evaluation.eval_cross_sim_trajdump \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --num_steps 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from rlworld.rl.evals import PolicyEvaluator


def _rollout(eval_sim: str, policy_path: str, num_steps: int, seed: int, wandb_run_path: str | None = None):
    """Build env+policy on ``eval_sim``, run ``num_steps`` deterministic
    steps from a fresh reset, return trajectory dict."""
    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e9,
            "seed": seed,
        },
    }
    evaluator = PolicyEvaluator(
        policy_path=policy_path,
        eval_target=eval_sim,
        wandb_run_path=wandb_run_path,
        num_evals=1,
        record_video=False,
        extra_overrides=overrides,
    )
    env = evaluator.env
    policy = evaluator.policy

    env.reset()
    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    T = num_steps
    nj = env.act_manager.total_action_dim
    out = {
        "joint_pos": np.zeros((T, nj), dtype=np.float32),
        "joint_vel": np.zeros((T, nj), dtype=np.float32),
        "joint_torque": np.zeros((T, nj), dtype=np.float32),
        "root_pos": np.zeros((T, 3), dtype=np.float32),
        "root_quat": np.zeros((T, 4), dtype=np.float32),
        "root_lin_vel": np.zeros((T, 3), dtype=np.float32),
        "root_ang_vel": np.zeros((T, 3), dtype=np.float32),
        "action": np.zeros((T, nj), dtype=np.float32),
    }

    for t in range(T):
        action = policy.get_action(obs, robot_states, deterministic=True)
        rd = env.robot_data
        out["joint_pos"][t] = rd.joint_pos[0].cpu().numpy()
        out["joint_vel"][t] = rd.joint_vel[0].cpu().numpy()
        out["root_pos"][t] = rd.root_link_pos_w[0].cpu().numpy()
        out["root_quat"][t] = rd.root_link_quat_w[0].cpu().numpy()
        out["root_lin_vel"][t] = rd.root_link_lin_vel_w[0].cpu().numpy()
        out["root_ang_vel"][t] = rd.root_link_ang_vel_w[0].cpu().numpy()
        out["action"][t] = action[0].cpu().numpy()

        obs, *_ = env.step(action)
        # Read torque AFTER step. ``robot_data.applied_torque`` reads
        # ``state.mujoco.qfrc_actuator`` on Newton, which is only
        # populated by the implicit-PD path; explicit-PD actuators
        # (IdealPDActuator etc.) write torques directly to
        # ``control.joint_f`` and stash a copy in ``act.applied_effort``.
        # Pull from actuator buffers if any are registered (matches
        # ``NewtonOpenLoopEvaluator._read_applied_torques``); fall
        # back to the data-side qfrc_actuator otherwise (mjlab path).
        actuators = env.act_manager._actuators
        if actuators:
            nj_total = env.act_manager.total_action_dim
            tau = torch.zeros(env.num_envs, nj_total, device=env.device)
            for act, idx in actuators:
                tau[:, idx] = act.applied_effort
            out["joint_torque"][t] = tau[0].cpu().numpy()
        else:
            out["joint_torque"][t] = env.robot_data.applied_torque[0].cpu().numpy()
        robot_states = env.get_robot_state()

    return out, env.act_manager.actuated_joint_names


def _summary(name: str, x: np.ndarray):
    return f"  {name:<14s}  mean|·|={np.abs(x).mean():.4f}  " f"max|·|={np.abs(x).max():.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy_path",
        default=None,
        help="Local checkpoint dir. Mutually exclusive with " "--wandb_run_path; provide exactly one.",
    )
    p.add_argument(
        "--wandb_run_path",
        default=None,
        help="Wandb run path (e.g. entity/project/run_id). " "PolicyEvaluator auto-downloads the checkpoint.",
    )
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/cross_sim_trajdump",
        help="Where to save per-sim npz files.",
    )
    args = p.parse_args()

    if (args.policy_path is None) == (args.wandb_run_path is None):
        p.error("Provide exactly one of --policy_path / --wandb_run_path.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] rollout on Newton  ({args.num_steps} steps)")
    nw, names_n = _rollout("newton", args.policy_path, args.num_steps, args.seed, args.wandb_run_path)
    print(f"[2/2] rollout on MuJoCo  ({args.num_steps} steps)")
    mj, names_m = _rollout("mujoco", args.policy_path, args.num_steps, args.seed, args.wandb_run_path)

    # Joint name sanity check (same actuator order both sides)
    if list(names_n) != list(names_m):
        print("\nWARNING: actuator joint name lists differ between sims!")
        print(f"  newton: {list(names_n)}")
        print(f"  mujoco: {list(names_m)}")

    # Save raw rollouts.
    np.savez(out_dir / "newton.npz", **nw, joint_names=np.array(list(names_n)))
    np.savez(out_dir / "mujoco.npz", **mj, joint_names=np.array(list(names_m)))
    print(f"\nSaved: {out_dir/'newton.npz'}")
    print(f"Saved: {out_dir/'mujoco.npz'}")

    # Per-step diff summary.
    print("\n=== Δ (Newton − MuJoCo) per-step magnitude ===")
    for k in ("joint_pos", "joint_vel", "joint_torque", "root_pos", "root_lin_vel", "root_ang_vel", "action"):
        d = nw[k] - mj[k]
        print(_summary(k, d))

    # First / mid / last step joint_torque and joint_vel head-to-head.
    pick = [0, args.num_steps // 2, args.num_steps - 1]
    print("\n=== Sample steps (Newton vs MuJoCo) ===")
    for t in pick:
        print(f"\nstep t={t}")
        print(f"  joint_vel newton    : {nw['joint_vel'][t].round(3)}")
        print(f"  joint_vel mujoco    : {mj['joint_vel'][t].round(3)}")
        print(f"  joint_torque newton : {nw['joint_torque'][t].round(2)}")
        print(f"  joint_torque mujoco : {mj['joint_torque'][t].round(2)}")
        print(f"  root_lin_vel n / m  : {nw['root_lin_vel'][t].round(3)} / " f"{mj['root_lin_vel'][t].round(3)}")


if __name__ == "__main__":
    main()
