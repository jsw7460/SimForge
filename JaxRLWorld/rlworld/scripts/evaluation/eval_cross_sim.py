"""Cross-simulator evaluation: evaluate a checkpoint on a different simulator.

The robot is auto-detected from the checkpoint, and observation/algorithm/nn
configs are automatically restored.  You only need to specify which simulator
to evaluate on.

By default, launches an interactive real-time viewer (play mode).
Pass --eval for batch evaluation with statistics.

Examples:
    # Interactive play (default)
    python -m rlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim newton

    # Batch evaluation with viser viewer
    python -m rlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim genesis --eval
"""

import argparse

from rlworld.rl.evals import PolicyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Cross-simulator evaluation")
    parser.add_argument("--policy_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--wandb_run_path", type=str, default=None, help="W&B run path")
    parser.add_argument("--eval_sim", type=str, required=True, choices=["genesis", "newton", "mujoco"])
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e+9,
        },
    }

    # Eval mode: env's built-in viser viewer runs during batch eval.
    # Play mode: no env viewer; PlayViewer creates its own.
    if args.eval:
        overrides["visualization"] = {
            "viser_port": args.port,
            "viewer_type": "viser",
        }

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        wandb_run_path=args.wandb_run_path,
        num_evals=args.num_evals,
        record_video=args.record_video,
        extra_overrides=overrides,
    )

    # ====== Friction
    env = evaluator.env
    # terrain = env.scene_manager.scene.terrain
    # geom_ids = terrain.indexing.geom_ids
    # env.scene_manager.model.geom_friction[:, geom_ids, 0] = 0.01

    robot = env.scene_manager.scene["robot"]

    # entity-local idx → global geom id convert
    foot_geom_names = ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")
    local_ids, _ = robot.find_geoms(foot_geom_names)
    foot_geom_ids = robot.indexing.geom_ids[local_ids]

    env.scene_manager.model.geom_friction[:, foot_geom_ids, 0] = 0.3
    # ===== Friction

    if args.eval:
        stats = evaluator.evaluate()
        print(f"\nMean return: {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
    else:
        evaluator.play(port=args.port)


if __name__ == "__main__":
    main()
