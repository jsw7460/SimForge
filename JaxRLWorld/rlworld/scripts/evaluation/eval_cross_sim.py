"""Cross-simulator evaluation: evaluate a checkpoint on a different simulator.

The robot is auto-detected from the checkpoint, and observation/algorithm/nn
configs are automatically restored.  You only need to specify which simulator
to evaluate on.

Examples:
    # Genesis-trained Go2 checkpoint -> Newton eval
    python -m rlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim newton

    # MultiSim (Genesis+Newton) checkpoint -> Genesis-only eval
    python -m rlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim genesis --num_envs 5
"""

import argparse

from rlworld.rl.evals import PolicyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Cross-simulator evaluation")
    parser.add_argument("--policy_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--eval_sim", type=str, required=True, choices=["genesis", "newton", "mujoco"])
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    args = parser.parse_args()

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        num_evals=args.num_evals,
        show_viewer=args.show_viewer,
        record_video=args.record_video,
        extra_overrides={
            "env": {
                "num_envs": 1,
                "episode_length_s": 10e+9,
            },
            "visualization": {
                "viser_port": 2026,
                "viewer_type": "viser",
            },
            "command": {
                "rel_standing_envs": 0.3
            }
        },
    )

    stats = evaluator.evaluate()
    print(f"\nMean return: {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")


if __name__ == "__main__":
    main()
