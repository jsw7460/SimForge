import argparse

from rlworld.rl.evals import PolicyEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Newton evaluation")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e+9,
        },
        # "command": {
        #     "rel_standing_envs": 0.3,
        # },
    }

    if args.eval:
        overrides["visualization"] = {
            "viser_port": args.port,
            "viewer_type": "viser",
        }

    evaluator = PolicyEvaluator(
        policy_path="./outputs/models/2026-04-01/11-09-28/checkpoint_latest/",
        seed=42,
        num_evals=100000000,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )

    if args.eval:
        evaluator.evaluate()
    else:
        evaluator.play(port=args.port)
