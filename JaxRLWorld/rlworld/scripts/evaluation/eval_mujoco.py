"""MuJoCo (mjlab) policy evaluation entry point.

Mirrors ``eval_newton.py`` and ``eval_genesis.py``. Loads a trained
policy from a local path or W&B run and either runs batch evaluation
or launches the interactive viser viewer.

The mjlab scene config (``mjlab_scene_cfg``) is **automatically
reconstructed** from the checkpoint via the ``preset_class_name +
preset_kwargs`` mechanism added in the eval-script fix — there is no
need to pass ``mjlab_scene_cfg`` through ``extra_overrides`` like the
old workflow required.

Usage::

    # Interactive viewer
    python rlworld/scripts/evaluation/eval_mujoco.py

    # Batch evaluation (no viewer)
    python rlworld/scripts/evaluation/eval_mujoco.py --eval

    # Custom viser port
    python rlworld/scripts/evaluation/eval_mujoco.py --port 2027
"""
import argparse

from rlworld.rl.evals import PolicyEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo (mjlab) evaluation")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run batch evaluation instead of interactive viewer",
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e9,
        },
    }

    if not args.eval:
        overrides["visualization"] = {
            "viewer_type": "viser",
            "viser_port": args.port,
        }

    evaluator = PolicyEvaluator(
        policy_path="outputs/models/2026-04-19/22-05-45/checkpoint_latest/",
        # wandb_run_path="jsw7460/RLArchitecture/tcaoir1x",
        num_evals=1,
        seed=42,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )

    if args.eval:
        evaluator.evaluate()
    else:
        evaluator.play(port=args.port)
