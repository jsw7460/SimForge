"""Cross-simulator evaluation: evaluate a checkpoint on a different simulator.

The robot is auto-detected from the checkpoint, and observation/algorithm/nn
configs are automatically restored.  You only need to specify which simulator
to evaluate on.

By default, launches an interactive real-time viewer (play mode).
Pass --eval for batch evaluation with statistics.

Examples:
    # Interactive play (default)
    python -m jaxrlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim newton

    # Batch evaluation with viser viewer
    python -m jaxrlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ \
        --eval_sim genesis --eval

    # Straight from a W&B run (downloads/caches the latest checkpoint;
    # --wandb_checkpoint_iter picks one of the run's checkpoint-iterN artifacts)
    python -m jaxrlworld.scripts.evaluation.eval_cross_sim \
        --wandb_run_path <entity>/<project>/<run_id> --eval_sim newton \
        --wandb_checkpoint_iter 4000

    # Per-clip motion tracking protocol: one env per clip, each played once
    # from frame 0 (``clip_per_env``), startup randomization pinned, one row
    # per clip with the return normalized by the task's max step reward
    python -m jaxrlworld.scripts.evaluation.eval_cross_sim \
        --policy_path outputs/models/.../checkpoint_latest/ --eval_sim newton \
        --eval --num_envs 40 --per_env --max_step_reward 0.1 \
        command.terms.motion.sampling_mode=clip_per_env \
        event.foot_friction.params.friction_range=[1.0,1.0] event.randomize_body_com=null

Any ``section.path=value`` argument is a config override applied after the
checkpoint's config is restored (``null`` switches a term off), the same
syntax the training entry points take.
"""

import argparse

from jaxrlworld.rl.configs.base_config import parse_override_args
from jaxrlworld.rl.evals import PolicyEvaluator
from jaxrlworld.rl.vis.viser import get_look, list_looks


def _merge_into(target: dict, source: dict) -> None:
    """Deep-merge ``source`` into ``target``; a leaf in ``source`` wins."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_into(target[key], value)
        else:
            target[key] = value


def main():
    parser = argparse.ArgumentParser(description="Cross-simulator evaluation")
    parser.add_argument("--policy_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--wandb_run_path", type=str, default=None, help="W&B run path")
    parser.add_argument(
        "--wandb_checkpoint_iter",
        type=int,
        default=None,
        help="Iteration of the W&B checkpoint artifact to evaluate (default: the latest)",
    )
    parser.add_argument("--eval_sim", type=str, required=True, choices=["genesis", "newton", "mujoco"])
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help=(
            "Number of parallel envs. ``1`` keeps interactive viewer responsiveness "
            "minimal; larger values let Genesis amortise kernel-launch overhead per "
            "step (useful when interactive Genesis is slow at ``num_envs=1``)."
        ),
    )
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument(
        "--per_env",
        action="store_true",
        help="Batch eval: print one row per environment (return, length, how the episode ended)",
    )
    parser.add_argument(
        "--max_step_reward",
        type=float,
        default=None,
        help="Batch eval: the task's largest per-step reward; adds return / (length * this) to the per-env rows",
    )
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    parser.add_argument(
        "--look",
        type=str,
        default=None,
        choices=list_looks(),
        help="Viser scene 'look' (robot material / ground / lighting). See jaxrlworld/rl/vis/viser/looks.py. "
        "Omit to use the preset's default (near-black metallic robot, earthy ground, outdoor sun).",
    )
    args, rest = parser.parse_known_args()
    unknown = [arg for arg in rest if "=" not in arg]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if (args.policy_path is None) == (args.wandb_run_path is None):
        parser.error("pass exactly one of --policy_path / --wandb_run_path")

    overrides: dict = {
        "env": {
            "num_envs": args.num_envs,
            "episode_length_s": 10e9,
        },
    }
    # Eval mode: env's built-in viser viewer runs during batch eval.
    # Play mode: no env viewer; PlayViewer creates its own.
    if args.eval:
        overrides["visualization"] = {
            "viser_port": args.port,
            "viewer_type": "viser",
        }
    if args.look is not None:
        # Flows through to both the eval-mode env viewer and the play-mode
        # PlayViewer (both read VisualizationConfig.viser_scene).
        overrides.setdefault("visualization", {})["viser_scene"] = get_look(args.look)

    # ``section.path=value`` arguments win over every default above.
    for section, values in parse_override_args().items():
        _merge_into(overrides.setdefault(section, {}), values)

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        wandb_run_path=args.wandb_run_path,
        wandb_checkpoint_iter=args.wandb_checkpoint_iter,
        num_evals=args.num_evals,
        record_video=args.record_video,
        extra_overrides=overrides,
        per_env_report=args.per_env,
        max_step_reward=args.max_step_reward,
    )

    if args.eval:
        stats = evaluator.evaluate()
        print(f"\nMean return: {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
    else:
        evaluator.play(port=args.port)


if __name__ == "__main__":
    main()
