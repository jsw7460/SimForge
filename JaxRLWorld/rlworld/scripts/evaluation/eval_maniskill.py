"""Evaluate a trained ManiSkill policy.

Give only ``--policy_path``: PolicyEvaluator reads the checkpoint metadata,
re-runs the saved preset (which carries the task), rebuilds the ManiSkill env,
and runs a deterministic batch evaluation. The task is recovered from the
checkpoint -- there is no per-task eval entry point.

ManiSkill has no Viser bridge, so this runs batch eval only; ``--record_video``
saves an mp4 via ManiSkill's native RecordEpisode (SAPIEN renderer).

Run (JAX-based -> jaxpy to avoid GPU preallocation/OOM):

    jaxpy -m rlworld.scripts.evaluation.eval_maniskill \
        --policy_path outputs/models/.../checkpoint_latest --record_video
"""

import argparse

from rlworld.rl.evals import PolicyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ManiSkill policy")
    parser.add_argument("--policy_path", type=str, required=True, help="checkpoint dir")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_evals", type=int, default=100, help="episodes to evaluate")
    parser.add_argument("--record_video", action="store_true", help="save mp4 via RecordEpisode")
    args = parser.parse_args()

    # eval_target=None -> evaluate on the training simulator (ManiSkill),
    # auto-detected from the checkpoint along with the task.
    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=None,
        num_evals=args.num_evals,
        record_video=args.record_video,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )

    stats = evaluator.evaluate()
    print(f"\nMean return: {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
    if stats.get("success_rate") is not None:
        print(f"Success rate: {stats['success_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
