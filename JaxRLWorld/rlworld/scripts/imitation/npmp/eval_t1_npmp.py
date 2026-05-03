"""Standalone evaluation entry point for trained NPMP policies.

Two modes:

  --eval (default):
      Deterministic batch rollout. Prints a diagnostic table with
      tracking reward, episode length, per-reward-term breakdown,
      termination breakdown, latent z statistics, encoder posterior
      log-std, and per-motion breakdown. Optionally also computes
      ``||action_NPMP - μ_E||`` per step when ``--with_experts`` is
      set (the same expert checkpoint dict from the training entry
      script is reused here to query each motion's expert at the
      observations the NPMP module visits).

  --play:
      Launches the existing ``ViserPlayViewer`` with NPMP driving the
      env. The viewer's Motion tab automatically appears and clicks
      on the dropdown switch ``MotionCommand.set_motion_clip``; the
      encoder picks up the new motion's reference window and the
      decoder's action tracks it in real time.

Run::

    jaxpy JaxRLWorld/rlworld/scripts/imitation/npmp/eval_t1_npmp.py \
        --policy_path outputs/models/.../checkpoint_latest

    jaxpy JaxRLWorld/rlworld/scripts/imitation/npmp/eval_t1_npmp.py \
        --wandb_run_path jsw7460/T1_NPMP/abc123 --play --port 2026
"""
from __future__ import annotations

import argparse

import jax

from rlworld.imitation.npmp import (
    CheckpointRef,
    MultiExpertDispatcher,
    NPMPEvaluator,
    T1NPMPDistillConfig,
)
from rlworld.scripts.imitation.npmp.train_t1_npmp import EXPERT_REFS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained NPMP")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--policy_path", type=str,
        help="Local NPMP checkpoint dir (contains model.eqx + npmp_meta.yaml).",
    )
    src.add_argument(
        "--wandb_run_path", type=str,
        help="WandB run path 'entity/project/run_id' for an NPMP training run.",
    )
    parser.add_argument(
        "--wandb_checkpoint_iter", type=int, default=None,
        help="Specific NPMP iteration to pull from wandb (default: latest).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--eval", action="store_true", default=True,
        help="Batch deterministic eval (default).",
    )
    mode.add_argument(
        "--play", action="store_true",
        help="Launch viser interactive viewer instead of batch eval.",
    )

    parser.add_argument(
        "--num_envs", type=int, default=90,
        help="Env count for eval (must be ≥ 9 for per-motion split).",
    )
    parser.add_argument(
        "--num_steps", type=int, default=500,
        help="Eval rollout length in env steps.",
    )
    parser.add_argument(
        "--port", type=int, default=2026,
        help="Viser port (--play mode).",
    )
    parser.add_argument(
        "--with_experts", action="store_true",
        help=(
            "Also load the 9 T1 tracking experts (from train_t1_npmp.py's "
            "EXPERT_REFS) to compute the action-gap diagnostic during eval."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.policy_path is not None:
        ckpt_ref = CheckpointRef(local_path=args.policy_path)
    else:
        ckpt_ref = CheckpointRef(
            wandb_run_path=args.wandb_run_path,
            wandb_checkpoint_iter=args.wandb_checkpoint_iter,
        )

    # eval-time config: smaller env count, eval-friendly defaults.
    cfg = T1NPMPDistillConfig(
        sim_type="newton",
        num_envs=args.num_envs,
        expert_refs=EXPERT_REFS,
        seed=args.seed,
        use_wandb=False,  # eval shouldn't open a new wandb run
    )

    evaluator = NPMPEvaluator(npmp_ckpt=ckpt_ref, cfg=cfg, seed=args.seed)

    if args.with_experts:
        expert_paths = cfg.resolve_expert_paths()
        key = jax.random.PRNGKey(args.seed + 1)
        dispatcher = MultiExpertDispatcher(
            checkpoint_paths=expert_paths,
            env=evaluator.env,
            key=key,
        )
        evaluator.attach_dispatcher(dispatcher)
        print(f"Loaded {len(expert_paths)} experts for action-gap eval.")

    if args.play:
        print(
            f"Launching viser viewer on port {args.port}. "
            "Use the Motion tab to switch the tracked clip."
        )
        evaluator.play(port=args.port)
    else:
        stats = evaluator.evaluate(num_steps=args.num_steps)
        print(stats.format_table())


if __name__ == "__main__":
    main()
