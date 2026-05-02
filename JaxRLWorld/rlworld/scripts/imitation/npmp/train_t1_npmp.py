"""Entry point for T1 NPMP distillation training.

Loads N pretrained T1 tracking experts (one per Booster motion clip),
distils them into a single NPMP motor primitive module via online
behavioural cloning with DART noise.

Fill in :data:`EXPERT_REFS` with the actual checkpoint refs (local
directory or wandb run path) corresponding 1:1 to ``MOTION_FILES`` in
the same order. ``T1NPMPDistillConfig`` defaults ``motion_files`` to
all nine ``booster_t1_converted`` clips; override here if a different
set is desired.

Run::

    jaxpy JaxRLWorld/rlworld/scripts/imitation/npmp/train_t1_npmp.py
"""
from __future__ import annotations

import jax

from rlworld.imitation.npmp import (
    CheckpointRef,
    MultiExpertDispatcher,
    NPMPTrainer,
    T1NPMPDistillConfig,
)
from rlworld.rl.runners import BaseRunner


# ── Expert checkpoints — keyed by motion clip basename ─────────────────
#
# Keys MUST cover exactly the set of basenames in
# T1NPMPDistillConfig.motion_files (default: the nine booster T1
# clips). Each value is a CheckpointRef pointing at the policy
# trained on that specific clip:
#
#   CheckpointRef(local_path="outputs/models/.../checkpoint_latest/")
#   CheckpointRef(wandb_run_path="entity/project/run_id")
#   CheckpointRef(wandb_run_path="entity/project/run_id", wandb_checkpoint_iter=N)
#
# Mixing local and wandb refs in the same dict is fine.
EXPERT_REFS: dict[str, CheckpointRef] = {
    "goal_kick":        CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/pjx0oy02"),
    "jogging":          CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/d7i19dh1"),
    "kick_ball2":       CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/sg4v0w07"),
    "kick_ball3":       CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/ejv6dw58"),
    "pass_ball1":       CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/qhghwhkc"),
    "powerful_kick":    CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/ba3k5rvn"),
    "running":          CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/myu6iqr3"),
    "soccer_drill_run": CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/r5nftv4e"),
    "walking1":         CheckpointRef(wandb_run_path="jsw7460/T1_Tracking/48a9w8jz"),
}


def main() -> None:
    cfg = T1NPMPDistillConfig(
        sim_type="newton",
        expert_refs=EXPERT_REFS,
    )

    cfgs_for_run = cfg.build()

    # Build env via the existing runner factory — handles sim-specific
    # wiring (Newton scene manager, sensors, etc.) the same way training
    # entry scripts do.
    env = BaseRunner._create_env_from_config(cfgs_for_run)

    key = jax.random.PRNGKey(cfg.seed)
    key_dispatcher, key_trainer = jax.random.split(key)

    expert_paths = cfg.resolve_expert_paths()
    print(f"Loading {len(expert_paths)} experts:")
    for i, p in enumerate(expert_paths):
        print(f"  [{i}] {p}")

    dispatcher = MultiExpertDispatcher(
        checkpoint_paths=expert_paths,
        env=env,
        key=key_dispatcher,
    )

    trainer = NPMPTrainer(
        cfg=cfg, env=env, dispatcher=dispatcher, key=key_trainer,
    )

    print(f"Saving NPMP checkpoints to: {trainer.model_log_dir}")
    trainer.train(num_iterations=cfg.num_iterations)


if __name__ == "__main__":
    main()
