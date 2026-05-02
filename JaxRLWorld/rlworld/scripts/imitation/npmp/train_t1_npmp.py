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
from rlworld.rl.utils.utils import setup_log_dir


# ── Expert checkpoints — one per motion clip, same order as motion_files ──
#
# Each entry is either:
#   CheckpointRef(local_path="outputs/models/.../checkpoint_latest/")
#   CheckpointRef(wandb_run_path="entity/project/run_id")
#   CheckpointRef(wandb_run_path="entity/project/run_id", wandb_checkpoint_iter=N)
#
# The order MUST match T1NPMPDistillConfig's motion_files order:
#   0 goal_kick    1 jogging      2 kick_ball2     3 kick_ball3
#   4 pass_ball1   5 powerful_kick  6 running       7 soccer_drill_run
#   8 walking1
EXPERT_REFS: tuple[CheckpointRef, ...] = (
    CheckpointRef(local_path="<TODO: goal_kick checkpoint path>"),
    CheckpointRef(local_path="<TODO: jogging checkpoint path>"),
    CheckpointRef(local_path="<TODO: kick_ball2 checkpoint path>"),
    CheckpointRef(local_path="<TODO: kick_ball3 checkpoint path>"),
    CheckpointRef(local_path="<TODO: pass_ball1 checkpoint path>"),
    CheckpointRef(local_path="<TODO: powerful_kick checkpoint path>"),
    CheckpointRef(local_path="<TODO: running checkpoint path>"),
    CheckpointRef(local_path="<TODO: soccer_drill_run checkpoint path>"),
    CheckpointRef(local_path="<TODO: walking1 checkpoint path>"),
)


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

    save_dir, _ = setup_log_dir(output_dir="auto")
    print(f"Saving NPMP checkpoints to: {save_dir}")

    trainer.train(num_iterations=cfg.num_iterations, save_dir=save_dir)


if __name__ == "__main__":
    main()
