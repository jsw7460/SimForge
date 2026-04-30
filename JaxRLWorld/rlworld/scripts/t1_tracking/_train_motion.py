"""Shared entry point for per-motion T1 tracking training scripts.

Each ``rlworld/scripts/t1_tracking/<sim>/motion_<name>.py`` is a 3-line
file that calls :func:`train_single_motion` with its sim and motion
clip name. The preset (:class:`T1TrackingConfig`) is reused unchanged
— only ``motion_files`` and ``run_name`` are overridden per script,
so wandb runs and checkpoints stay separated by clip without forking
the preset itself.
"""
from __future__ import annotations

from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig
from rlworld.rl.runners import BaseRunner

_MOTION_DIR = "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted"
_SIM_TAG = {"newton": "NT", "mujoco": "MJ", "genesis": "GS"}


def train_single_motion(sim_type: str, motion_name: str) -> None:
    """Train a T1 tracking policy on a single Booster motion clip.

    Args:
        sim_type: ``"newton"`` / ``"mujoco"`` / ``"genesis"``.
        motion_name: NPZ basename (without ``.npz``) under
            ``booster_t1_converted/`` — e.g. ``"running"``.
    """
    cfgs_for_run = (
        T1TrackingConfig(
            sim_type=sim_type,
            motion_files=(f"{_MOTION_DIR}/{motion_name}.npz",),
            run_name=f"T1_Tracking_{_SIM_TAG[sim_type]}_MLP_{motion_name}",
        )
        .build()
        .with_cli_overrides()
    )
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )
