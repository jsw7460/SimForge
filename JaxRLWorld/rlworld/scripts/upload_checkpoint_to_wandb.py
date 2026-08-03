"""Attach a local checkpoint directory to its wandb run as a checkpoint Artifact.

Training records the run it logged to in ``train_state.yaml``
(``wandb_run_path``) even when checkpoint-artifact upload was disabled, so a
local-only checkpoint can be uploaded to the right run after the fact:

    python -m rlworld.scripts.upload_checkpoint_to_wandb \
        --checkpoint outputs/models/2026-07-27/17-16-39/checkpoint_18000

The artifact is named/typed exactly as the training uploader
(``checkpoint-iter<N>`` / type ``checkpoint``) so the download path
(``get_wandb_checkpoint`` / ``export_deploy_policy --wandb_run_path``) picks it
up. The target run and iteration are read from the checkpoint; override with
``--wandb_run_path entity/project/run_id`` and ``--iteration N`` if needed.

Reads only wandb + PyYAML (no jax/torch), so run it with plain ``python``.
"""

from __future__ import annotations

import argparse
import os
import re

import wandb
import yaml


def _load_train_state(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "train_state.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"train_state.yaml not found in {checkpoint_dir!r}")
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_iteration(checkpoint_dir: str, train_state: dict, override: int | None) -> int:
    if override is not None:
        return override
    # A dir like ``checkpoint_18000`` names its iteration; ``checkpoint_latest``
    # does not, so fall back to the value train_state.yaml recorded.
    base = os.path.basename(os.path.normpath(checkpoint_dir))
    m = re.search(r"checkpoint[_-](\d+)", base)
    if m:
        return int(m.group(1))
    it = train_state.get("iteration")
    if it is None:
        raise ValueError(
            f"Cannot determine iteration: {base!r} has no number and "
            "train_state.yaml has no 'iteration'. Pass --iteration N."
        )
    return int(it)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach a local checkpoint to its wandb run.")
    ap.add_argument("--checkpoint", required=True, help="Local checkpoint directory.")
    ap.add_argument(
        "--wandb_run_path",
        default=None,
        help="Override target run 'entity/project/run_id' (default: from train_state.yaml).",
    )
    ap.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Override the checkpoint iteration (default: dir name / train_state.yaml).",
    )
    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise NotADirectoryError(f"Not a directory: {ckpt}")

    train_state = _load_train_state(ckpt)
    run_path = args.wandb_run_path or train_state.get("wandb_run_path")
    if not run_path:
        raise ValueError(
            "No 'wandb_run_path' in train_state.yaml (the run was not logged to "
            "wandb). Pass --wandb_run_path entity/project/run_id explicitly."
        )
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"wandb_run_path must be 'entity/project/run_id', got {run_path!r}")
    entity, project, run_id = parts

    iteration = _resolve_iteration(ckpt, train_state, args.iteration)
    artifact_name = f"checkpoint-iter{iteration}"

    print(f"[upload] checkpoint : {ckpt}")
    print(f"[upload] run        : {run_path}")
    print(f"[upload] artifact   : {artifact_name} (type=checkpoint)")

    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    artifact = wandb.Artifact(name=artifact_name, type="checkpoint")
    artifact.add_dir(ckpt)
    run.log_artifact(artifact)
    artifact.wait()  # block until the upload finishes before closing the run
    run.finish()
    print(f"[upload] done -> {run_path} :: {artifact_name}")


if __name__ == "__main__":
    main()
