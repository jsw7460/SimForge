"""Utilities for downloading JaxRLWorld checkpoints from wandb Artifacts."""

from __future__ import annotations

import os
import re


def get_wandb_checkpoint(
    wandb_run_path: str,
    iteration: int | None = None,
    cache_dir: str | None = None,
) -> tuple[str, bool]:
    """Download a checkpoint artifact from a wandb run.

    Args:
        wandb_run_path: Run path in the form "entity/project/run_id".
        iteration: Specific iteration to download. None means latest (highest iteration).
        cache_dir: Local directory to cache downloads. Defaults to ./wandb_checkpoints/<run_id>/.

    Returns:
        (checkpoint_path, was_cached) — local path and whether the result came from cache.

    Raises:
        ImportError: If wandb is not installed.
        ValueError: If no checkpoint artifacts are found for the run.
    """
    import wandb

    run_id = wandb_run_path.split("/")[-1]

    # Collect checkpoint artifacts
    checkpoints = list_wandb_checkpoints(wandb_run_path)

    if not checkpoints:
        raise ValueError(f"No checkpoint artifacts found for run {wandb_run_path}")

    # Select artifact
    if iteration is not None:
        matches = [c for c in checkpoints if c["iteration"] == iteration]
        if not matches:
            available = [c["iteration"] for c in checkpoints]
            raise ValueError(
                f"Iteration {iteration} not found. Available: {available}"
            )
        selected = matches[0]
    else:
        selected = max(checkpoints, key=lambda c: c["iteration"])

    # Determine cache path
    if cache_dir is None:
        cache_dir = os.path.join("wandb_checkpoints", run_id)
    download_root = os.path.join(cache_dir, selected["name"])

    # Check cache
    if os.path.isdir(download_root) and os.listdir(download_root):
        print(f"Using cached checkpoint: {download_root}")
        return download_root, True

    # Download
    api = wandb.Api()
    artifact = api.artifact(selected["full_name"])
    artifact.download(root=download_root)
    print(f"Downloaded checkpoint to: {download_root}")
    return download_root, False


def list_wandb_checkpoints(wandb_run_path: str) -> list[dict]:
    """List available checkpoint artifacts for a wandb run.

    Args:
        wandb_run_path: Run path in the form "entity/project/run_id".

    Returns:
        List of dicts with keys: name, full_name, iteration, version, created_at.
    """
    import wandb

    api = wandb.Api()

    results = []
    for artifact in api.run(wandb_run_path).logged_artifacts():
        if artifact.type != "checkpoint":
            continue

        # Parse iteration from artifact name (e.g. "checkpoint-iter5000:v0")
        match = re.search(r"iter(\d+)", artifact.name)
        if match:
            iteration = int(match.group(1))
        else:
            continue

        results.append({
            "name": artifact.name.split(":")[0],
            "full_name": artifact.qualified_name,
            "iteration": iteration,
            "version": artifact.version,
            "created_at": artifact.created_at,
        })

    results.sort(key=lambda c: c["iteration"])
    return results
