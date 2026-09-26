"""Convert the LAFAN1 locomotion clips retargeted to the Booster K1 into
MotionCommand-ready NPZs (the reference motions of the K1 motion prior).

Source: the Hugging Face dataset ``whirlwind-ams/lafan_locomotion_k1``, one
parquet shard with one row per clip: ``root_pos`` (m), ``root_rot_xyzw``,
``dof_pos`` (rad, 22 joints in the K1 MJCF's joint order) at 30 fps. Each clip
is resampled to ``output_fps`` (LERP / SLERP, central-difference velocities,
the same passes as ``jaxrlworld.tools.motion.csv_to_npz``) and replayed
through the K1 MJCF for per-body world state, then written in the NPZ layout
every motion consumer in the framework reads.

The parquet also carries per-clip ``local_body_pos`` for a different
pipeline; it is not used, the replayer recomputes body state from the MJCF.

Run once (needs ``pyarrow``; the Hub shard is cached by ``huggingface_hub``)::

    python -m jaxrlworld.scripts.k1.convert_lafan_k1

``--source`` also takes a local ``.parquet`` path. A ``manifest.json`` next
to the NPZs records the dataset revision and conversion settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro
from huggingface_hub import HfApi, hf_hub_download

from jaxrlworld.rl.configs.robots.k1 import K1Config
from jaxrlworld.tools.motion.motion_loader import InterpolatedMotion, _lerp, _slerp, _so3_derivative
from jaxrlworld.tools.motion.mujoco_replayer import replay_motion

DATASET_ID = "whirlwind-ams/lafan_locomotion_k1"
_PARQUET_IN_REPO = "data/motions.parquet"
_DEFAULT_OUT = str(Path(__file__).resolve().parents[2] / "assets" / "motions" / "lafan1_k1")


def resample_pose_trajectory(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> InterpolatedMotion:
    """LERP/SLERP resample of a pose trajectory plus central-difference velocities.

    The interpolation and velocity passes of ``CsvMotionLoader`` on arrays
    instead of a CSV: frame ``i`` of the input sits at ``i / input_fps``, the
    output grid is ``arange(0, duration, 1 / output_fps)``.
    """
    input_dt = 1.0 / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    n_in = base_pos.shape[0]
    duration = (n_in - 1) * input_dt

    times = np.arange(0.0, duration, output_dt, dtype=np.float32)
    phase = times / duration
    idx_0 = np.floor(phase * (n_in - 1)).astype(np.int64)
    idx_1 = np.minimum(idx_0 + 1, n_in - 1)
    blend = (phase * (n_in - 1) - idx_0)[:, None]

    base_pos_o = _lerp(base_pos[idx_0], base_pos[idx_1], blend).astype(np.float32)
    base_quat_o = _slerp(base_quat_wxyz[idx_0], base_quat_wxyz[idx_1], blend).astype(np.float32)
    dof_pos_o = _lerp(dof_pos[idx_0], dof_pos[idx_1], blend).astype(np.float32)

    return InterpolatedMotion(
        base_pos=base_pos_o,
        base_quat_wxyz=base_quat_o,
        base_lin_vel=np.gradient(base_pos_o, output_dt, axis=0).astype(np.float32),
        base_ang_vel=_so3_derivative(base_quat_o, output_dt).astype(np.float32),
        dof_pos=dof_pos_o,
        dof_vel=np.gradient(dof_pos_o, output_dt, axis=0).astype(np.float32),
        fps=float(output_fps),
    )


def _resolve_source(source: str) -> tuple[Path, dict]:
    """A local parquet path, or a Hub dataset id fetched through the cache."""
    path = Path(source)
    if path.is_file():
        return path, {"source": str(path.resolve())}
    sha = HfApi().dataset_info(source).sha
    shard = hf_hub_download(source, _PARQUET_IN_REPO, repo_type="dataset", revision=sha)
    return Path(shard), {"source": source, "revision": sha, "file": _PARQUET_IN_REPO}


def main(
    source: str = DATASET_ID,
    output_dir: str = _DEFAULT_OUT,
    output_fps: float = 50.0,
) -> None:
    """Convert every clip of the parquet into ``<output_dir>/<clip name>.npz``.

    Args:
        source: Hub dataset id (``namespace/repo``) or a local parquet path.
        output_dir: Where the NPZs and ``manifest.json`` go.
        output_fps: Target rate; 50 Hz is the K1 velocity preset's control rate.
    """
    parquet, provenance = _resolve_source(source)
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"{parquet}: no clips")

    mjcf = Path(K1Config().mjcf_path)
    if not mjcf.is_file():
        raise FileNotFoundError(f"K1 MJCF not found: {mjcf} (run from the SimForge root)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips = []
    joint_names: list[str] | None = None
    print(f"[convert_lafan_k1] {len(rows)} clips from {provenance['source']} -> {out}")
    for i, row in enumerate(rows, 1):
        name = str(row["name"])
        fps_in = float(row["fps"])
        names = [str(n) for n in row["joint_names"]]
        if joint_names is None:
            joint_names = names
        elif names != joint_names:
            raise ValueError(f"{name}: joint order differs from the first clip's")
        root_pos = np.asarray(row["root_pos"], dtype=np.float64)
        root_rot_xyzw = np.asarray(row["root_rot_xyzw"], dtype=np.float64)
        dof_pos = np.asarray(row["dof_pos"], dtype=np.float64)
        if root_pos.shape[0] < 2:
            raise ValueError(f"{name}: {root_pos.shape[0]} frames")
        if dof_pos.shape[1] != len(joint_names):
            raise ValueError(f"{name}: {dof_pos.shape[1]} dof columns vs {len(joint_names)} joint names")
        if not (np.isfinite(root_pos).all() and np.isfinite(root_rot_xyzw).all() and np.isfinite(dof_pos).all()):
            raise ValueError(f"{name}: non-finite input")
        quat_norm = np.linalg.norm(root_rot_xyzw, axis=1)
        if np.abs(quat_norm - 1.0).max() > 1e-4:
            raise ValueError(f"{name}: root quaternion norm off by {np.abs(quat_norm - 1.0).max():.2e}")

        motion = resample_pose_trajectory(
            base_pos=root_pos,
            base_quat_wxyz=root_rot_xyzw[:, [3, 0, 1, 2]],
            dof_pos=dof_pos,
            input_fps=fps_in,
            output_fps=output_fps,
        )
        baked = replay_motion(mjcf_path=str(mjcf), motion=motion, joint_names=joint_names)
        dst = out / f"{name}.npz"
        np.savez(dst, **baked)
        n_out = int(motion.dof_pos.shape[0])
        print(
            f"[{i}/{len(rows)}] {name}: {root_pos.shape[0]} frames @ {fps_in:g} Hz -> "
            f"{n_out} @ {output_fps:g} Hz ({n_out / output_fps:.2f} s) -> {dst.name}"
        )
        clips.append({"name": name, "input_frames": int(root_pos.shape[0]), "output_frames": n_out})

    manifest = {
        **provenance,
        "output_fps": output_fps,
        "mjcf": str(mjcf),
        "joint_names": joint_names,
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[convert_lafan_k1] wrote {len(clips)} clips + manifest.json")


if __name__ == "__main__":
    tyro.cli(main)
