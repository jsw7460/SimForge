"""CLI: convert a Booster Robotics humanoid-trajectory NPZ into a MotionCommand-ready NPZ.

The Hugging Face dataset ``SaiResearch/booster_dataset`` stores per-frame
trajectories of Booster's T1 / K1 humanoids as NumPy archives whose body-
level fields (``xpos``, ``xquat``, ``body_names``, ...) are placeholder
zeros / ``None``; only ``qpos``, ``qvel``, ``joint_names``,
``split_points``, and ``frequency`` carry real content. JaxRLWorld's
``MotionLoader`` needs dense per-body world state, so this tool replays
the Booster trajectory through an MJCF (``mujoco_replayer.replay_motion``)
to recover ``body_pos_w`` / ``body_quat_w`` / ``body_lin_vel_w`` /
``body_ang_vel_w`` / ``body_names`` and writes the result in the same NPZ
layout produced by ``csv_to_npz.py``.

Booster NPZ fields consumed here:
    - ``qpos``        ``(T, 7 + J)`` — ``[root_pos(3), root_quat_wxyz(4), dof_pos(J)]``
    - ``joint_names`` ``(1 + J,)`` — first entry ``"root"`` is the free joint;
      remaining ``J`` hinge names match the ``qpos[:, 7:]`` column order.
    - ``frequency``   scalar — recorded FPS (500 Hz on the T1 clips).
    - ``split_points`` ``(>=2,)`` — segment boundaries; single-clip files
      carry ``[start_inclusive, end_inclusive]``.

The Booster ``qvel`` array is intentionally NOT consumed: we re-derive
velocities from the resampled pose path so the finite-difference dt
matches ``output_fps`` exactly (500 Hz central-diff velocities baked into
the source NPZ carry high-frequency noise that is inconsistent with the
downsampled 50 Hz pose stream).

Usage::

    uv run python -m rlworld.tools.motion.booster_to_npz \\
        --input-file JaxRLWorld/rlworld/assets/motions/booster/booster_t1/walking1.npz \\
        --output-file /tmp/t1_walking1.npz

Defaults target Booster T1 with the menagerie T1 MJCF, which is verified
body/joint-identical to ``Mjlab/.../booster_t1/xmls/t1.xml`` — so the
produced NPZ works for the mujoco, newton, and genesis builders of the
``t1_tracking`` preset.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import tyro

from rlworld.tools.motion.motion_loader import (
    InterpolatedMotion,
    _lerp,
    _slerp,
    _so3_derivative,
)
from rlworld.tools.motion.mujoco_replayer import replay_motion


# Default MJCF: menagerie T1. Verified (body 24 / hinge 23 / freejoint) to
# match mjlab's ``booster_t1/xmls/t1.xml`` exactly in body and joint layout,
# so the NPZ produced with this MJCF works across all three simulator
# backends of the t1_tracking preset.
_DEFAULT_MJCF = "./JaxRLWorld/rlworld/assets/menagerie_T1/t1.xml"


def _resample_pose_trajectory(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> InterpolatedMotion:
    """LERP/SLERP resample of a pose-only trajectory + central-diff velocities.

    Mirrors the interpolation + velocity passes inside ``CsvMotionLoader``
    without re-parsing a CSV. Used to downsample Booster's 500 Hz recording
    to the tracker's 50 Hz control frequency while producing dt-consistent
    velocities from the downsampled pose path.

    Booster recordings open with a brief settling transient (~50-200 ms)
    where the robot transitions from its controller's home pose to the
    motion's actual first frame. Once resampled to 50 Hz that transient
    becomes a 1-3 frame jump producing >30 rad/s joint velocities — i.e.
    nonsense. We trim leading frames where ``max(|dof_vel|)`` exceeds
    ``LEAD_VEL_THRESH``, with a small ``EPS_FRAMES`` lookahead so a single
    fast tracking frame doesn't keep the trim going indefinitely.
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
    base_quat_o = _slerp(
        base_quat_wxyz[idx_0], base_quat_wxyz[idx_1], blend,
    ).astype(np.float32)
    dof_pos_o = _lerp(dof_pos[idx_0], dof_pos[idx_1], blend).astype(np.float32)

    base_lin_vel = np.gradient(base_pos_o, output_dt, axis=0).astype(np.float32)
    dof_vel = np.gradient(dof_pos_o, output_dt, axis=0).astype(np.float32)
    base_ang_vel = _so3_derivative(base_quat_o, output_dt).astype(np.float32)

    # Trim leading high-velocity transient. Threshold chosen empirically:
    # human walking caps any individual joint at ~6 rad/s, kicks at ~10
    # rad/s. >15 rad/s on any joint at the very start of a clip is almost
    # certainly the recording's settling transient, not real motion.
    LEAD_VEL_THRESH = 15.0  # rad/s
    EPS_FRAMES = 3
    n_out = dof_pos_o.shape[0]
    stable_start = 0
    for t in range(n_out):
        max_v = float(np.max(np.abs(dof_vel[t])))
        if max_v >= LEAD_VEL_THRESH:
            stable_start = t + 1
            continue
        # Look ahead to confirm we've actually settled, not just passed
        # through a single low-velocity frame.
        end = min(t + EPS_FRAMES, n_out)
        max_lookahead = float(np.max(np.abs(dof_vel[t:end])))
        if max_lookahead < LEAD_VEL_THRESH:
            stable_start = t
            break
    if stable_start > 0:
        print(
            f"[booster_to_npz]   trimmed {stable_start} leading "
            f"high-velocity frames (>{LEAD_VEL_THRESH:.0f} rad/s settle "
            f"transient)"
        )
        base_pos_o = base_pos_o[stable_start:]
        base_quat_o = base_quat_o[stable_start:]
        dof_pos_o = dof_pos_o[stable_start:]
        base_lin_vel = base_lin_vel[stable_start:]
        dof_vel = dof_vel[stable_start:]
        base_ang_vel = base_ang_vel[stable_start:]

    return InterpolatedMotion(
        base_pos=base_pos_o,
        base_quat_wxyz=base_quat_o,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        dof_pos=dof_pos_o,
        dof_vel=dof_vel,
        fps=float(output_fps),
    )


def _resolve_segments(split_points: np.ndarray) -> List[Tuple[int, int]]:
    """Turn Booster's ``split_points`` into half-open ``[start, stop)`` slices.

    The dataset description calls ``split_points`` "start and end indices for
    trajectory segmentation". Observed cases so far have exactly two values
    for a single clip (``[start_inclusive, end_inclusive]``). For longer
    arrays we conservatively treat consecutive entries as segment boundaries
    — this is the only interpretation that is consistent with a single-clip
    file also being a valid degenerate multi-segment file.
    """
    sp = np.asarray(split_points, dtype=np.int64).ravel()
    if sp.shape[0] < 2:
        raise ValueError(
            f"split_points must have at least 2 entries; got shape {sp.shape}."
        )
    if sp.shape[0] == 2:
        return [(int(sp[0]), int(sp[1]) + 1)]
    return [(int(sp[i]), int(sp[i + 1])) for i in range(sp.shape[0] - 1)]


def _segment_output_path(output_file: str, index: int, n_segments: int) -> str:
    """Insert ``_seg{i}`` before ``.npz`` when the input has multiple segments."""
    if n_segments == 1:
        return output_file
    if output_file.endswith(".npz"):
        return f"{output_file[:-4]}_seg{index}.npz"
    return f"{output_file}_seg{index}.npz"


def main(
    input_file: str,
    output_file: str,
    mjcf_path: str = _DEFAULT_MJCF,
    output_fps: float = 50.0,
    expected_joint_dim: "int | None" = 23,
) -> None:
    """Convert a Booster Robotics NPZ into a JaxRLWorld MotionCommand NPZ.

    Args:
        input_file: Path to the Booster NPZ (from the HF
            ``SaiResearch/booster_dataset`` repo).
        output_file: Target NPZ path. When the input has more than one
            segment, ``_seg{N}`` is inserted before ``.npz``.
        mjcf_path: MJCF whose body/joint layout matches the simulator
            backends' robot models. Defaults to the menagerie T1 XML.
        output_fps: Target control frequency in Hz. 50 Hz matches the
            ``t1_tracking`` preset (control dt = 0.02s).
        expected_joint_dim: Guard against accidentally feeding a K1 (or
            ``booster_lower_t1``) NPZ into the T1 pipeline. Defaults to
            23 — the T1 hinge count. Pass any other int to retarget, or
            ``None`` to disable the check.
    """
    print(f"[booster_to_npz] Loading {input_file!r}")
    data = np.load(input_file, allow_pickle=True)

    for required in ("qpos", "joint_names", "frequency", "split_points"):
        if required not in data.files:
            raise ValueError(
                f"Booster NPZ {input_file!r} is missing required field "
                f"{required!r}. Available: {list(data.files)}"
            )

    qpos = np.asarray(data["qpos"], dtype=np.float64)  # (T, 7 + J)
    joint_names_all = [str(n) for n in data["joint_names"].tolist()]
    input_fps = float(np.asarray(data["frequency"]).item())
    split_points = np.asarray(data["split_points"], dtype=np.int64)

    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError(
            f"qpos has unexpected shape {qpos.shape}; expected (T, 7+J) with J>=1."
        )

    hinge_names = [n for n in joint_names_all if n != "root"]
    n_dof = qpos.shape[1] - 7
    if len(hinge_names) != n_dof:
        raise ValueError(
            f"Booster NPZ joint schema mismatch: qpos has {n_dof} DoF columns "
            f"but joint_names (excluding 'root') has {len(hinge_names)} entries."
        )
    if expected_joint_dim is not None and n_dof != expected_joint_dim:
        raise ValueError(
            f"Expected {expected_joint_dim} DoFs for target robot but NPZ "
            f"exposes {n_dof}. If this is intentional (e.g. lower-body-only "
            f"clip, or a different humanoid), pass --expected-joint-dim={n_dof} "
            f"or disable the check."
        )
    print(
        f"[booster_to_npz] T={qpos.shape[0]}  J={n_dof}  input_fps={input_fps}  "
        f"output_fps={output_fps}"
    )

    base_pos_in = qpos[:, :3]
    base_quat_in = qpos[:, 3:7]  # MuJoCo free-joint convention: wxyz
    dof_pos_in = qpos[:, 7:]

    segments_raw = _resolve_segments(split_points)
    segments = [(s, e) for (s, e) in segments_raw if (e - s) >= 2]
    dropped = len(segments_raw) - len(segments)
    if dropped:
        print(f"[booster_to_npz] Dropped {dropped} segment(s) with < 2 frames.")
    n_segments = len(segments)
    if n_segments == 0:
        raise ValueError(
            f"No usable segments in {input_file!r}: split_points={split_points.tolist()}"
        )
    print(f"[booster_to_npz] Processing {n_segments} segment(s).")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i, (start, stop) in enumerate(segments):
        n_in = stop - start
        print(
            f"[booster_to_npz] seg {i}: input frames [{start},{stop}) "
            f"({n_in} frames, {n_in / input_fps:.3f}s) "
            f"-> resample to {output_fps} Hz"
        )
        motion = _resample_pose_trajectory(
            base_pos=base_pos_in[start:stop],
            base_quat_wxyz=base_quat_in[start:stop],
            dof_pos=dof_pos_in[start:stop],
            input_fps=input_fps,
            output_fps=output_fps,
        )
        n_out = motion.dof_pos.shape[0]
        print(
            f"[booster_to_npz]   resampled T={n_out} "
            f"({n_out / output_fps:.3f}s)"
        )

        print(f"[booster_to_npz]   FK replay via {mjcf_path!r}")
        baked = replay_motion(
            mjcf_path=mjcf_path,
            motion=motion,
            joint_names=hinge_names,
        )
        seg_out = _segment_output_path(output_file, i, n_segments)
        print(f"[booster_to_npz]   writing {seg_out!r}")
        np.savez(seg_out, **baked)

    print("[booster_to_npz] Done.")


if __name__ == "__main__":
    tyro.cli(main)
