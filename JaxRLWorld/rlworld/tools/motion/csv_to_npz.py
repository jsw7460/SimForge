"""CLI: convert a retargeted-motion CSV into an NPZ consumable by MotionCommand.

Usage::

    uv run python -m rlworld.tools.motion.csv_to_npz \\
        --input-file /path/to/motion.csv \\
        --mjcf-path  /path/to/robot.xml \\
        --output-file /tmp/motion.npz \\
        --input-fps 30 --output-fps 50

By default every 1-DoF non-free joint in the MJCF is used (in XML order)
and the CSV must have exactly that many DoF columns. Use ``--joint-names``
to override the mapping if your CSV produces a different joint order.

The resulting NPZ contains per-frame per-body world state +
``body_names`` / ``joint_names`` metadata so that the runtime
MotionLoader can reorder bodies to match the preset's ``body_names``
list. Uses only the ``mujoco`` Python package; no mjlab dependency.
"""

from __future__ import annotations

import numpy as np
import tyro

from rlworld.tools.motion.motion_loader import CsvMotionLoader
from rlworld.tools.motion.mujoco_replayer import replay_motion


def main(
    input_file: str,
    mjcf_path: str,
    output_file: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    joint_names: tuple[str, ...] | None = None,
    line_range: tuple[int, int] | None = None,
) -> None:
    """Convert a retargeted motion CSV + MJCF into a MotionCommand-ready NPZ.

    Args:
        input_file: Path to the CSV. Columns:
            ``[base_x, base_y, base_z, rot_x, rot_y, rot_z, rot_w,
               dof_1, ..., dof_N]`` (quaternion is xyzw).
        mjcf_path: MuJoCo XML for the robot (with a free joint).
        output_file: Output NPZ path.
        input_fps: Frame rate of the CSV.
        output_fps: Target frame rate (typically matches ``control_dt``,
            e.g. 50 Hz → 1 / 0.02).
        joint_names: Ordered list of actuated joint names, matching the
            CSV's DoF column order. If omitted, the MJCF's non-free
            1-DoF joints in XML order are used.
        line_range: Optional ``(start, end)`` 1-indexed inclusive line
            range to extract a sub-clip without touching the CSV.
    """
    print(f"[csv_to_npz] Loading + interpolating {input_file!r}")
    motion = CsvMotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        line_range=line_range,
    ).get_all()
    print(f"[csv_to_npz] {motion.dof_pos.shape[0]} output frames @ {output_fps} Hz ({motion.dof_pos.shape[1]} DoFs).")

    print(f"[csv_to_npz] Forward-kinematics replay via {mjcf_path!r}")
    baked = replay_motion(
        mjcf_path=mjcf_path,
        motion=motion,
        joint_names=list(joint_names) if joint_names is not None else None,
    )

    print(f"[csv_to_npz] Writing {output_file!r}")
    np.savez(output_file, **baked)
    print("[csv_to_npz] Done.")


if __name__ == "__main__":
    tyro.cli(main)
