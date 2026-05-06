"""CSV motion loader: parse → interpolate → finite-difference velocities.

Pure numpy. No mujoco / torch dependencies — the downstream replayer
(``mujoco_replayer.py``) consumes these arrays to compute per-body
forward kinematics.

Input CSV format (Mjlab convention):
    ``[base_x, base_y, base_z, rot_x, rot_y, rot_z, rot_w, dof_1, ..., dof_N]``
with ``N`` = number of actuated joints. Base quaternion is xyzw;
internally converted to wxyz.

This module replicates Mjlab's ``MotionLoader`` in
``Mjlab/src/mjlab/scripts/csv_to_npz.py`` byte-for-byte, minus the
mjlab / torch / sim dependencies:
    - position interpolation: LERP
    - rotation interpolation: SLERP (scalar-first wxyz)
    - linear / joint velocities: central finite difference
    - angular velocity: axis-angle of relative quaternion / (2·dt)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterpolatedMotion:
    """Container for interpolated per-frame motion state.

    Arrays are in the preprocessing frame's output FPS, with ``N``
    output frames and ``J`` actuated joints.
    """

    base_pos: np.ndarray  # (N, 3)
    base_quat_wxyz: np.ndarray  # (N, 4)
    base_lin_vel: np.ndarray  # (N, 3) world frame
    base_ang_vel: np.ndarray  # (N, 3) world frame
    dof_pos: np.ndarray  # (N, J)
    dof_vel: np.ndarray  # (N, J)
    fps: float


def _lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return a * (1.0 - blend) + b * blend


def _quat_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1, keepdims=True)


def _slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternion arrays.

    ``a`` and ``b`` shape ``(N, 4)``; ``blend`` shape ``(N, 1)``. Handles
    sign-flips when the dot product is negative (shortest-path slerp).
    """
    # Flip b where dot(a, b) < 0 for shortest-path interpolation.
    dot = _quat_dot(a, b)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)

    # For near-parallel quats, fall back to linear interpolation and
    # renormalize (avoids divide-by-zero).
    near_one = dot > (1.0 - eps)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    with np.errstate(invalid="ignore", divide="ignore"):
        w_a = np.where(sin_theta > eps, np.sin((1.0 - blend) * theta) / sin_theta, 1.0 - blend)
        w_b = np.where(sin_theta > eps, np.sin(blend * theta) / sin_theta, blend)

    out = w_a * a + w_b * b
    # Use pure lerp for near-identical quats.
    lerp = _lerp(a, b, blend)
    out = np.where(near_one, lerp, out)
    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)


def _axis_angle_from_quat_wxyz(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Quaternion (wxyz) → axis-angle vector (radians · axis direction)."""
    # Flip sign so w >= 0 (shortest rotation).
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = np.linalg.norm(q[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / np.where(angle != 0.0, angle, 1.0),
        0.5 - angle * angle / 48.0,
    )
    return q[..., 1:4] / sin_half_over_angle[..., None]


def _so3_derivative(rotations_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a sequence of wxyz quaternions.

    Uses central differences: ``omega[t] = axis_angle(q[t+1] · q[t-1]⁻¹) / (2·dt)``.
    End-points are repeated so the output has the same length as input.
    """
    q_prev = rotations_wxyz[:-2]
    q_next = rotations_wxyz[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
    omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt)
    # Replicate first / last so shape matches input length.
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


class CsvMotionLoader:
    """Load + interpolate + compute velocities for a retargeted motion CSV.

    Args:
        motion_file: Path to CSV with columns
            ``[base_x, base_y, base_z, rot_x, rot_y, rot_z, rot_w, dof_1, ..., dof_N]``.
            Quaternion is xyzw; converted to wxyz internally.
        input_fps: Frame rate of the CSV.
        output_fps: Target frame rate (typically 50 Hz, matching control dt).
        line_range: Optional ``(start_line, end_line)`` (1-indexed, inclusive)
            to extract a sub-clip without re-editing the CSV.
    """

    def __init__(
        self,
        motion_file: str,
        input_fps: float,
        output_fps: float,
        line_range: tuple[int, int] | None = None,
    ) -> None:
        self.motion_file = motion_file
        self.input_fps = float(input_fps)
        self.output_fps = float(output_fps)
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.line_range = line_range

        self._load()
        self._interpolate()
        self._compute_velocities()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.line_range is None:
            motion = np.loadtxt(self.motion_file, delimiter=",").astype(np.float32)
        else:
            start, end = self.line_range
            motion = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=start - 1,
                max_rows=end - start + 1,
            ).astype(np.float32)

        self._base_pos_in = motion[:, :3]
        quat_xyzw = motion[:, 3:7]
        # Convert xyzw → wxyz.
        self._base_quat_in_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        self._dof_pos_in = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    # ------------------------------------------------------------------
    def _interpolate(self) -> None:
        times = np.arange(0.0, self.duration, self.output_dt, dtype=np.float32)
        self.output_frames = times.shape[0]

        phase = times / self.duration
        idx_0 = np.floor(phase * (self.input_frames - 1)).astype(np.int64)
        idx_1 = np.minimum(idx_0 + 1, self.input_frames - 1)
        blend = (phase * (self.input_frames - 1) - idx_0)[:, None]

        self._base_pos = _lerp(self._base_pos_in[idx_0], self._base_pos_in[idx_1], blend)
        self._base_quat_wxyz = _slerp(
            self._base_quat_in_wxyz[idx_0],
            self._base_quat_in_wxyz[idx_1],
            blend,
        )
        self._dof_pos = _lerp(self._dof_pos_in[idx_0], self._dof_pos_in[idx_1], blend)

    # ------------------------------------------------------------------
    def _compute_velocities(self) -> None:
        self._base_lin_vel = np.gradient(self._base_pos, self.output_dt, axis=0)
        self._dof_vel = np.gradient(self._dof_pos, self.output_dt, axis=0)
        self._base_ang_vel = _so3_derivative(self._base_quat_wxyz, self.output_dt)

    # ------------------------------------------------------------------
    def get_all(self) -> InterpolatedMotion:
        return InterpolatedMotion(
            base_pos=self._base_pos,
            base_quat_wxyz=self._base_quat_wxyz,
            base_lin_vel=self._base_lin_vel,
            base_ang_vel=self._base_ang_vel,
            dof_pos=self._dof_pos,
            dof_vel=self._dof_vel,
            fps=self.output_fps,
        )
