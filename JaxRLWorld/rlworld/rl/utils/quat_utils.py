"""Unified quaternion operations — pure torch, no simulator dependencies.

All quaternion functions use **wxyz** convention unless otherwise noted.
"""
from __future__ import annotations

import torch
from torch import Tensor


def quat_rotate_inverse_wxyz(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector by the inverse of a quaternion (wxyz convention).

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).
        v: Vector to rotate, shape (..., 3).

    Returns:
        Rotated vector, shape (..., 3).
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0

    return a - b + c


def quat_rotate_wxyz(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector by a quaternion (wxyz convention).

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).
        v: Vector to rotate, shape (..., 3).

    Returns:
        Rotated vector, shape (..., 3).
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0

    return a + b + c


def quat_from_angle_axis_wxyz(angle: Tensor, axis: Tensor) -> Tensor:
    """Create quaternion from angle-axis representation (wxyz convention).

    Args:
        angle: Rotation angles, shape (...,).
        axis: Unit rotation axis, shape (3,).

    Returns:
        Quaternion in (w, x, y, z) format, shape (..., 4).
    """
    half = angle * 0.5
    sin_half = torch.sin(half)
    w = torch.cos(half)
    xyz = axis * sin_half.unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def quat_mul_wxyz(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply two quaternions (wxyz convention).

    Args:
        q1, q2: Quaternions in (w, x, y, z) format, shape (..., 4).

    Returns:
        Product quaternion, shape (..., 4).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_apply_yaw_wxyz(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector by yaw-only component of a quaternion (wxyz convention).

    Zeros out roll/pitch components (x, y in wxyz), keeping only yaw (z).

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).
        v: Vector to rotate, shape (..., 3).

    Returns:
        Rotated vector, shape (..., 3).
    """
    q_yaw = q.clone()
    q_yaw[..., 1:3] = 0.0  # zero out x, y (roll/pitch)
    q_yaw = q_yaw / torch.norm(q_yaw, dim=-1, keepdim=True)
    return quat_rotate_wxyz(q_yaw, v)


def quat_conjugate_wxyz(q: Tensor) -> Tensor:
    """Quaternion conjugate (wxyz convention).

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).

    Returns:
        Conjugate quaternion, shape (..., 4).
    """
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def xyzw_to_wxyz(q: Tensor) -> Tensor:
    """Convert quaternion from xyzw to wxyz convention.

    Args:
        q: Quaternion in (x, y, z, w) format, shape (..., 4).

    Returns:
        Quaternion in (w, x, y, z) format, shape (..., 4).
    """
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: Tensor) -> Tensor:
    """Convert quaternion from wxyz to xyzw convention.

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).

    Returns:
        Quaternion in (x, y, z, w) format, shape (..., 4).
    """
    return q[..., [1, 2, 3, 0]]


def quat_to_euler_wxyz(q: Tensor) -> Tensor:
    """Convert wxyz quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion in (w, x, y, z) format, shape (..., 4).

    Returns:
        Euler angles (roll, pitch, yaw) in radians, shape (..., 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_from_euler_xyz_wxyz(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """Convert XYZ-convention Euler angles to quaternion (wxyz convention).

    Args:
        roll: Rotation around x-axis (radians). Shape (...,).
        pitch: Rotation around y-axis (radians). Shape (...,).
        yaw: Rotation around z-axis (radians). Shape (...,).

    Returns:
        Quaternion in (w, x, y, z) format. Shape (..., 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)


def quat_inv_wxyz(q: Tensor, eps: float = 1e-9) -> Tensor:
    """Inverse of a quaternion (wxyz). Equivalent to conjugate / |q|².

    Args:
        q: Quaternion in (w, x, y, z) format. Shape (..., 4).
        eps: Small value to prevent division by zero.

    Returns:
        Inverse quaternion. Shape (..., 4).
    """
    return quat_conjugate_wxyz(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def yaw_quat_wxyz(q: Tensor) -> Tensor:
    """Extract the yaw-only component of a quaternion as a quaternion.

    Returns a quaternion representing rotation around the z-axis only.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape (..., 4).

    Returns:
        Yaw-only quaternion. Shape (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(yaw * 0.5)
    out[:, 3] = torch.sin(yaw * 0.5)
    out = out / torch.norm(out, dim=-1, keepdim=True)
    return out.view(shape)


def matrix_from_quat_wxyz(q: Tensor) -> Tensor:
    """Convert quaternion to 3x3 rotation matrix (wxyz convention).

    Args:
        q: Quaternion in (w, x, y, z) format. Shape (..., 4).

    Returns:
        Rotation matrix. Shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def axis_angle_from_quat_wxyz(q: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert quaternion to axis-angle representation (wxyz convention).

    The returned vector's magnitude is the rotation angle (radians) around
    the vector's direction (right-hand rule).

    Args:
        q: Quaternion in (w, x, y, z) format. Shape (..., 4).
        eps: Threshold below which a Taylor approximation is used.

    Returns:
        Axis-angle vector. Shape (..., 3).
    """
    # Flip sign so that w >= 0 (shortest-path rotation).
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = torch.linalg.norm(q[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(
        angle.abs() > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    return q[..., 1:4] / sin_half_over_angle.unsqueeze(-1)


def quat_error_magnitude_wxyz(q1: Tensor, q2: Tensor) -> Tensor:
    """Shortest-path angular error between two quaternions (wxyz, radians).

    Args:
        q1: First quaternion in (w, x, y, z). Shape (..., 4).
        q2: Second quaternion in (w, x, y, z). Shape (..., 4).

    Returns:
        Angular error magnitude in radians. Shape (...,).
    """
    q_diff = quat_mul_wxyz(q1, quat_conjugate_wxyz(q2))
    return torch.linalg.norm(axis_angle_from_quat_wxyz(q_diff), dim=-1)


def subtract_frame_transforms_wxyz(
    t01: Tensor,
    q01: Tensor,
    t02: Tensor,
    q02: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute pose of frame 2 relative to frame 1 in frame 0's coordinates.

    Given ``(t01, q01)`` = pose of frame 1 in frame 0 and ``(t02, q02)`` =
    pose of frame 2 in frame 0, returns ``(t12, q12)`` = pose of frame 2
    expressed in frame 1's local frame.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape (..., 3).
        q01: Quaternion of frame 1 w.r.t. frame 0 (wxyz). Shape (..., 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape (..., 3).
        q02: Quaternion of frame 2 w.r.t. frame 0 (wxyz). Shape (..., 4).

    Returns:
        Tuple ``(t12, q12)``: position (``..., 3``) and quaternion
        (``..., 4``) of frame 2 expressed in frame 1.
    """
    q10 = quat_inv_wxyz(q01)
    q12 = quat_mul_wxyz(q10, q02)
    t12 = quat_rotate_wxyz(q10, t02 - t01)
    return t12, q12
