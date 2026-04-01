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
