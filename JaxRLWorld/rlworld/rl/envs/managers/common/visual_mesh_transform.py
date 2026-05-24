"""Pure ``trimesh`` + ``numpy`` mesh-transform helpers, shared across
sim backends. **No mujoco import** — so Genesis can use it without
pulling mujoco into its dependency closure.

The MjModel-based extractor (mjlab + Newton path) lives in the sibling
:mod:`visual_mesh` module, which does need mujoco.
"""

from __future__ import annotations

import numpy as np
import trimesh

__all__ = ["apply_local_transform"]


def apply_local_transform(
    mesh: trimesh.Trimesh,
    local_pos: np.ndarray,
    local_quat_wxyz: np.ndarray,
) -> trimesh.Trimesh:
    """Bake ``rotate(local_quat) ∘ translate(local_pos)`` into ``mesh``'s
    vertices in place. fp64 internally because primitive-derived vertex
    tables can carry coords with large magnitudes that overflow an
    fp32 matmul; the result is cast back to fp32."""
    rot = _quat_wxyz_to_mat(np.asarray(local_quat_wxyz, dtype=np.float64))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pos = np.asarray(local_pos, dtype=np.float64)
    mesh.vertices = ((rot @ verts.T).T + pos).astype(np.float32)
    return mesh


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n > 0.0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
