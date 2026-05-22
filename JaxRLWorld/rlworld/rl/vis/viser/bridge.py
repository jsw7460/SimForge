"""SimulatorBridge protocol — the only simulator-specific abstraction.

Each simulator implements this protocol to provide:
1. One-time geometry extraction (meshes for scene setup)
2. Per-frame state queries (body transforms for scene update)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import trimesh


def terrain_data_to_trimesh(data) -> trimesh.Trimesh:
    """Tessellate a canonical ``TerrainData`` height grid into a world-frame mesh.

    The grid is centred on the origin spanning ``size_xy``; ``z`` is the
    metre height directly, so the surface lines up with the physics terrain
    in every backend (the terrain group is rendered as a fixed body, so its
    vertices must already be in world coordinates).
    """
    heights = np.asarray(data.heights_m, dtype=np.float32)
    nrow, ncol = heights.shape
    lx, ly = data.size_xy
    xs = np.linspace(-lx / 2.0, lx / 2.0, nrow, dtype=np.float32)  # rows → x
    ys = np.linspace(-ly / 2.0, ly / 2.0, ncol, dtype=np.float32)  # cols → y
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([xx.ravel(), yy.ravel(), heights.ravel()], axis=1).astype(np.float32)
    rr, cc = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
    v00 = (rr * ncol + cc).ravel()
    v01 = (rr * ncol + cc + 1).ravel()
    v10 = ((rr + 1) * ncol + cc).ravel()
    v11 = ((rr + 1) * ncol + cc + 1).ravel()
    faces = np.concatenate([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)], axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@dataclass
class BodyMeshGroup:
    """A group of visual meshes attached to one body/link."""

    body_id: int
    body_name: str
    is_fixed: bool  # True for world-fixed bodies (ground plane, static objects)
    meshes: list[trimesh.Trimesh] = field(default_factory=list)

    # Local offset of each mesh relative to the body frame.
    # If None, mesh vertices are already in body-local coordinates.
    local_positions: list[np.ndarray] | None = None  # list of (3,)
    local_quaternions: list[np.ndarray] | None = None  # list of (4,) wxyz


@dataclass
class SimulatorGeometry:
    """All geometry needed to set up the Viser scene."""

    mesh_groups: list[BodyMeshGroup]
    num_bodies: int
    # Index of the body to track with the camera (typically the robot base).
    tracked_body_id: int | None = None
    tracked_body_name: str = "base"
    # True when the simulator already supplies a real, renderable
    # ground/terrain mesh (e.g. a heightfield). The viewer then skips its
    # synthetic fallback ground plane so the two don't overlap — matching
    # how MuJoCo / IsaacLab render the actual terrain geom. Stays False for
    # flat analytic ground planes (which have no mesh), so those tasks keep
    # the cosmetic ground.
    has_ground_mesh: bool = False


@runtime_checkable
class SimulatorBridge(Protocol):
    """Protocol for simulator-specific data extraction."""

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        ...

    def extract_geometry(self) -> SimulatorGeometry:
        """Extract all visual geometry from the simulator (called once at setup)."""
        ...

    def get_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Body poses for one environment in a *single* read.

        Returns ``(positions, quaternions)`` where ``positions`` is
        ``(num_bodies, 3)`` and ``quaternions`` is ``(num_bodies, 4)`` in
        wxyz order.  This is the per-frame hot path — implementations
        must do exactly one GPU→CPU transfer per simulated entity, not
        one per body.  ``ViserScene.update`` calls only this (the
        tracked-body position is sliced from the result).
        """
        ...

    def get_body_positions(self, env_idx: int) -> np.ndarray:
        """Body positions for one environment. Returns (num_bodies, 3)."""
        ...

    def get_body_quaternions(self, env_idx: int) -> np.ndarray:
        """Body orientations for one environment. Returns (num_bodies, 4) wxyz."""
        ...

    def get_tracked_position(self, env_idx: int) -> np.ndarray:
        """Position of the tracked body (for camera). Returns (3,)."""
        ...

    def get_body_velocity(self, env_idx: int) -> np.ndarray | None:
        """Body-frame linear velocity of the tracked body. Returns (2,) [vx, vy] or None."""
        ...
