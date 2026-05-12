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
