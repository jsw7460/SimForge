"""Simulator bridge — the only simulator-specific abstraction.

Each simulator implements :class:`SimulatorBridge` to provide:

1. One-time geometry extraction (meshes for scene setup).
2. Per-frame body transforms and tracked-body velocity.

The base class caches the per-frame fetches so that multiple ``get_*``
calls within one visualization tick (scene update + command arrows +
tracked-body queries) issue exactly one GPU→CPU sync per data kind,
regardless of how many helpers read the same data.  Viewers must call
:meth:`SimulatorBridge.begin_frame` at the start of each visualization
tick.

The world→body velocity rotation lives in the base class and reuses
the cached quaternion, so subclasses never reach back into the
simulator for the same data twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


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
    """A group of visual meshes attached to one body/link.

    Mesh vertices are assumed to be in the body's local frame —
    bridges bake any per-mesh shape transform into the vertices at
    :meth:`SimulatorBridge.extract_geometry` time so the per-frame
    scene update only needs the body's own pose.  See the
    ``_newton_mesh_to_trimesh`` / Genesis ``init_pos+init_quat`` /
    MuJoCo ``geom_pos+geom_quat`` paths in the per-sim bridges.
    """

    body_id: int
    body_name: str
    is_fixed: bool  # True for world-fixed bodies (ground plane, static objects)
    meshes: list[trimesh.Trimesh] = field(default_factory=list)


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


class SimulatorBridge(ABC):
    """Abstract base for simulator-specific data extraction.

    Subclasses implement the minimal sim-specific surface:

    * :attr:`num_envs` — number of parallel environments.
    * :attr:`tracked_body_id` — body index the camera follows.
    * :meth:`extract_geometry` — one-time geometry extraction.
    * :meth:`_fetch_body_transforms` — single-sync fetch of
      ``(positions, quaternions)`` for one env.
    * :meth:`_fetch_tracked_world_velocity` — single-sync fetch of the
      tracked body's world-frame linear velocity.

    Everything else (public ``get_*`` methods, world→body rotation,
    per-frame caching) lives here so each backend stays narrow and the
    hot path collapses to one sync per data kind per frame.
    """

    def __init__(self) -> None:
        # Per-frame cache.  ``_cache_env_idx`` doubles as the cache-validity
        # token: a fetch overwrites it with the env it just read, and
        # ``begin_frame`` resets it to ``None`` to invalidate.
        self._cache_env_idx: int | None = None
        self._cached_pos: np.ndarray | None = None
        self._cached_quat: np.ndarray | None = None
        self._cached_world_vel: np.ndarray | None = None
        self._world_vel_fetched: bool = False

    # ── Viewer hooks ────────────────────────────────────────────────

    def begin_frame(self) -> None:
        """Invalidate the per-frame cache.

        Viewers call this at the top of each visualization tick.
        Subsequent ``get_*`` calls re-fetch from the simulator the
        first time they are needed and then reuse the result for the
        rest of the frame.
        """
        self._cache_env_idx = None
        self._cached_pos = None
        self._cached_quat = None
        self._cached_world_vel = None
        self._world_vel_fetched = False

    # ── Required sim-specific surface ───────────────────────────────

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""

    @property
    @abstractmethod
    def tracked_body_id(self) -> int | None:
        """Body index the camera follows (the floating base on
        articulated robots).  ``None`` when no such body exists.
        """

    @abstractmethod
    def extract_geometry(self) -> SimulatorGeometry:
        """Extract all visual geometry — called once at setup."""

    @abstractmethod
    def _fetch_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read body transforms for one env from the simulator.

        Returns ``(positions, quaternions)`` of shapes
        ``(num_bodies, 3)`` / ``(num_bodies, 4)`` (wxyz).
        Implementations should issue at most one GPU→CPU sync.  The
        base class calls this at most once per ``(env_idx, frame)``.
        """

    @abstractmethod
    def _fetch_tracked_world_velocity(self, env_idx: int) -> np.ndarray | None:
        """World-frame linear velocity at the tracked body.

        Returns ``(3,)`` or ``None`` when the simulator can't supply
        it.  Body-frame conversion is handled by the base class using
        the cached quaternion.
        """

    # ── Public, cached read API ─────────────────────────────────────

    def get_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Body poses for one env, cached per frame.

        Returns ``(positions, quaternions)``.  Repeated calls within
        the same frame return the cached buffers without re-fetching;
        callers must not retain the returned arrays across frames
        (subclasses may reuse internal buffers).
        """
        if self._cache_env_idx != env_idx or self._cached_pos is None:
            self._cached_pos, self._cached_quat = self._fetch_body_transforms(env_idx)
            self._cache_env_idx = env_idx
            # An env switch must also invalidate the velocity cache.
            self._cached_world_vel = None
            self._world_vel_fetched = False
        return self._cached_pos, self._cached_quat

    def get_body_positions(self, env_idx: int) -> np.ndarray:
        """Body positions for one env. Returns ``(num_bodies, 3)``."""
        return self.get_body_transforms(env_idx)[0]

    def get_body_quaternions(self, env_idx: int) -> np.ndarray:
        """Body orientations for one env. Returns ``(num_bodies, 4)`` wxyz."""
        return self.get_body_transforms(env_idx)[1]

    def get_tracked_position(self, env_idx: int) -> np.ndarray:
        """Position of the tracked body. Returns ``(3,)``."""
        positions = self.get_body_transforms(env_idx)[0]
        tracked = self.tracked_body_id
        return positions[tracked if tracked is not None else 0]

    def get_body_velocity(self, env_idx: int) -> np.ndarray | None:
        """Body-frame linear velocity of the tracked body.

        Returns ``(2,)`` ``[vx, vy]`` or ``None``.  The world→body
        rotation reuses the cached quaternion from
        :meth:`get_body_transforms`, so this costs at most one extra
        simulator query (the world-frame velocity) per frame.
        """
        tracked = self.tracked_body_id
        if tracked is None:
            return None
        if not self._world_vel_fetched or self._cache_env_idx != env_idx:
            self._cached_world_vel = self._fetch_tracked_world_velocity(env_idx)
            self._world_vel_fetched = True
        world_vel = self._cached_world_vel
        if world_vel is None:
            return None
        quat_wxyz = self.get_body_quaternions(env_idx)[tracked]
        w, x, y, z = quat_wxyz
        body_vel = Rotation.from_quat([x, y, z, w]).inv().apply(world_vel)
        return body_vel[:2].astype(np.float32)
