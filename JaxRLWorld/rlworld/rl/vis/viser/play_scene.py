"""PlayScene abstraction for ViserPlayViewer.

Decouples the play viewer from any specific scene implementation.
Two backends:

- BridgePlayScene: Newton/Genesis via SimulatorBridge + ViserScene.
- MujocoPlayScene: MuJoCo via mjlab's ViserMujocoScene (batched GLB + LOD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.mujoco.scene import MujocoSceneManager

    from .bridge import SimulatorBridge


@dataclass
class TrackedBodyData:
    """Data needed for command/velocity arrows."""

    position: np.ndarray  # (3,) world position
    yaw: float  # heading angle in radians
    scene_offset: np.ndarray  # (3,) offset applied by camera tracking
    body_velocity: np.ndarray | None  # (2,) body-frame [vx, vy] or None


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class PlayScene(Protocol):
    """Minimal interface that ViserPlayViewer needs from a scene."""

    @property
    def env_idx(self) -> int: ...

    @property
    def needs_update(self) -> bool: ...

    @needs_update.setter
    def needs_update(self, value: bool) -> None: ...

    def create(self, server: viser.ViserServer) -> None: ...
    def update(self) -> None: ...
    def setup_gui(self, tabs: Any) -> None: ...
    def set_on_env_switch(self, callback: Any) -> None: ...
    def clear_debug(self) -> None: ...
    def cleanup(self) -> None: ...
    def get_tracked_body_data(self) -> TrackedBodyData | None: ...


# ── BridgePlayScene (Newton / Genesis) ─────────────────────────


class BridgePlayScene:
    """PlayScene backed by SimulatorBridge + ViserScene."""

    def __init__(self, bridge: SimulatorBridge):
        self._bridge = bridge
        self._scene = None

    def create(self, server: viser.ViserServer) -> None:
        from .scene import ViserScene

        self._scene = ViserScene.create(server, self._bridge)

    @property
    def env_idx(self) -> int:
        return self._scene.env_idx

    @property
    def needs_update(self) -> bool:
        return self._scene.needs_update

    @needs_update.setter
    def needs_update(self, value: bool) -> None:
        self._scene.needs_update = value

    def update(self) -> None:
        self._scene.update()

    def setup_gui(self, tabs: Any) -> None:
        self._scene.create_gui(tabs)

    def set_on_env_switch(self, callback: Any) -> None:
        self._scene.set_on_env_switch(callback)

    def clear_debug(self) -> None:
        self._scene.clear_debug()

    def cleanup(self) -> None:
        self._scene.cleanup()

    def get_tracked_body_data(self) -> TrackedBodyData | None:
        tracked_id = self._scene.geometry.tracked_body_id
        if tracked_id is None:
            return None
        env_idx = self._scene.env_idx
        pos = self._bridge.get_tracked_position(env_idx)
        quats = self._bridge.get_body_quaternions(env_idx)
        yaw = _yaw_from_wxyz(quats[tracked_id])
        vel = self._bridge.get_body_velocity(env_idx)
        return TrackedBodyData(
            position=pos,
            yaw=yaw,
            scene_offset=self._scene._scene_offset,
            body_velocity=vel,
        )


# ── MujocoPlayScene (MuJoCo) ───────────────────────────────────


class MujocoPlayScene:
    """PlayScene backed by mjlab's ViserMujocoScene (batched GLB + LOD)."""

    def __init__(self, scene_manager: MujocoSceneManager):
        self._scene_manager = scene_manager
        self._mj_scene = None

    def create(self, server: viser.ViserServer) -> None:
        from mjlab.viewer.viser.scene import MjlabViserScene

        mj_model = self._scene_manager.mj_model
        num_envs = self._scene_manager.scene.num_envs
        self._mj_scene = MjlabViserScene(server, mj_model, num_envs)

    @property
    def env_idx(self) -> int:
        return self._mj_scene.env_idx

    @property
    def needs_update(self) -> bool:
        return self._mj_scene.needs_update

    @needs_update.setter
    def needs_update(self, value: bool) -> None:
        self._mj_scene.needs_update = value

    def update(self) -> None:
        wp_data = self._scene_manager.data
        self._mj_scene.update(wp_data)

    def setup_gui(self, tabs: Any) -> None:
        with tabs.add_tab("Scene", icon=viser.Icon.EYE):
            self._mj_scene.create_visualization_gui()

    def set_on_env_switch(self, callback: Any) -> None:
        pass

    def clear_debug(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def get_tracked_body_data(self) -> TrackedBodyData | None:
        tracked_id = self._mj_scene._tracked_body_id
        if tracked_id is None:
            return None
        env_idx = self._mj_scene.env_idx
        data = self._scene_manager.data
        xpos = data.xpos.cpu().numpy()  # (num_envs, nbody, 3)
        xquat = data.xquat.cpu().numpy()  # (num_envs, nbody, 4) wxyz
        pos = xpos[env_idx, tracked_id]
        yaw = _yaw_from_wxyz(xquat[env_idx, tracked_id])
        scene_offset = self._mj_scene._scene_offset if hasattr(self._mj_scene, "_scene_offset") else np.zeros(3)
        # Body velocity from cvel if available.
        vel = None
        if hasattr(data, "cvel"):
            try:
                cvel = data.cvel.cpu().numpy()
                vel_6d = cvel[env_idx, tracked_id]  # (6,) [wx,wy,wz,vx,vy,vz]
                world_vel = vel_6d[3:6]
                w, x, y, z = xquat[env_idx, tracked_id]
                from scipy.spatial.transform import Rotation

                body_vel = Rotation.from_quat([x, y, z, w]).inv().apply(world_vel)
                vel = body_vel[:2].astype(np.float32)
            except Exception:
                pass
        return TrackedBodyData(
            position=pos.copy(),
            yaw=yaw,
            scene_offset=scene_offset,
            body_velocity=vel,
        )
