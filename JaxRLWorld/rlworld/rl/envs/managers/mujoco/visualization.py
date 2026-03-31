# rlworld/rl/envs/managers/mujoco/visualization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import MujocoEnv


@dataclass
class MujocoVisualizationManagerConfig:
    """Config for MujocoVisualizationManager."""
    show_viewer: bool = False
    viewer_type: Literal["viser"] = "viser"
    viser_port: int = 8080
    camera_tracking: bool = True
    camera_distance: float = 3.0
    camera_azimuth: float = 45.0
    camera_elevation: float = 30.0


class MujocoVisualizationManager(BaseManager):
    """Manages Viser visualization for MujocoEnv using ViserMujocoScene."""

    def __init__(self, env: "MujocoEnv", config: MujocoVisualizationManagerConfig):
        super().__init__(env=env)
        self.config = config
        self._scene = None
        self._server = None

    def setup(self) -> None:
        """Setup viewer after scene is built."""
        if self.config.viewer_type != "viser":
            return

        import viser
        from mjlab.viewer.viser.scene import ViserMujocoScene

        mj_model = self.env.scene_manager.mj_model
        num_envs = self.env.num_envs

        self._server = viser.ViserServer(
            port=self.config.viser_port,
            label="rlworld-mjlab",
        )
        self._server.request_share_url()

        self._scene = ViserMujocoScene.create(
            server=self._server,
            mj_model=mj_model,
            num_envs=num_envs,
        )

        self._scene.camera_tracking_enabled = self.config.camera_tracking

        # Add GUI controls
        self._scene.create_visualization_gui(
            camera_distance=self.config.camera_distance,
            camera_azimuth=self.config.camera_azimuth,
            camera_elevation=self.config.camera_elevation,
        )

        print(f"[INFO] Viser viewer running at: http://localhost:{self.config.viser_port}")

    def advance(self) -> None:
        """Update visualization with current sim state."""
        if self._scene is None:
            return

        wp_data = self.env.scene_manager.data
        self._scene.update(wp_data)

    def close(self) -> None:
        """Close viewer."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def reset(self, env_ids=None) -> None:
        """Reset visualization state."""
        pass