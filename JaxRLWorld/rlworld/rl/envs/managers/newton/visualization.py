from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import warp as wp

import newton
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


@dataclass
class NewtonVisualizationManagerConfig:
    """Internal config for NewtonVisualizationManager."""
    show_viewer: bool = False
    record_video: bool = False
    video_dir: str = ""
    video_fps: int = 60
    viewer_type: Literal["gl", "viser", "rerun", "usd", "file"] = "gl"
    viser_port: int = 8080
    viser_share: bool = True
    rerun_web_port: int = 9191


class NewtonVisualizationManager(BaseManager):
    """Manages visualization and video recording for Newton environments."""

    def __init__(self, env: "NewtonEnv", config: NewtonVisualizationManagerConfig):
        super().__init__(env=env)
        self.config = config
        self.viewer = None
        self.sim_time = 0.0

    def setup(self) -> None:
        """Setup viewer after scene is built."""

        model = self.env.scene_manager.model

        def get_output_path(ext: str) -> str:
            if self.config.video_dir:
                base, _ = os.path.splitext(self.config.video_dir)
                return base + ext
            return f"output{ext}"

        if self.config.show_viewer:
            if self.config.viewer_type == "viser":
                self.viewer = newton.viewer.ViewerViser(
                    port=self.config.viser_port,
                    share=self.config.viser_share,
                    record_to_viser=get_output_path(".viser") if self.config.record_video else None,
                )
            elif self.config.viewer_type == "rerun":
                self.viewer = newton.viewer.ViewerRerun(
                    web_port=self.config.rerun_web_port,
                )
            elif self.config.viewer_type == "gl":
                self.viewer = newton.viewer.ViewerGL()
            elif self.config.viewer_type == "usd":
                self.viewer = newton.viewer.ViewerUSD(
                    output_path=get_output_path(".usd"),
                    fps=self.config.video_fps,
                    up_axis="Z",
                )
            else:
                return
            self.viewer.set_model(model, max_worlds=1)

        elif self.config.record_video:
            if self.config.viewer_type == "viser":
                self.viewer = newton.viewer.ViewerViser(
                    port=self.config.viser_port,
                    share=self.config.viser_share,
                    record_to_viser=get_output_path(".viser"),
                )
            elif self.config.viewer_type == "usd":
                self.viewer = newton.viewer.ViewerUSD(
                    output_path=get_output_path(".usd"),
                    fps=self.config.video_fps,
                    up_axis="Z",
                )
            elif self.config.viewer_type == "file":
                self.viewer = newton.viewer.ViewerFile(
                    get_output_path(".bin"),
                    auto_save=False,
                )
            else:
                raise NotImplementedError(f"Unknown viewer type: {self.config.viewer_type}")
            self.viewer.set_model(model, max_worlds=1)

    def advance(self) -> None:
        if self.viewer is None:
            return

        if not isinstance(self.viewer, newton.viewer.ViewerUSD):
            if hasattr(self.viewer, 'is_running') and not self.viewer.is_running():
                return

        state = self.env.scene_manager.state_0
        base_pos = state.joint_q.numpy()[:3]

        # Camera tracking
        if isinstance(self.viewer, newton.viewer.ViewerGL) and hasattr(self.viewer, 'set_camera'):
            self.viewer.set_camera(
                pos=wp.vec3(base_pos[0] + 3.0, base_pos[1] + 3.0, base_pos[2] + 1.0),
                pitch=-20.0,
                yaw=-135.0,
            )
        elif isinstance(self.viewer, newton.viewer.ViewerViser):
            for client in self.viewer._server.get_clients().values():
                client.camera.position = (
                    base_pos[0] + 3.0,
                    base_pos[1] + 3.0,
                    base_pos[2] + 1.0,
                )
                client.camera.look_at = tuple(base_pos)

        self.sim_time += self.env.control_dt

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(state)
        self.viewer.end_frame()

    def start_recording(self) -> None:
        """Start video recording."""
        pass

    def stop_recording(self) -> None:
        """Stop recording and save."""
        if self.viewer is None:
            return
        print(f"[DEBUG] stop_recording called, viewer type: {type(self.viewer)}")

        if isinstance(self.viewer, newton.viewer.ViewerViser):
            self.viewer.save_recording()
        elif isinstance(self.viewer, newton.viewer.ViewerUSD):
            print("[DEBUG] Closing USD viewer")
            self.viewer.close()
            self.viewer = None
        elif isinstance(self.viewer, newton.viewer.ViewerFile):
            self.viewer.close()
            self.viewer = None

    def close(self) -> None:
        """Close viewer."""
        if self.viewer is not None:
            if hasattr(self.viewer, 'close'):
                self.viewer.close()
            self.viewer = None

    def reset(self, env_ids=None) -> None:
        """Reset visualization state."""
        pass
