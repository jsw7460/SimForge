import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from jaxrlworld.rl.envs.managers.base import BaseManager
from jaxrlworld.rl.vis.overlays import (
    Base3DOverlay,
    CommandArrowConfig,
    CommandArrowOverlay,
    OverlayState,
    TextHUDConfig,
    TextHUDOverlay,
)
from jaxrlworld.rl.vis.overlays.hud_items import (
    BaseHeightItem,
    CommandVelItem,
    DOFPositionItem,
    EpisodeInfoItem,
    FeetHeightItem,
    FeetHeightItemConfig,
)

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import GenesisEnv
    from jaxrlworld.rl.vis.overlays import TextHUDOverlay


@dataclass
class VisualizationManagerConfig:
    """Configuration for visualization manager."""

    show_viewer: bool = False
    record_video: bool = False
    video_dir: str = ""
    video_fps: int | None = None

    # Recording settings
    record_env_ids: list[int] = field(default_factory=lambda: [0])
    grid_layout: bool = True

    # 3D Overlay settings
    enable_command_arrow: bool = True
    command_arrow_radius: float = 0.02
    command_arrow_length_scale: float = 0.5
    max_arrow_length: float = 1.0

    # 2D HUD settings
    enable_text_hud: bool = True
    hud_position: str = "top_left"
    feet_names: tuple[str, ...] | None = None  # None: no feet-height row
    extra_hud_items: list = field(default_factory=list)  # Added after default items


class VisualizationManager(BaseManager):
    """
    Manages visualization, overlays, and video recording.

    Usage:
        vis_manager = VisualizationManager(env, config)

        # Option 1: Use default HUD
        vis_manager.start_recording()

        # Option 2: Customize HUD before recording
        vis_manager.text_hud.add_item(DOFPositionItem())
        vis_manager.text_hud.remove_item("feet_height")
        vis_manager.start_recording()

        # Option 3: Set completely custom HUD
        custom_hud = TextHUDOverlay()
        custom_hud.add_item(DOFPositionItem())
        vis_manager.set_text_hud(custom_hud)
    """

    def __init__(self, env: "GenesisEnv", config: VisualizationManagerConfig):
        super().__init__(env=env)
        self.config = config
        self.cameras: dict[str, Any] = {}

        # Recording state
        self._is_recording = False
        self._recorded_frames: list[np.ndarray] = []

        # 3D overlays, created on the built scene by ``build_overlays``
        self._overlays_3d: dict[str, Base3DOverlay] = {}

        # Initialize HUD immediately so users can customize before recording
        self._text_hud: TextHUDOverlay | None = None
        if self.config.enable_text_hud:
            self._init_default_hud()

    def _setup_visualization_cameras(self) -> None:
        """Setup cameras. Called after scene entities are registered."""
        scene = self.env.scene_manager.scene
        if self.config.record_video or self.config.show_viewer:
            base_pos = (1.5, 1.5, 0.5)
            cam_pos = (2.5, 2.5, 0.5)

            # A debug camera draws the markers of the 3D overlays; a plain camera draws the scene alone
            cam = scene.add_camera(
                res=(1280, 960),
                pos=tuple(cam_pos),
                lookat=tuple(base_pos),
                fov=60,
                GUI=True,
                debug=True,
            )

            self.cameras["recorder"] = cam

            robot = self.env.scene_manager.entities.get("robot")
            if robot is not None:
                cam.follow_entity(robot, fix_orientation=False)

    def build_overlays(self) -> None:
        """Create the 3D overlays. Called once the scene is built."""
        scene = self.env.scene_manager.scene
        if self.config.enable_command_arrow:
            arrow_config = CommandArrowConfig(
                arrow_radius=self.config.command_arrow_radius,
                arrow_length_scale=self.config.command_arrow_length_scale,
                max_arrow_length=self.config.max_arrow_length,
            )
            self._overlays_3d["command_arrow"] = CommandArrowOverlay(scene, arrow_config)

    def _overlay_state(self) -> OverlayState:
        """Collect what the 3D overlays draw, one row per rendered environment."""
        scene = self.env.scene_manager.scene
        rendered_envs_idx = scene.options.vis.rendered_envs_idx
        rows = slice(None) if rendered_envs_idx is None else list(rendered_envs_idx)

        state = OverlayState()
        robot = self.env.scene_manager.entities.get("robot")
        if robot is None:
            return state
        base_quat = robot.get_quat()
        state.base_pos = robot.get_pos()[rows]
        state.base_quat = base_quat[rows]
        state.actual_lin_vel = transform_by_quat(robot.get_vel(), inv_quat(base_quat))[rows]

        cmd_manager = self.env.command_manager
        columns = {col for _, term in cmd_manager.iter_terms() for col in term.column_names}
        if {"lin_vel_x", "lin_vel_y"} <= columns:
            state.cmd_lin_vel = torch.stack((cmd_manager.lin_vel_x, cmd_manager.lin_vel_y), dim=-1)[rows]
        if "ang_vel" in columns:
            state.cmd_ang_vel = cmd_manager.ang_vel[rows]
        return state

    # =========================================================================
    # Recording
    # =========================================================================

    def start_recording(self) -> None:
        """Start video recording."""
        if "recorder" not in self.cameras:
            gs.logger.warning("No recorder camera available.")
            return

        self._is_recording = True
        self._recorded_frames.clear()
        gs.logger.info("Video recording started.")

    def stop_recording(self) -> None:
        """Stop recording and save video."""
        if not self._is_recording:
            gs.logger.warning("Recording not started.")
            return

        self._is_recording = False

        if not self._recorded_frames:
            gs.logger.warning("No frames recorded.")
            return

        fps = self.config.video_fps or int(1.0 / self.env.control_dt)
        self._save_video(self.config.video_dir, fps)

        self._recorded_frames.clear()

        gs.logger.info(f"Video saved to {self.config.video_dir}")

    def advance(self) -> None:
        """Advance visualization. Called every step."""
        if "recorder" not in self.cameras:
            return

        if self._overlays_3d:
            state = self._overlay_state()
            for overlay in self._overlays_3d.values():
                overlay.update(state)

        cam = self.cameras["recorder"]
        cam.render()

        if self._is_recording:
            frame = self._capture_frame()
            if frame is not None:
                self._recorded_frames.append(frame)

    def _capture_frame(self) -> np.ndarray | None:
        """Capture frame with 2D overlays."""
        if "recorder" not in self.cameras:
            return None

        cam = self.cameras["recorder"]
        env_ids = self.config.record_env_ids

        if len(env_ids) == 1:
            rgb_arr = cam.render(rgb=True)[0]
            if rgb_arr is None:
                return None

            if self._text_hud is not None and self._text_hud.enabled:
                rgb_arr = self._text_hud.render(self.env, rgb_arr, env_ids[0])

            return rgb_arr
        else:
            frames = []
            for env_idx in env_ids:
                rgb_arr = cam.render(rgb=True)[0]
                if rgb_arr is None:
                    continue

                if self._text_hud is not None and self._text_hud.enabled:
                    rgb_arr = self._text_hud.render(self.env, rgb_arr, env_idx)

                frames.append(rgb_arr)

            if not frames:
                return None

            return self._create_grid(frames)

    # =========================================================================
    # HUD Management
    # =========================================================================

    def _init_default_hud(self) -> None:
        """Initialize text HUD with default items."""
        self._text_hud = TextHUDOverlay(
            TextHUDConfig(
                position=self.config.hud_position,
            )
        )

        self._text_hud.add_item(BaseHeightItem())
        self._text_hud.add_item(CommandVelItem())
        if self.config.feet_names is not None:
            self._text_hud.add_item(FeetHeightItem(FeetHeightItemConfig(feet_names=self.config.feet_names)))
        self._text_hud.add_item(EpisodeInfoItem())
        self._text_hud.add_item(DOFPositionItem())

        # Add extra items from config
        for item in self.config.extra_hud_items:
            self._text_hud.add_item(item)

    @property
    def text_hud(self) -> "TextHUDOverlay | None":
        """Access the TextHUDOverlay for customization."""
        return self._text_hud

    def set_text_hud(self, hud: "TextHUDOverlay") -> None:
        """Set a custom TextHUDOverlay instance."""
        self._text_hud = hud

    # =========================================================================
    # Utility
    # =========================================================================

    def _create_grid(self, frames: list[np.ndarray]) -> np.ndarray:
        """Create grid layout from frames."""
        n = len(frames)
        if n == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if n == 1:
            return frames[0]

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        h, w = frames[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame

        return grid

    def _save_video(self, filepath: str, fps: int) -> None:
        """Save recorded frames as MP4."""
        if not self._recorded_frames:
            return
        gs.tools.animate(self._recorded_frames, filepath, fps)

    # =========================================================================
    # Overlay Control
    # =========================================================================

    def enable_command_arrow(self) -> None:
        overlay = self._overlays_3d.get("command_arrow")
        if overlay is not None:
            overlay.enable()

    def disable_command_arrow(self) -> None:
        overlay = self._overlays_3d.get("command_arrow")
        if overlay is not None:
            overlay.disable()

    def enable_text_hud(self) -> None:
        if self._text_hud:
            self._text_hud.enable()

    def disable_text_hud(self) -> None:
        if self._text_hud:
            self._text_hud.disable()

    def reset(self, env_ids=None) -> None:
        """Reset visualization state."""
        pass

    def get_subscribed_events(self):
        return None
