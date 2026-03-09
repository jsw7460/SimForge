from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from genesis.vis.rasterizer_context import RasterizerContext
from .overlays import (
    Base3DOverlay,
    CommandArrowOverlay,
    CommandArrowConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@dataclass
class OverlaySettings:
    """Settings for visualization overlays."""
    enable_command_arrow: bool = True
    command_arrow_config: CommandArrowConfig | None = None

    # Feet link names for height visualization
    feet_links: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # Environment indices to render overlays for
    render_env_ids: list[int] = field(default_factory=lambda: [0])


class RLWorldRasterizerContext(RasterizerContext):
    """
    Extended RasterizerContext with 3D overlay support.

    This class is injected into Genesis Scene's visualizer after build,
    replacing the default RasterizerContext. It adds:
    - Command arrow visualization
    - Extensible overlay system
    - State collection from RL environment
    """

    def __init__(self, options, overlay_settings: OverlaySettings | None = None):
        """
        Initialize context.

        Args:
            options: Genesis VisOptions
            overlay_settings: Overlay configuration
        """
        super().__init__(options)

        self._overlay_settings = overlay_settings or OverlaySettings()
        self._overlays_3d: dict[str, Base3DOverlay] = {}
        self._visualization_state: dict[str, Any] = {}

        # Set by inject_into_scene()
        self._env: "GenesisEnv | None" = None

    def set_env(self, env: "GenesisEnv") -> None:
        """Set environment reference for data access."""
        self._env = env

    def initialize_overlays(self) -> None:
        """Initialize 3D overlays. Call after context is fully built."""
        settings = self._overlay_settings

        # Command arrow overlay
        if settings.enable_command_arrow:
            config = settings.command_arrow_config or CommandArrowConfig()
            self._overlays_3d["command_arrow"] = CommandArrowOverlay(
                context=self,
                config=config
            )

    def update(self, force_render: bool = False) -> dict:
        """Update context and render 3D overlays."""
        buffer_updates = super().update(force_render)

        # Collect state from environment
        if self._env is not None:
            self._update_visualization_state()

        # Render 3D overlays
        for overlay in self._overlays_3d.values():
            if overlay.enabled:
                overlay.update(self._visualization_state)

        return buffer_updates

    def _update_visualization_state(self) -> None:
        """Collect visualization data from environment."""
        if self._env is None:
            return

        state = {}

        # Pass render_env_ids to overlays
        state["render_env_ids"] = self._overlay_settings.render_env_ids

        # Robot position
        robot = self._env.scene_manager.entities.get("robot")
        if robot is not None:
            state["base_pos"] = robot.get_pos()
            state["base_quat"] = robot.get_quat()

        # Command data
        cmd_manager = getattr(self._env, "command_manager", None)
        if cmd_manager is not None:
            if hasattr(cmd_manager, "lin_vel_x"):
                state["cmd_lin_vel_x"] = cmd_manager.lin_vel_x
            if hasattr(cmd_manager, "lin_vel_y"):
                state["cmd_lin_vel_y"] = cmd_manager.lin_vel_y
            if hasattr(cmd_manager, "ang_vel"):
                state["cmd_ang_vel"] = cmd_manager.ang_vel

        # Actual base velocity (body frame) for tracking visualization
        if robot is not None:
            from genesis.utils.geom import transform_by_quat, inv_quat
            world_vel = robot.get_vel()
            body_vel = transform_by_quat(world_vel, inv_quat(robot.get_quat()))
            state["actual_lin_vel"] = body_vel

        # Feet heights
        feet_height = self._get_feet_height()
        if feet_height is not None:
            state["feet_height"] = feet_height

        # Episode info
        term_manager = getattr(self._env, "termination_manager", None)
        if term_manager is not None:
            episode_buf = getattr(term_manager, "episode_length_buf", None)
            if episode_buf is not None:
                state["episode_step"] = int(episode_buf[0].item())

        self._visualization_state = state

    def _get_feet_height(self) -> torch.Tensor | None:
        """Get feet z-positions."""
        if self._env is None:
            return None

        robot = self._env.scene_manager.entities.get("robot")
        if robot is None:
            return None

        feet_links = list(self._overlay_settings.feet_links)

        try:
            from rlworld.rl.utils import entity_utils as eu
            links_idx, _ = eu.find_links(robot, feet_links, global_ids=False, preserve_order=True)
            feet_pos = robot.get_links_pos(links_idx_local=links_idx)
            return feet_pos[..., 2]
        except Exception:
            return None

    def get_visualization_state(self) -> dict[str, Any]:
        """Get current state for 2D overlays."""
        return self._visualization_state.copy()

    # ===== Overlay management =====

    def get_overlay(self, name: str) -> Base3DOverlay | None:
        return self._overlays_3d.get(name)

    def enable_overlay(self, name: str) -> None:
        overlay = self._overlays_3d.get(name)
        if overlay:
            overlay.enable()

    def disable_overlay(self, name: str) -> None:
        overlay = self._overlays_3d.get(name)
        if overlay:
            overlay.disable()

    def register_overlay(self, name: str, overlay: Base3DOverlay) -> None:
        """Register a custom 3D overlay."""
        self._overlays_3d[name] = overlay


def inject_into_scene(
    scene,
    env: "GenesisEnv",
    overlay_settings: OverlaySettings | None = None
) -> RLWorldRasterizerContext:
    """
    Inject RLWorldRasterizerContext into an existing Genesis Scene.

    This should be called AFTER scene.build() completes.

    Args:
        scene: Built Genesis Scene
        env: GenesisEnv instance
        overlay_settings: Overlay configuration

    Returns:
        The injected RLWorldRasterizerContext
    """
    visualizer = scene._visualizer
    old_context = visualizer._context

    # Create new context with same options
    new_context = RLWorldRasterizerContext(
        scene.vis_options,
        overlay_settings=overlay_settings
    )

    # Copy essential state from old context
    new_context._scene = old_context._scene
    new_context.scene = old_context.scene
    new_context.sim = old_context.sim
    new_context.visualizer = old_context.visualizer
    new_context.jit = old_context.jit
    new_context._t = old_context._t

    # Copy node registries
    new_context.world_frame_node = old_context.world_frame_node
    new_context.link_frame_nodes = old_context.link_frame_nodes
    new_context.frustum_nodes = old_context.frustum_nodes
    new_context.rigid_nodes = old_context.rigid_nodes
    new_context.static_nodes = old_context.static_nodes
    new_context.dynamic_nodes = old_context.dynamic_nodes
    new_context.external_nodes = old_context.external_nodes
    new_context.seg_node_map = old_context.seg_node_map
    new_context.seg_color_map = old_context.seg_color_map

    # Copy rendering state
    new_context.buffer = old_context.buffer
    new_context._external_node_buffer = old_context._external_node_buffer
    new_context.rendered_envs_idx = old_context.rendered_envs_idx

    # Copy flags
    new_context.world_frame_shown = old_context.world_frame_shown
    new_context.link_frame_shown = old_context.link_frame_shown
    new_context.camera_frustum_shown = old_context.camera_frustum_shown

    # Replace context in visualizer
    visualizer._context = new_context

    # Also update rasterizer's context reference
    if visualizer._rasterizer is not None:
        visualizer._rasterizer._context = new_context

    # Initialize overlays and set environment
    new_context.initialize_overlays()
    new_context.set_env(env)

    return new_context
