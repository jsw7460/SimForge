"""Unified Viser-based visualization manager for SimForge.

Passive observer pattern: advance() is called from the environment's
_step_physics() method. The viewer does NOT own the simulation loop.

Works with any simulator through SimulatorBridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import viser

from .bridge import SimulatorBridge
from .scene import ViserScene
from .overlays import ViserTermOverlays, ViserDebugOverlays

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


@dataclass
class ViserViewerConfig:
    """Configuration for the Viser viewer."""

    port: int = 8080
    share: bool = True
    label: str = "SimForge"
    update_every_n_steps: int = 2  # 30Hz at 60Hz physics
    enable_reward_plots: bool = True
    enable_debug_viz: bool = False


class ViserVisualizationManager:
    """Passive Viser-based visualization manager.

    Integrates with the existing VisualizationManager pattern used by
    GenesisEnv and NewtonEnv. Call advance() every physics step.
    """

    def __init__(
        self,
        env: World,
        bridge: SimulatorBridge,
        config: ViserViewerConfig | None = None,
    ):
        self.env = env
        self.bridge = bridge
        self.config = config or ViserViewerConfig()

        self._step_counter = 0

        # Create Viser server.
        self.server = viser.ViserServer(
            port=self.config.port,
            label=self.config.label,
        )
        if self.config.share:
            self.server.request_share_url()

        # Create scene.
        self.scene = ViserScene.create(self.server, self.bridge)

        # Setup GUI.
        self._setup_gui()

        print(
            f"[ViserViewer] Started on port {self.config.port}. "
            f"Open the URL above to view."
        )

    def _setup_gui(self) -> None:
        """Create GUI tabs and overlays."""
        tabs = self.server.gui.add_tab_group()

        # Scene tab (env selector, camera controls).
        self.scene.create_gui(tabs)
        self.scene.set_on_env_switch(self._on_env_switch)

        # Reward plots tab.
        self._term_overlays: ViserTermOverlays | None = None
        if self.config.enable_reward_plots:
            self._term_overlays = ViserTermOverlays(
                server=self.server,
                env=self.env,
                scene=self.scene,
            )
            self._term_overlays.setup_tabs(tabs)

        # Debug overlays.
        self._debug_overlays: ViserDebugOverlays | None = None
        if self.config.enable_debug_viz:
            self._debug_overlays = ViserDebugOverlays(
                env=self.env,
                scene=self.scene,
            )

    def _on_env_switch(self) -> None:
        """Handle environment index change."""
        if self._term_overlays:
            self._term_overlays.on_env_switch()
        if self._debug_overlays:
            self._debug_overlays.on_env_switch()

    def setup(self) -> None:
        """Post-initialization setup (called after scene is built)."""
        pass  # Everything initialized in __init__ for now.

    def advance(self) -> None:
        """Called every physics step from the environment.

        Gates updates to reduce overhead (default: every 2 steps = 30Hz).
        """
        self._step_counter += 1
        if self._step_counter % self.config.update_every_n_steps != 0:
            return

        with self.server.atomic():
            # Update 3D scene.
            self.scene.update()

            # Update debug visualization.
            if self._debug_overlays:
                self._debug_overlays.queue()

            # Update reward plots.
            if self._term_overlays:
                self._term_overlays.update()

    def start_recording(self) -> None:
        """Start video recording (placeholder)."""
        pass

    def stop_recording(self) -> None:
        """Stop video recording (placeholder)."""
        pass

    def close(self) -> None:
        """Shut down the Viser server."""
        self.scene.cleanup()
        if self._term_overlays:
            self._term_overlays.cleanup()
        # Note: viser server doesn't have an explicit close method.

    def reset(self, env_ids=None) -> None:
        """Reset visualization state."""
        pass
