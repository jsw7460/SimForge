from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import genesis as gs


@dataclass
class OverlayConfig:
    """Base configuration for overlays."""

    enabled: bool = True


@dataclass
class Overlay3DConfig(OverlayConfig):
    """Configuration for 3D overlays rendered in world space."""

    pass


@dataclass
class Overlay2DConfig(OverlayConfig):
    """Configuration for 2D overlays rendered as HUD."""

    pass


@dataclass
class OverlayState:
    """What the 3D overlays draw from, one row per rendered environment.

    Every tensor holds the rendered environments in the order Genesis renders
    them (``VisOptions.rendered_envs_idx``), so the overlays hand the rows to
    the scene's marker API as they are. A field is None when the scene has no
    source for it (no robot, no velocity command).
    """

    base_pos: torch.Tensor | None = None
    """Robot base position, world frame, ``(n_rendered, 3)``."""
    base_quat: torch.Tensor | None = None
    """Robot base orientation, wxyz, ``(n_rendered, 4)``."""
    cmd_lin_vel: torch.Tensor | None = None
    """Commanded planar linear velocity, body frame, ``(n_rendered, 2)``."""
    cmd_ang_vel: torch.Tensor | None = None
    """Commanded yaw rate, ``(n_rendered,)``."""
    actual_lin_vel: torch.Tensor | None = None
    """Measured base linear velocity, body frame, ``(n_rendered, 3)``."""


class BaseOverlay(ABC):
    """Abstract base class for all visualization overlays."""

    def __init__(self, config: OverlayConfig):
        self.config = config
        self._enabled = config.enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled


class Base3DOverlay(BaseOverlay):
    """
    Base class for 3D overlays drawn as Genesis markers.

    The markers go through the scene's debug-draw API (``scene.draw_debug_*``),
    which the interactive viewer and the debug cameras draw
    (``scene.add_camera(debug=True)``); camera sensors and plain cameras leave
    them out. An overlay redraws its markers every frame: ``update`` clears
    the previous frame's markers first, so a disabled overlay also vanishes.
    """

    def __init__(self, scene: "gs.Scene", config: Overlay3DConfig):
        super().__init__(config)
        self.scene = scene
        self._markers: list = []

    @abstractmethod
    def update(self, state: OverlayState) -> None:
        """Redraw the overlay's markers from the current state."""
        pass

    def clear(self) -> None:
        """Remove every marker this overlay drew."""
        for marker in self._markers:
            self.scene.clear_debug_object(marker)
        self._markers.clear()

    def draw_arrows(self, pos: torch.Tensor, vec: torch.Tensor, radius: float, color: tuple[float, ...]) -> None:
        """Draw one arrow per rendered environment, from ``pos`` along ``vec``; a zero vector draws nothing."""
        marker = self.scene.draw_debug_arrow(pos, vec, radius=radius, color=color)
        if marker is not None:
            self._markers.append(marker)


class Base2DOverlay(BaseOverlay):
    """
    Base class for 2D HUD overlays.

    These overlays are drawn on top of the rendered image using OpenCV.
    """

    @abstractmethod
    def render(self, frame: np.ndarray, env_idx: int = 0) -> np.ndarray:
        """
        Render overlay on frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)
            env_idx: Environment index to render

        Returns:
            Modified frame with overlay
        """
        pass
