from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rlworld.rl.vis.rasterizer_context import RLWorldRasterizerContext


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
    Base class for 3D overlays rendered in world space.

    These overlays add meshes to the pyrender scene and are rendered
    as part of the 3D world (affected by camera movement, depth, etc.)
    """

    def __init__(self, context: "RLWorldRasterizerContext", config: Overlay3DConfig):
        super().__init__(config)
        self.context = context

    @abstractmethod
    def update(self, state: dict[str, Any]) -> None:
        """
        Update overlay with current state.

        Args:
            state: Dictionary containing visualization state data
        """
        pass

    def add_dynamic_mesh(self, mesh, pose: np.ndarray | None = None) -> None:
        """Add a mesh that will be cleared next frame."""
        if pose is not None:
            self.context.add_dynamic_node(None, mesh, pose=pose)
        else:
            self.context.add_dynamic_node(None, mesh)


class Base2DOverlay(BaseOverlay):
    """
    Base class for 2D HUD overlays.

    These overlays are drawn on top of the rendered image using OpenCV.
    """

    def __init__(self, config: Overlay2DConfig):
        super().__init__(config)
        self._cached_data: dict[str, Any] = {}

    @abstractmethod
    def update(self, state: dict[str, Any]) -> None:
        """Update cached state data."""
        pass

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
