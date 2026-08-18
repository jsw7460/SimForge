"""Live view of the images an observation group feeds the policy.

Not a render of the scene from the camera's pose — the tensor the
observation manager actually produced, drawn as it stands. The two can
disagree in every way that matters (a stale buffer, the wrong
normalisation, a camera that never moved with the wrist) and a
prettier picture drawn from the scene would hide all of it.

Depth arrives as one channel in [0, 1] where 0 is at the lens and 1 is
the far plane, so it is drawn straight to grey: dark is close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs import World

__all__ = ["ViserCameraPanel"]

_TARGET_PIXELS = 256
"""Roughly how wide a panel should be on screen. A policy image is
tiny — 32x32 is mjlab's own size — and drawn at its true size it is a
postage stamp, so it is enlarged by whole-pixel repetition. Nearest
neighbour, never interpolation: a smoothed image is no longer the one
the policy was given, and the blur would hide single-pixel artefacts."""


class ViserCameraPanel:
    """A tab holding one image per channel of every image observation group."""

    def __init__(self, server: viser.ViserServer, env: World):
        self._server = server
        self._env = env
        self._handles: dict[tuple[str, int], Any] = {}
        self._groups = self._find_image_groups()

    @property
    def is_supported(self) -> bool:
        """True when the env has an observation group shaped like an image."""
        return bool(self._groups)

    def _find_image_groups(self) -> dict[str, tuple[int, ...]]:
        """Observation groups whose per-env shape is ``(C, H, W)``."""
        shapes = self._env.obs_manager.calculate_obs_shapes()
        return {group: shape for group, shape in shapes.items() if isinstance(shape, tuple) and len(shape) == 3}

    @staticmethod
    def _scale_for(height: int, width: int) -> int:
        return max(1, _TARGET_PIXELS // max(height, width))

    @staticmethod
    def _enlarge(image: np.ndarray, scale: int) -> np.ndarray:
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    def build_ui(self, tabs: Any) -> None:
        with tabs.add_tab("Camera", icon=viser.Icon.CAMERA):
            for group, (channels, height, width) in self._groups.items():
                with self._server.gui.add_folder(f"{group}  ({channels}x{height}x{width})"):
                    scale = self._scale_for(height, width)
                    for channel in range(channels):
                        label = group if channels == 1 else f"{group} [{channel}]"
                        self._handles[(group, channel)] = self._server.gui.add_image(
                            np.zeros((height * scale, width * scale, 3), dtype=np.uint8),
                            label=label,
                            format="png",
                        )
            self._server.gui.add_markdown("Dark is near, bright is the far plane. 1.0 also means 'nothing hit'.")

    def update(self, env_idx: int) -> None:
        """Redraw from the observation manager's current output."""
        if not self._handles:
            return
        observations = self._env.obs_manager.get_observation()
        for (group, channel), handle in self._handles.items():
            plane = observations[group][env_idx, channel]
            grey = (plane.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
            rgb = np.repeat(grey[:, :, None], 3, axis=2)
            handle.image = self._enlarge(rgb, self._scale_for(*grey.shape))
