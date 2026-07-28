"""Interactive external-force tool for the Viser play viewer.

Adds a MuJoCo-viewer-style perturbation: with the tool enabled the user
drags a 3D gizmo attached to a chosen robot link, and a spring force
``F = k * (gizmo - link)`` is applied to that link every physics
substep. Releasing (or disabling the tool) removes the force.

Interaction model
==================

The tool is a TOGGLE — it never steals the default left-drag camera
orbit. While the checkbox is off nothing changes. While it is on:

- Camera tracking is turned OFF so the rendered scene offset is zero;
  the gizmo lives in true world coordinates and the spring math is a
  plain ``gizmo - link_world`` with no offset bookkeeping. Tracking is
  restored when the tool is disabled.
- The gizmo follows the selected link while the user is NOT dragging
  (zero force), so a walking robot doesn't accumulate a spring pull.
- The moment the user drags the gizmo it detaches; the spring pulls the
  link toward the gizmo until the user presses Release (or disables the
  tool), which re-attaches the gizmo and zeroes the force.

Force is applied to the single env the viewer follows
(``play_scene.env_idx``). The magnitude is capped so a large drag can't
explode the sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from rlworld.rl.envs import World

    from .play_scene import PlayScene


class ForceDragController:
    """Owns the force-drag UI, gizmo, and per-frame wrench update."""

    @staticmethod
    def is_supported(play_scene: PlayScene) -> bool:
        """True only for the bridge-backed scene (Newton / Genesis).

        The MuJoCo/mjlab play scene renders through mjlab's own viser
        scene, whose offset and gizmo frame we do not control, so the
        drag tool is not wired there (the env-side wrench injection is
        still implemented for all three backends).
        """
        scene = getattr(play_scene, "_scene", None)
        return scene is not None and hasattr(scene, "bridge") and hasattr(scene, "camera_tracking_enabled")

    def __init__(self, server: Any, env: World, play_scene: PlayScene) -> None:
        self._server = server
        self._env = env
        self._play_scene = play_scene

        self._enabled = False
        self._dragging = False
        self._programmatic = False  # guards on_update while we move the gizmo
        self._gizmo: Any | None = None
        self._spring_line: Any | None = None

        # Selectable robot links (exclude world-fixed bodies).
        scene = play_scene._scene
        groups = [g for g in scene.geometry.mesh_groups if not g.is_fixed]
        self._link_names: list[str] = [g.body_name for g in groups]
        self._link_body_id: dict[str, int] = {g.body_name: g.body_id for g in groups}
        default = scene.geometry.tracked_body_name
        if default not in self._link_body_id and self._link_names:
            default = self._link_names[0]
        self._link_name: str = default

    # ── UI ─────────────────────────────────────────────────────────

    def build_ui(self, tabs: Any) -> None:
        if not self._link_names:
            return
        with tabs.add_tab("Force"):
            self._enable_cb = self._server.gui.add_checkbox("External force (drag)", initial_value=False)
            self._link_dd = self._server.gui.add_dropdown(
                "Link", options=self._link_names, initial_value=self._link_name
            )
            self._stiffness = self._server.gui.add_slider(
                "Stiffness (N/m)", min=10.0, max=3000.0, step=10.0, initial_value=500.0
            )
            self._max_force = self._server.gui.add_slider(
                "Max force (N)", min=10.0, max=2000.0, step=10.0, initial_value=500.0
            )
            self._release_btn = self._server.gui.add_button("Release")

            @self._enable_cb.on_update
            def _(_e: Any) -> None:
                self._set_enabled(self._enable_cb.value)

            @self._link_dd.on_update
            def _(_e: Any) -> None:
                self._link_name = self._link_dd.value
                self._release()  # re-attach gizmo to the new link next tick

            @self._release_btn.on_click
            def _(_e: Any) -> None:
                self._release()

    # ── Enable / release ───────────────────────────────────────────

    def _set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        scene = self._play_scene._scene
        if enabled:
            # Freeze the scene offset to zero so gizmo == world frame.
            scene.camera_tracking_enabled = False
            self._ensure_gizmo()
        else:
            self._release()
            scene.camera_tracking_enabled = True
            if self._gizmo is not None:
                self._gizmo.remove()
                self._gizmo = None
            if self._spring_line is not None:
                self._spring_line.remove()
                self._spring_line = None

    def _release(self) -> None:
        """Stop applying force and re-attach the gizmo to the link."""
        self._dragging = False
        self._env.clear_external_wrench()
        if self._spring_line is not None:
            self._spring_line.visible = False

    def _ensure_gizmo(self) -> None:
        if self._gizmo is not None:
            return
        pos = self._link_world_pos()
        self._gizmo = self._server.scene.add_transform_controls(
            "/force_drag_gizmo",
            scale=0.25,
            disable_rotations=True,
            disable_sliders=True,
            position=tuple(float(x) for x in pos),
        )

        @self._gizmo.on_update
        def _(_e: Any) -> None:
            if self._programmatic:
                return
            # First user drag detaches the gizmo from the link.
            self._dragging = True

    # ── Per-frame update (called under the sim lock) ───────────────

    def tick(self) -> None:
        """Recompute and push the wrench for the current viewer env.

        Must run under the viewer's sim lock: it reads a body transform
        and writes ``env.set_external_wrench``.
        """
        if not self._enabled or self._gizmo is None:
            return
        link_pos = self._link_world_pos()

        if not self._dragging:
            # Follow the link: keep the gizmo on it, apply no force.
            self._programmatic = True
            self._gizmo.position = tuple(float(x) for x in link_pos)
            self._programmatic = False
            return

        gizmo_pos = np.asarray(self._gizmo.position, dtype=np.float64)
        disp = gizmo_pos - link_pos
        force = self._stiffness.value * disp
        mag = float(np.linalg.norm(force))
        fmax = self._max_force.value
        if mag > fmax and mag > 0.0:
            force = force * (fmax / mag)

        force_t = torch.as_tensor(force, dtype=torch.float32, device=self._env.device)
        self._env.set_external_wrench(self._link_name, force_t, int(self._play_scene.env_idx))
        self._draw_spring(link_pos, gizmo_pos)

    def _link_world_pos(self) -> np.ndarray:
        """Selected link's world position for the current env.

        Uses the bridge (same source as rendering); with camera tracking
        off the scene offset is zero, so this equals the on-screen
        position.
        """
        env_idx = int(self._play_scene.env_idx)
        positions = self._play_scene._scene.bridge.get_body_positions(env_idx)
        return positions[self._link_body_id[self._link_name]].astype(np.float64)

    def _draw_spring(self, link_pos: np.ndarray, gizmo_pos: np.ndarray) -> None:
        pts = np.stack([link_pos, gizmo_pos], axis=0).astype(np.float32)[None]  # (1, 2, 3)
        if self._spring_line is None:
            self._spring_line = self._server.scene.add_line_segments(
                "/force_drag_spring",
                points=pts,
                colors=(255, 80, 80),
                line_width=3.0,
            )
        else:
            self._spring_line.points = pts
            self._spring_line.visible = True
