"""Interactive external-force tool for the Viser play viewer.

Adds a MuJoCo-viewer-style perturbation: with the tool enabled the user
drags a 3D gizmo attached to a chosen robot link, and a spring force
``F = k * (gizmo - link)`` is applied to that link every physics
substep. Releasing (or disabling the tool) removes the force.

Interaction model
==================

The tool is a TOGGLE — it never steals the default left-drag camera
orbit. While the checkbox is off nothing changes. While it is on:

- Camera tracking stays ON so the robot remains centred on screen and
  is easy to grab even while it moves fast. Everything (gizmo, spring)
  is computed in SCENE coordinates (world + the scene's re-centring
  offset); the offset cancels in ``gizmo_scene - link_scene``, so the
  applied world-frame force needs no offset bookkeeping.
- The gizmo follows the selected link while the user is NOT dragging
  (zero force), so it sits right on the link — the user just grabs the
  handle at screen centre.
- The moment the user drags the gizmo it detaches; the spring pulls the
  link toward the gizmo until the user presses Release (or disables the
  tool), which re-attaches the gizmo and zeroes the force.

Force is applied to the single env the viewer follows
(``play_scene.env_idx``). The magnitude is capped so a large drag can't
explode the sim. Use the viewer's Speed control for slow-motion if the
robot still moves too fast to grab comfortably.
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

        # UI-thread flags (set by viser callbacks; read by tick()). Every
        # simulator touch — wrench set/clear, gizmo create/remove — is
        # deferred to tick(), which runs under the viewer's sim lock, so
        # the viser callback threads never race the physics step.
        self._enabled = False
        self._dragging = False
        self._force_active = False  # a wrench is currently set on the env
        self._gizmo: Any | None = None
        self._spring_line: Any | None = None

        # Selectable robot links (exclude world-fixed bodies).
        #
        # Entries are made unique before they reach the dropdown. Two
        # copies of one robot give their links identical names, and a
        # duplicated option value takes the whole web client down — the
        # page renders blank, with no controls at all, so the failure
        # does not look like it came from here. The lookup below would
        # also have silently kept only the last body of each name, which
        # sends every drag on the first robot to the second.
        scene = play_scene._scene
        groups = [g for g in scene.geometry.mesh_groups if not g.is_fixed]
        seen: dict[str, int] = {}
        self._link_names: list[str] = []
        self._link_body_id: dict[str, int] = {}
        for group in groups:
            label = group.body_name
            if label in seen:
                label = f"{group.body_name} #{group.body_id}"
            seen[group.body_name] = seen.get(group.body_name, 0) + 1
            self._link_names.append(label)
            self._link_body_id[label] = group.body_id
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
                self._enabled = self._enable_cb.value
                self._request_update()

            @self._link_dd.on_update
            def _(_e: Any) -> None:
                self._link_name = self._link_dd.value
                self._dragging = False  # re-attach the gizmo to the new link
                self._request_update()

            @self._release_btn.on_click
            def _(_e: Any) -> None:
                self._dragging = False
                self._request_update()

    def _request_update(self) -> None:
        """Wake the render loop so tick() processes a flag change promptly."""
        self._play_scene.needs_update = True

    # ── Per-frame update (called under the sim lock) ───────────────

    def tick(self) -> None:
        """Drive the gizmo and wrench for the current viewer env.

        Runs under the viewer's sim lock inside the atomic scene update,
        so this is the only place that touches the simulator (wrench
        set/clear) or the gizmo scene node — the viser callbacks only
        flip flags.
        """
        if not self._enabled:
            self._teardown()
            return

        if self._gizmo is None:
            self._create_gizmo()

        link_scene = self._link_scene_pos()

        if not self._dragging:
            # Not dragging: drop any force so the robot recovers, then
            # keep the gizmo sitting on the link.
            if self._force_active:
                self._env.clear_external_wrench()
                self._force_active = False
            if self._spring_line is not None:
                self._spring_line.visible = False
            self._gizmo.position = tuple(float(x) for x in link_scene)
            return

        gizmo_pos = np.asarray(self._gizmo.position, dtype=np.float64)
        # gizmo and link both carry the same scene offset this frame, so
        # the difference is the true world-frame displacement.
        disp = gizmo_pos - link_scene
        force = self._stiffness.value * disp
        mag = float(np.linalg.norm(force))
        fmax = self._max_force.value
        if mag > fmax and mag > 0.0:
            force = force * (fmax / mag)

        force_t = torch.as_tensor(force, dtype=torch.float32, device=self._env.device)
        self._env.set_external_wrench(self._link_name, force_t, int(self._play_scene.env_idx))
        self._force_active = True
        self._draw_spring(link_scene, gizmo_pos)

    # ── Gizmo lifecycle (called from tick, under the sim lock) ─────

    def _create_gizmo(self) -> None:
        pos = self._link_scene_pos()
        self._gizmo = self._server.scene.add_transform_controls(
            "/force_drag_gizmo",
            scale=0.3,
            disable_rotations=True,
            disable_sliders=True,
            position=tuple(float(x) for x in pos),
        )

        # Force is applied only between mouse-down and mouse-up on the
        # gizmo. The callbacks only flip ``_dragging``; tick() clears the
        # wrench and re-attaches the gizmo on release, so the robot springs
        # back and recovers instead of being dragged forever.
        @self._gizmo.on_drag_start
        def _(_e: Any) -> None:
            self._dragging = True
            self._request_update()

        @self._gizmo.on_drag_end
        def _(_e: Any) -> None:
            self._dragging = False
            self._request_update()

    def _teardown(self) -> None:
        """Disabled: drop the force and remove the gizmo / spring."""
        if self._force_active:
            self._env.clear_external_wrench()
            self._force_active = False
        self._dragging = False
        if self._gizmo is not None:
            self._gizmo.remove()
            self._gizmo = None
        if self._spring_line is not None:
            self._spring_line.remove()
            self._spring_line = None

    def _link_scene_pos(self) -> np.ndarray:
        """Selected link's on-screen (scene) position for the current env.

        Scene position = world position + the scene's re-centring offset,
        matching where the body is actually rendered so the gizmo sits on
        the link even while camera tracking is on.
        """
        scene = self._play_scene._scene
        env_idx = int(self._play_scene.env_idx)
        world = scene.bridge.get_body_positions(env_idx)[self._link_body_id[self._link_name]]
        return world.astype(np.float64) + np.asarray(scene._scene_offset, dtype=np.float64)

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
