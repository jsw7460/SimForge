"""Interactive Viser play viewer with real-time pacing and simulation controls.

Concrete implementation of PlayViewerBase using Viser for 3D rendering
and GUI. Scene rendering is delegated to a PlayScene instance, which
abstracts over simulator-specific backends (ViserScene for Newton/Genesis,
ViserMujocoScene for MuJoCo).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
import trimesh.visual
import viser

from ._ghost import MotionGhost
from .camera_panel import ViserCameraPanel
from .command_panel import ViserCommandPanel
from .contact_overlay import ContactForceOverlay
from .force_drag import ForceDragController
from .genesis_contact_overlay import GenesisContactOverlay
from .mjv_contact_overlay import MjvContactOverlay
from .overlays import ViserDebugOverlays, ViserTermOverlays
from .play_scene import PlayScene
from .play_viewer_base import PlayViewerBase
from .viewer import (
    _ACTUAL_ARROW_COLOR,
    _ANG_VEL_NEG_COLOR,
    _ANG_VEL_POS_COLOR,
    _ANG_VEL_THRESHOLD,
    _ARROW_HEAD_RADIUS,
    _ARROW_LENGTH_BUCKETS,
    _ARROW_LENGTH_SCALE,
    _ARROW_SHAFT_RADIUS,
    _ARROW_Z_OFFSET,
    _CMD_ARROW_COLOR,
    _HEAD_LENGTH_RATIO,
    _MAX_ARROW_LENGTH,
    _SHAFT_LENGTH_RATIO,
    _get_unit_head_mesh,
    _get_unit_shaft_mesh,
    _rotation_quat_from_vectors,
)

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World
    from rlworld.rl.evals.policy_wrappers import PolicyWrapper


class _UpdateReason(Enum):
    ACTION = auto()
    ENV_SWITCH = auto()
    SCENE_REQUEST = auto()


class ViserPlayViewer(PlayViewerBase):
    """Interactive Viser-based viewer with playback controls."""

    def __init__(
        self,
        env: World,
        play_scene: PlayScene,
        policy: PolicyWrapper,
        frame_rate: float = 60.0,
        port: int = 8080,
        share: bool = True,
    ) -> None:
        super().__init__(env, policy, frame_rate)
        self._play_scene = play_scene
        self._port = port
        self._share = share
        self._sim_lock = Lock()
        self._term_overlays: ViserTermOverlays | None = None
        self._debug_overlays: ViserDebugOverlays | None = None
        self._cmd_arrow_handles: tuple | None = None
        self._actual_arrow_handles: tuple | None = None
        self._ang_vel_handle = None
        # One panel per CommandTerm that declares ``get_ui_spec()``. Built
        # in :meth:`_setup_command_panels`; iterated in
        # :meth:`_apply_command_override` and :meth:`_on_env_switch`.
        self._command_panels: list[ViserCommandPanel] = []
        # Env index the panels are currently locked to. Updated when the
        # camera switches the followed env so panels can release the old
        # env and re-lock the new one.
        self._panel_locked_env: int | None = None
        # Translucent reference-pose overlay (motion-tracking presets
        # only; ``MotionGhost.is_active`` is False otherwise).
        self._motion_ghost: MotionGhost | None = None
        self._camera_panel: ViserCameraPanel | None = None

    # ── Setup ──────────────────────────────────────────────────────

    def setup(self) -> None:
        self._server = viser.ViserServer(port=self._port, label="SimForge PlayViewer")
        if self._share:
            self._server.request_share_url()

        self._threadpool = ThreadPoolExecutor(max_workers=1)
        self._counter = 0
        self._pending_reasons: set[_UpdateReason] = set()

        # Build 3D scene (geometry, ground plane, etc.).
        self._play_scene.create(self._server)

        # Translucent reference-pose overlay (no-op when the env has no
        # 'motion' command — e.g. locomotion / getup presets).
        self._motion_ghost = MotionGhost(self._server, self.env)

        # GUI.
        tabs = self._server.gui.add_tab_group()
        self._build_controls_tab(tabs)
        self._build_inspect_tab(tabs)
        self._play_scene.setup_gui(tabs)
        self._play_scene.set_on_env_switch(self._on_env_switch)
        self._setup_overlays(tabs)
        self._setup_command_panels(tabs)
        # What the policy sees, when it sees anything: one panel per
        # channel of every image observation group. Absent on a
        # state-only preset.
        self._camera_panel = ViserCameraPanel(self._server, self.env)
        if self._camera_panel.is_supported:
            self._camera_panel.build_ui(tabs)
        else:
            self._camera_panel = None
        # Interactive external-force drag tool (Newton / Genesis bridge
        # scenes only; see ForceDragController.is_supported).
        self._force_drag = None
        if ForceDragController.is_supported(self._play_scene):
            self._force_drag = ForceDragController(self._server, self.env, self._play_scene)
            self._force_drag.build_ui(tabs)
        # Contact visualisation: sensor-group arrows plus the engine's own
        # contacts (each class gates on its own backend, so at most one of
        # the two engine overlays has anything to show). Their controls go
        # in one tab — built from the constructor they would land at the
        # root of the panel and follow the viewer onto every other tab.
        self._contact_overlay = ContactForceOverlay(self._server, self.env)
        self._mjv_contacts = MjvContactOverlay(self._server, self.env)
        self._gs_contacts = GenesisContactOverlay(self._server, self.env)
        self._build_contacts_tab(tabs)
        # Motion picker (only renders when the env exposes a 'motion' command
        # term — i.e. tracking presets; no-op on locomotion / getup / ...).
        self._build_motion_controls(tabs)

        self._update_status_display()
        print(f"[PlayViewer] Started on port {self._port}. Open the URL above to view. Press Play to start.")

    def _build_contacts_tab(self, tabs: Any) -> None:
        """One "Contacts" tab holding every contact overlay's controls.

        Skipped when no overlay applies, so a backend without contact
        sensors or engine access does not get an empty tab.
        """
        overlays = (self._contact_overlay, self._mjv_contacts, self._gs_contacts)
        if not any(o.is_available for o in overlays):
            return
        with tabs.add_tab("Contacts"):
            for overlay in overlays:
                overlay.build_ui()

    def _build_controls_tab(self, tabs: Any) -> None:
        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            with self._server.gui.add_folder("Info"):
                self._status_html = self._server.gui.add_html("")

            with self._server.gui.add_folder("Simulation"):
                self._pause_button = self._server.gui.add_button(
                    "Play" if self._is_paused else "Pause",
                    icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
                )
                self._pause_button.on_click(lambda _: self.request_toggle_pause())

                step_btn = self._server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
                step_btn.on_click(lambda _: self.request_single_step())

                reset_btn = self._server.gui.add_button("Reset Environment")
                reset_btn.on_click(lambda _: self.request_reset())

                speed_btns = self._server.gui.add_button_group("Speed", options=["Slower", "1x", "Faster"])

                @speed_btns.on_click
                def _(event) -> None:
                    v = event.target.value
                    if v == "Slower":
                        self.request_speed_down()
                    elif v == "1x":
                        self.request_reset_speed()
                    else:
                        self.request_speed_up()

    # ── Inspect tab: live link/joint readout + foot-height reference ──
    def _build_inspect_tab(self, tabs: Any) -> None:
        """Live per-frame readout of foot height / link positions / joint
        angles, plus a movable horizontal reference plane. All values are read
        from the SAME accessors the rewards use (``robot_data.body_pos_w_by_ids``
        / ``joint_pos``), so the foot z shown here is exactly the height the
        ``feet_clearance`` / ``feet_swing_height`` rewards optimise against.
        """
        self._foot_peak: dict[int, float] = {}
        self._foot_ids: dict[str, int] | None = None  # resolved lazily
        with tabs.add_tab("Inspect", icon=viser.Icon.RULER_2):
            with self._server.gui.add_folder("Readout"):
                self._inspect_mode = self._server.gui.add_dropdown(
                    "Show",
                    options=["Off", "Foot height", "Link positions", "Joint angles"],
                    initial_value="Foot height",
                )
                self._inspect_html = self._server.gui.add_html("")
                reset_peak = self._server.gui.add_button("Reset foot peak")
                reset_peak.on_click(lambda _: self._foot_peak.clear())
            with self._server.gui.add_folder("Height reference"):
                self._ref_show = self._server.gui.add_checkbox("Show plane", initial_value=False)
                self._ref_height = self._server.gui.add_slider(
                    # Up to 1.2 m: tall enough to lay the plane through a
                    # tabletop (0.6) or a lift goal band (0.72-0.88), not
                    # just foot-swing heights.
                    "Height (m)",
                    min=0.0,
                    max=1.20,
                    step=0.005,
                    initial_value=0.11,
                )
                self._ref_show.on_update(lambda _: self._sync_ref_plane())
                self._ref_height.on_update(lambda _: self._sync_ref_plane())

        # A thin translucent plate at world z = slider; foot mesh renders above
        # it iff its world z clears the slider height (Z is unshifted by the
        # scene-follow offset, so render z == world z).
        plate = trimesh.creation.box(extents=(3.0, 3.0, 0.004))
        plate.visual = trimesh.visual.ColorVisuals(plate, face_colors=[80, 180, 255, 110])
        self._ref_plane = self._server.scene.add_mesh_trimesh(name="/height_ref", mesh=plate)
        self._ref_plane.position = (0.0, 0.0, float(self._ref_height.value))
        self._ref_plane.visible = False

    def _sync_ref_plane(self) -> None:
        plane = getattr(self, "_ref_plane", None)
        if plane is None:
            return
        plane.position = (0.0, 0.0, float(self._ref_height.value))
        plane.visible = bool(self._ref_show.value)

    def _resolve_foot_ids(self, rd: Any) -> dict[str, int]:
        candidates = (
            "left_foot_link",
            "right_foot_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_foot",
            "right_foot",
        )
        ids: dict[str, int] = {}
        for name in candidates:
            try:
                ids[name] = int(rd.find_body_index(name))
            except Exception:  # noqa: BLE001 — body simply absent on this robot
                pass
        return ids

    def _update_inspect_display(self) -> None:
        html = getattr(self, "_inspect_html", None)
        if html is None:
            return
        mode = self._inspect_mode.value
        if mode == "Off":
            html.content = ""
            return

        def _wrap(inner: str) -> str:
            return (
                '<div style="font-size:0.8em;line-height:1.3;padding:0 1em .5em 1em;'
                'font-family:monospace;">' + inner + "</div>"
            )

        try:
            import torch

            env_idx = int(self._play_scene.env_idx)
            rd = self.env.get_robot_data("robot")

            if mode == "Foot height":
                if self._foot_ids is None:
                    self._foot_ids = self._resolve_foot_ids(rd)
                if not self._foot_ids:
                    html.content = _wrap("<i>no foot bodies found on this robot</i>")
                    return
                rows = []
                for name, bid in self._foot_ids.items():
                    ids_t = torch.tensor([bid], device=self.env.device)
                    z = float(rd.body_pos_w_by_ids(ids_t)[env_idx, 0, 2])
                    peak = max(self._foot_peak.get(bid, z), z)
                    self._foot_peak[bid] = peak
                    rows.append(f"{name:22s} z={z:6.3f}  peak={peak:6.3f}")
                note = "<span style='color:#888'>link-origin z (== reward height); sole ≈ z − 0.038</span>"
                html.content = _wrap("<br/>".join(rows) + "<br/>" + note)

            elif mode == "Link positions":
                pos = rd.body_pos_w_all[env_idx].detach().cpu().numpy()  # (num_bodies, 3)
                rows = [f"body {i:2d}  x={p[0]:6.3f} y={p[1]:6.3f} z={p[2]:6.3f}" for i, p in enumerate(pos)]
                html.content = _wrap("<br/>".join(rows))

            elif mode == "Joint angles":
                names = list(self.env.act_manager.actuated_joint_names)
                q = rd.joint_pos[env_idx].detach().cpu().numpy()
                rows = [f"{n:22s} {float(v):+.3f}" for n, v in zip(names, q)]
                html.content = _wrap("<br/>".join(rows))
        except Exception as e:  # noqa: BLE001 — diagnostic readout, never crash the viewer
            html.content = _wrap(f"<i>inspect error: {e}</i>")

    def _build_motion_controls(self, tabs: Any) -> None:
        """Add a Motion tab exposing clip selection + rollover-lock toggle.

        Shown only for tracking envs (where the command manager has a
        ``"motion"`` term). The tab is skipped entirely on locomotion /
        getup presets so we don't pollute their GUIs with dead controls.
        """
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return

        # Lock defaults ON for interactive eval — otherwise short clips
        # (e.g. walking1 at 1.28s vs a 10s episode) teleport ~8x per
        # episode which ruins visualization. Using the same ``_set_motion_lock``
        # path the checkbox uses also suspends the tracking-termination
        # terms, so the episode stays alive while the clip loops. Flip
        # at runtime via the "Lock motion" checkbox below.
        self._set_motion_lock(True)

        from pathlib import Path

        clip_names = [Path(p).stem for p in motion_cmd.cfg.motion_files]
        n_clips = motion_cmd._n_motions
        # Reflect whichever clip env 0 is currently tracking so the
        # dropdown's initial value matches sim state.
        try:
            current_idx = int(motion_cmd.motion_ids[0].item())
        except Exception:
            current_idx = 0
        current_idx = max(0, min(current_idx, n_clips - 1))

        with tabs.add_tab("Motion", icon=viser.Icon.PLAYER_PLAY):
            with self._server.gui.add_folder("Playback"):
                lock_cb = self._server.gui.add_checkbox(
                    "Lock motion (loop without teleport)",
                    initial_value=True,
                    hint=(
                        "When ON, motion-end rollover rewinds the clip "
                        "cursor without re-writing sim state, so short "
                        "clips loop smoothly instead of teleporting the "
                        "robot every cycle."
                    ),
                )

                @lock_cb.on_update
                def _on_lock(event) -> None:
                    self.request_set_motion_lock(event.target.value)

                if n_clips > 1:
                    dropdown = self._server.gui.add_dropdown(
                        "Clip",
                        options=clip_names,
                        initial_value=clip_names[current_idx],
                        hint=(
                            "Switch the clip being tracked by every env. "
                            "Takes effect next step; the robot holds its "
                            "current pose and starts following the new "
                            "reference from frame 0."
                        ),
                    )

                    @dropdown.on_update
                    def _on_clip(event) -> None:
                        idx = clip_names.index(event.target.value)
                        self.request_set_motion_clip(idx)

            # Reference-pose ghost overlay controls (only visible when
            # the ghost is active — i.e. MJCF meshes loaded successfully).
            if self._motion_ghost is not None and self._motion_ghost.is_active:
                with self._server.gui.add_folder("Reference ghost"):
                    ghost_cb = self._server.gui.add_checkbox(
                        "Show",
                        initial_value=True,
                        hint=(
                            "Translucent silhouette of the motion-reference "
                            "robot pose, drawn alongside the live robot so "
                            "tracking error is visible at a glance."
                        ),
                    )

                    @ghost_cb.on_update
                    def _on_show_ghost(event) -> None:
                        if self._motion_ghost is not None:
                            self._motion_ghost.set_visible(event.target.value)

                    ghost_op = self._server.gui.add_slider(
                        "Opacity",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        initial_value=0.35,
                    )

                    @ghost_op.on_update
                    def _on_ghost_opacity(event) -> None:
                        if self._motion_ghost is not None:
                            self._motion_ghost.set_opacity(event.target.value)

    def _setup_overlays(self, tabs: Any) -> None:
        self._term_overlays = ViserTermOverlays(
            server=self._server,
            env=self.env,
            scene=self._play_scene,
        )
        self._term_overlays.setup_tabs(tabs)

        self._debug_overlays = ViserDebugOverlays(env=self.env, scene=self._play_scene)

    # ── Command panels ─────────────────────────────────────────────

    def _setup_command_panels(self, tabs: Any) -> None:
        """Build one :class:`ViserCommandPanel` per command term that declares a UI spec.

        Only terms whose ``get_ui_spec()`` returns non-None get a panel —
        terms with no interactive knobs (e.g. ``MotionCommand``) are
        skipped here, and the "Commands" tab is omitted entirely if no
        term has a spec (no empty tab on tracking / getup presets).
        """
        cmd_manager = getattr(self.env, "command_manager", None)
        if cmd_manager is None:
            return
        terms_with_specs: list[tuple[str, Any, Any]] = []
        for name, term in cmd_manager.iter_terms():
            spec = term.get_ui_spec()
            if spec is None:
                continue
            terms_with_specs.append((name, term, spec))
        if not terms_with_specs:
            return
        # Initial lock target is whatever env the camera starts on.
        self._panel_locked_env = int(self._play_scene.env_idx)
        with tabs.add_tab("Commands"):
            for name, term, spec in terms_with_specs:
                self._command_panels.append(
                    ViserCommandPanel(
                        server=self._server,
                        term_name=name,
                        term=term,
                        spec=spec,
                    )
                )

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_env_switch(self) -> None:
        self._pending_reasons.add(_UpdateReason.ENV_SWITCH)
        if self._term_overlays:
            self._term_overlays.on_env_switch()
        if self._debug_overlays:
            self._debug_overlays.on_env_switch()
        # Release any panel locks held on the previous env. Panels
        # automatically re-lock the new env on the next ``apply`` tick
        # if manual override is still ON.
        new_idx = int(self._play_scene.env_idx)
        old_idx = self._panel_locked_env if self._panel_locked_env is not None else new_idx
        for panel in self._command_panels:
            panel.on_env_switch(old_idx, new_idx)
        self._panel_locked_env = new_idx

    def _apply_command_override(self) -> None:
        """Fan out per-tick UI state to every command panel.

        Each panel decides on its own whether to write into the
        underlying CommandTerm (only when manual override is ON for
        that panel). Cheap when no panel has manual mode active.
        """
        env_idx = int(self._play_scene.env_idx)
        for panel in self._command_panels:
            panel.apply(env_idx)

    def _process_actions(self) -> None:
        self._apply_command_override()
        had_actions = bool(self._actions)
        super()._process_actions()
        if had_actions:
            self._pending_reasons.add(_UpdateReason.ACTION)
            self._sync_ui_state()

    def _sync_ui_state(self) -> None:
        self._pause_button.label = "Play" if self._is_paused else "Pause"
        self._pause_button.icon = viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
        self._update_status_display()

    def reset_environment(self) -> None:
        with self._sim_lock:
            super().reset_environment()
        if self._term_overlays:
            self._term_overlays.on_env_switch()

    # ── Sync loop ──────────────────────────────────────────────────

    @staticmethod
    def _should_submit(counter: int, paused: bool, has_pending: bool) -> bool:
        """30Hz gating (every other 60Hz tick), skip when paused with no changes."""
        if counter % 2 != 0:
            return False
        return not paused or has_pending

    def sync_env_to_viewer(self) -> None:
        self._counter += 1

        if self._counter % 10 == 0:
            self._update_status_display()

        if self._term_overlays:
            self._term_overlays.update(paused=self._is_paused)

        has_pending = bool(self._pending_reasons) or self._play_scene.needs_update
        if self._play_scene.needs_update:
            self._pending_reasons.add(_UpdateReason.SCENE_REQUEST)

        will_submit = self._should_submit(self._counter, self._is_paused, has_pending)

        if will_submit and self._debug_overlays:
            with self._sim_lock:
                self._debug_overlays.queue()

        if not will_submit:
            return

        def _do_update() -> None:
            try:
                with self._sim_lock:
                    with self._server.atomic():
                        # Invalidate the bridge's per-frame cache so
                        # ``scene.update`` + ``_update_command_arrows`` +
                        # the motion ghost share one batched fetch.
                        self._play_scene.begin_frame()
                        # Queue debug visuals BEFORE update() so the same
                        # frame's _sync_debug_visuals (called inside update)
                        # picks them up — otherwise they lag by one frame.
                        self._update_command_markers()
                        self._play_scene.update()
                        if self._force_drag is not None:
                            # After scene.update so the frozen (offset-0)
                            # transforms are current; reads a body pose and
                            # writes env.set_external_wrench under the lock.
                            self._force_drag.tick()
                        self._update_command_arrows()
                        self._contact_overlay.update(self._play_scene.env_idx, self._play_scene.scene_offset)
                        self._mjv_contacts.update(self._play_scene.env_idx, self._play_scene.scene_offset)
                        self._gs_contacts.update(self._play_scene.env_idx, self._play_scene.scene_offset)
                        if self._motion_ghost is not None:
                            self._motion_ghost.update(self._play_scene.env_idx)
                        self._server.flush()
            except Exception:
                import traceback

                print(f"[PlayViewer] Scene update error:\n{traceback.format_exc()}")

        self._threadpool.submit(_do_update)
        self._pending_reasons.clear()
        self._play_scene.needs_update = False

    def sync_viewer_to_env(self) -> None:
        pass

    # ── Command arrows ──────────────────────────────────────────────

    def _update_command_arrows(self) -> None:
        tracked = self._play_scene.get_tracked_body_data()
        if tracked is None:
            return

        arrow_origin = tracked.position + tracked.scene_offset
        arrow_origin[2] += _ARROW_Z_OFFSET

        cmd_manager = getattr(self.env, "command_manager", None)
        if cmd_manager is None:
            return

        env_idx = self._play_scene.env_idx
        cmd_vx = getattr(cmd_manager, "lin_vel_x", None)
        cmd_vy = getattr(cmd_manager, "lin_vel_y", None)
        cmd_ang = getattr(cmd_manager, "ang_vel", None)

        if cmd_vx is not None and cmd_vy is not None:
            self._cmd_arrow_handles = self._draw_velocity_arrow(
                arrow_origin,
                float(cmd_vx[env_idx]),
                float(cmd_vy[env_idx]),
                tracked.yaw,
                _CMD_ARROW_COLOR,
                "/overlay/cmd_arrow",
                self._cmd_arrow_handles,
            )

        if tracked.body_velocity is not None:
            origin_actual = arrow_origin.copy()
            origin_actual[2] -= 0.05
            self._actual_arrow_handles = self._draw_velocity_arrow(
                origin_actual,
                float(tracked.body_velocity[0]),
                float(tracked.body_velocity[1]),
                tracked.yaw,
                _ACTUAL_ARROW_COLOR,
                "/overlay/actual_arrow",
                self._actual_arrow_handles,
            )

        if cmd_ang is not None:
            self._ang_vel_handle = self._draw_angular_indicator(
                arrow_origin,
                float(cmd_ang[env_idx]),
                self._ang_vel_handle,
            )

    # ── Command markers ─────────────────────────────────────────────
    #
    # Any command term that names a place in the world says so through
    # ``CommandTerm.get_marker_positions_w``, and it is drawn as a small
    # sphere. Asking the term rather than matching on its name is what
    # makes this work for a term the viewer has never heard of — the
    # previous version looked for one hard-coded name that no term in
    # this repository uses, so it drew nothing, ever.
    #
    # Uses the existing debug-sphere infra (ViserScene.add_sphere), which
    # applies the camera-tracking scene offset, so a marker stays put
    # relative to the rendered robot.
    _MARKER_RADIUS = 0.03
    _MARKER_COLORS = ((255, 80, 80), (80, 160, 255), (120, 220, 120), (240, 200, 80))

    def _update_command_markers(self) -> None:
        env_idx = self._play_scene.env_idx
        for order, (_, term) in enumerate(self.env.command_manager.iter_terms()):
            positions = term.get_marker_positions_w()
            if positions is None:
                continue
            self._play_scene.add_sphere(
                position=positions[env_idx].detach().cpu().numpy(),
                radius=self._MARKER_RADIUS,
                # One colour per term, so two arms reaching for two
                # targets do not both get a red dot.
                color=self._MARKER_COLORS[order % len(self._MARKER_COLORS)],
            )

    def _draw_velocity_arrow(
        self,
        origin: np.ndarray,
        vel_x: float,
        vel_y: float,
        yaw: float,
        color: tuple[int, int, int],
        name: str,
        old_handles: tuple | None,
    ) -> tuple | None:
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        world_vx = cos_yaw * vel_x - sin_yaw * vel_y
        world_vy = sin_yaw * vel_x + cos_yaw * vel_y
        magnitude = np.sqrt(world_vx**2 + world_vy**2)

        if magnitude < 1e-4:
            if old_handles is not None:
                old_handles[0].remove()
                old_handles[1].remove()
            return None

        direction = np.array([world_vx, world_vy, 0.0]) / magnitude
        z_axis = np.array([0.0, 0.0, 1.0])
        rot_quat = _rotation_quat_from_vectors(z_axis, direction)
        r, g, b = color

        # A remove + add round-trip blinks in the browser, so the meshes
        # are rebuilt only when the arrow LENGTH crosses a quantization
        # bucket; position and orientation mutate in place every frame.
        raw_length = min(_MAX_ARROW_LENGTH, magnitude * _ARROW_LENGTH_SCALE)
        bucket = max(1, round(raw_length / _MAX_ARROW_LENGTH * _ARROW_LENGTH_BUCKETS))
        arrow_length = bucket / _ARROW_LENGTH_BUCKETS * _MAX_ARROW_LENGTH

        shaft_length = _SHAFT_LENGTH_RATIO * arrow_length
        if old_handles is not None and old_handles[2] == bucket:
            shaft_h, head_h = old_handles[0], old_handles[1]
            shaft_h.position = tuple(origin)
            shaft_h.wxyz = tuple(rot_quat)
            head_h.position = tuple(origin + direction * shaft_length)
            head_h.wxyz = tuple(rot_quat)
            return old_handles

        if old_handles is not None:
            old_handles[0].remove()
            old_handles[1].remove()
        shaft = _get_unit_shaft_mesh().copy()
        shaft.visual = trimesh.visual.ColorVisuals(
            mesh=shaft,
            face_colors=np.tile([r, g, b, 255], (len(shaft.faces), 1)),
        )
        shaft_h = self._server.scene.add_mesh_trimesh(
            name=f"{name}/shaft",
            mesh=shaft,
            position=tuple(origin),
            wxyz=tuple(rot_quat),
            scale=(_ARROW_SHAFT_RADIUS, _ARROW_SHAFT_RADIUS, shaft_length),
        )

        head_length = _HEAD_LENGTH_RATIO * arrow_length
        head_pos = origin + direction * shaft_length
        head = _get_unit_head_mesh().copy()
        head.visual = trimesh.visual.ColorVisuals(
            mesh=head,
            face_colors=np.tile([r, g, b, 255], (len(head.faces), 1)),
        )
        head_h = self._server.scene.add_mesh_trimesh(
            name=f"{name}/head",
            mesh=head,
            position=tuple(head_pos),
            wxyz=tuple(rot_quat),
            scale=(_ARROW_HEAD_RADIUS, _ARROW_HEAD_RADIUS, head_length),
        )
        return (shaft_h, head_h, bucket)

    def _draw_angular_indicator(
        self,
        origin: np.ndarray,
        ang_vel: float,
        old_handle: Any,
    ) -> Any:
        if abs(ang_vel) < _ANG_VEL_THRESHOLD:
            if old_handle is not None:
                old_handle[0].remove()
            return None
        color = _ANG_VEL_POS_COLOR if ang_vel > 0 else _ANG_VEL_NEG_COLOR
        # Same anti-blink treatment as the arrows: rebuild the mesh only
        # when the quantized radius or the spin direction changes.
        radius = 0.03 + 0.03 * round(min(1.0, abs(ang_vel)) * 8) / 8
        pos = origin.copy()
        pos[2] += 0.15
        if old_handle is not None and old_handle[1] == (radius, color):
            old_handle[0].position = tuple(pos)
            return old_handle
        if old_handle is not None:
            old_handle[0].remove()
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        r, g, b = color
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile([r, g, b, 255], (len(mesh.faces), 1)),
        )
        handle = self._server.scene.add_mesh_trimesh(
            name="/overlay/ang_vel",
            mesh=mesh,
            position=tuple(pos),
        )
        return (handle, (radius, color))

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        for panel in self._command_panels:
            panel.cleanup()
        if self._term_overlays:
            self._term_overlays.cleanup()
        self._play_scene.cleanup()
        self._threadpool.shutdown(wait=True)
        self._server.stop()

    def is_running(self) -> bool:
        return True

    # ── Status display ─────────────────────────────────────────────

    def _update_status_display(self) -> None:
        s = self.get_status()
        rt = f"{s.actual_realtime:.2f}x" if s.actual_realtime > 0 else "\u2014"
        cap = ' <span style="color:#e74c3c;">[CAPPED]</span>' if s.capped else ""
        err = ""
        if s.last_error:
            line = s.last_error.strip().splitlines()[-1]
            err = f'<br/><span style="color:#e74c3c;"><strong>Error:</strong> {line}</span>'
        self._status_html.content = (
            '<div style="font-size:0.85em;line-height:1.25;padding:0 1em .5em 1em;">'
            f"<strong>Status:</strong> {'Paused' if s.paused else 'Running'}{cap}<br/>"
            f"<strong>Steps:</strong> {s.step_count}<br/>"
            f"<strong>Speed:</strong> {s.speed_label}<br/>"
            f"<strong>Target RT:</strong> {s.target_realtime:.2f}x<br/>"
            f"<strong>Actual RT:</strong> {rt} ({s.smoothed_fps:.0f} FPS){err}"
            "</div>"
        )
        self._update_inspect_display()
        if self._camera_panel is not None:
            self._camera_panel.update(int(self._play_scene.env_idx))
