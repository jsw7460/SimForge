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

from .overlays import ViserDebugOverlays, ViserTermOverlays
from .play_scene import PlayScene
from .play_viewer_base import PlayViewerBase
from .viewer import (
    _ACTUAL_ARROW_COLOR,
    _ANG_VEL_NEG_COLOR,
    _ANG_VEL_POS_COLOR,
    _ANG_VEL_THRESHOLD,
    _ARROW_HEAD_RADIUS,
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

        # GUI.
        tabs = self._server.gui.add_tab_group()
        self._build_controls_tab(tabs)
        self._play_scene.setup_gui(tabs)
        self._play_scene.set_on_env_switch(self._on_env_switch)
        self._setup_overlays(tabs)
        # Motion picker (only renders when the env exposes a 'motion' command
        # term — i.e. tracking presets; no-op on locomotion / getup / ...).
        self._build_motion_controls(tabs)

        self._update_status_display()
        print(f"[PlayViewer] Started on port {self._port}. Open the URL above to view. Press Play to start.")

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

    def _setup_overlays(self, tabs: Any) -> None:
        self._term_overlays = ViserTermOverlays(
            server=self._server,
            env=self.env,
            scene=self._play_scene,
        )
        self._term_overlays.setup_tabs(tabs)

        self._debug_overlays = ViserDebugOverlays(env=self.env, scene=self._play_scene)

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_env_switch(self) -> None:
        self._pending_reasons.add(_UpdateReason.ENV_SWITCH)
        if self._term_overlays:
            self._term_overlays.on_env_switch()
        if self._debug_overlays:
            self._debug_overlays.on_env_switch()

    def _process_actions(self) -> None:
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
                        # Queue debug visuals BEFORE update() so the same
                        # frame's _sync_debug_visuals (called inside update)
                        # picks them up — otherwise they lag by one frame.
                        self._update_target_position_marker()
                        self._play_scene.update()
                        self._update_command_arrows()
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

    # ── Target position marker ──────────────────────────────────────
    #
    # Generic hook: if the command manager exposes any 3D-position term
    # (e.g. drone hover ``target_position``), render it as a small
    # pinkish sphere at the world-frame target. Uses the existing
    # debug-sphere infra (ViserScene.add_sphere) which auto-applies the
    # camera-tracking scene offset so the marker stays consistent with
    # the rendered robot frame.
    _TARGET_TERM_NAMES = ("target_position",)
    _TARGET_MARKER_RADIUS = 0.05
    _TARGET_MARKER_COLOR = (255, 80, 80)

    def _update_target_position_marker(self) -> None:
        cmd_manager = getattr(self.env, "command_manager", None)
        if cmd_manager is None:
            return
        for name in self._TARGET_TERM_NAMES:
            try:
                cmd = cmd_manager.get_command(name)
            except KeyError:
                continue
            if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 3:
                continue
            env_idx = self._play_scene.env_idx
            target_xyz = cmd[env_idx].detach().cpu().numpy()
            self._play_scene.add_sphere(
                position=target_xyz,
                radius=self._TARGET_MARKER_RADIUS,
                color=self._TARGET_MARKER_COLOR,
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
                for h in old_handles:
                    h.remove()
            return None

        if old_handles is not None:
            for h in old_handles:
                h.remove()

        arrow_length = min(_MAX_ARROW_LENGTH, magnitude * _ARROW_LENGTH_SCALE)
        direction = np.array([world_vx, world_vy, 0.0]) / magnitude
        z_axis = np.array([0.0, 0.0, 1.0])
        rot_quat = _rotation_quat_from_vectors(z_axis, direction)
        r, g, b = color

        shaft_length = _SHAFT_LENGTH_RATIO * arrow_length
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
        return (shaft_h, head_h)

    def _draw_angular_indicator(
        self,
        origin: np.ndarray,
        ang_vel: float,
        old_handle: Any,
    ) -> Any:
        if old_handle is not None:
            old_handle.remove()
        if abs(ang_vel) < _ANG_VEL_THRESHOLD:
            return None
        color = _ANG_VEL_POS_COLOR if ang_vel > 0 else _ANG_VEL_NEG_COLOR
        radius = 0.03 + 0.03 * min(1.0, abs(ang_vel))
        pos = origin.copy()
        pos[2] += 0.15
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        r, g, b = color
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile([r, g, b, 255], (len(mesh.faces), 1)),
        )
        return self._server.scene.add_mesh_trimesh(
            name="/overlay/ang_vel",
            mesh=mesh,
            position=tuple(pos),
        )

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
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
