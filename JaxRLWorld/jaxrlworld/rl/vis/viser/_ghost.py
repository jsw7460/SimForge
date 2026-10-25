"""Translucent "ghost" overlay of a motion-tracking reference robot pose.

Draws the per-body reference pose of the env's active
:class:`MotionCommand` as a semi-transparent robot silhouette over the
live robot in the viser scene — so the user can eyeball tracking error
(height, tilt, joint deviation) at a glance.

The ghost has two frames, matching the two frames the tracking reward
uses. Anchored to the robot (the default), the reference bodies are
placed at the robot anchor's xy and heading with the reference height
kept (``MotionCommand.body_pos_relative_w`` / ``body_quat_relative_w``),
which is how the body position / orientation / velocity terms compare
them: pose, height and tilt errors show, world drift does not. In the
world frame (``body_pos_w`` / ``body_quat_w``) the ghost follows the
clip's own trajectory, so the anchor position / orientation terms'
error, the robot's world drift, shows instead. A policy that observes no
global position drifts over a clip while tracking the motion itself, so
in the world frame the ghost walks away from the robot by design.

Cross-sim: visual geometry comes from
``SceneManager.get_visual_meshes(body_names)``. Each backend builds
the meshes from whatever single-robot source is canonical for it
(mjlab → ``Scene.compile()``, Newton → re-parsed MJCF, Genesis →
native ``RigidVisGeom``); this file just consumes the dict. The
Mjlab analogue is ``mjlab.tasks.tracking.mdp.commands._debug_vis_impl``
+ ``DebugVisualizer.add_ghost_mesh``; ours is sim-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.mdp.commands.motion import MotionCommand
    from jaxrlworld.rl.envs.world import World


class MotionGhost:
    """Per-body ghost meshes + per-tick transform updates.

    ``color`` / ``opacity`` come from the look
    (``ViserSceneConfig.ghost_color`` / ``ghost_opacity``)."""

    def __init__(
        self,
        server: viser.ViserServer,
        env: World,
        color: tuple[int, int, int],
        opacity: float,
    ) -> None:
        self._server = server
        self._env = env
        self._color = color
        self._opacity = opacity
        self._anchor_to_robot = True
        self._handles: dict[str, viser.MeshHandle] = {}

        cmd = self._motion_command()
        if cmd is None:
            return  # non-tracking preset — nothing to draw

        meshes = env.scene_manager.get_visual_meshes(tuple(cmd.cfg.body_names))
        for name, mesh in meshes.items():
            if mesh is None or len(mesh.vertices) == 0:
                continue
            self._handles[name] = server.scene.add_mesh_simple(
                f"/motion_ghost/{name}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                color=color,
                opacity=opacity,
                cast_shadow=False,
                receive_shadow=False,
            )

    # ── Public API ──────────────────────────────────────────────────

    def update(self, env_idx: int, scene_offset: np.ndarray) -> None:
        """Pull the reference body transforms for ``env_idx`` from the
        MotionCommand, in the frame :meth:`set_anchor_to_robot` selected,
        and write them onto the viser handles. ``scene_offset`` is the
        camera-tracking shift the scene applies to every body so the
        tracked one stays at the origin; the ghost must move with it or it
        drifts away from the robot as the robot walks off the world
        origin. Cheap — only sets ``position`` / ``wxyz`` per handle."""
        cmd = self._motion_command()
        if cmd is None or not self._handles:
            return
        if self._anchor_to_robot:
            pos, quat = cmd.body_pos_relative_w, cmd.body_quat_relative_w
        else:
            pos, quat = cmd.body_pos_w, cmd.body_quat_w
        body_pos = pos[env_idx].detach().cpu().numpy() + scene_offset
        body_quat = quat[env_idx].detach().cpu().numpy()
        for i, name in enumerate(cmd.cfg.body_names):
            handle = self._handles.get(name)
            if handle is None:
                continue
            handle.position = tuple(body_pos[i].tolist())
            handle.wxyz = tuple(body_quat[i].tolist())

    def set_visible(self, visible: bool) -> None:
        for handle in self._handles.values():
            handle.visible = visible

    def set_anchor_to_robot(self, anchored: bool) -> None:
        """``True``: reference re-anchored at the robot's xy and heading
        (the body terms' frame). ``False``: the clip's world trajectory
        (the anchor terms' frame)."""
        self._anchor_to_robot = bool(anchored)

    def set_opacity(self, opacity: float) -> None:
        self._opacity = float(opacity)
        for handle in self._handles.values():
            handle.opacity = self._opacity

    @property
    def is_active(self) -> bool:
        return bool(self._handles)

    # ── Internal ────────────────────────────────────────────────────

    def _motion_command(self) -> MotionCommand | None:
        cm = self._env.command_manager
        terms = getattr(cm, "_terms", None)
        if not terms or "motion" not in terms:
            return None
        return cm.get_term("motion")
