"""Translucent "ghost" overlay of a motion-tracking reference robot pose.

Draws the per-body reference pose of the env's active
:class:`MotionCommand` as a semi-transparent robot silhouette next to
the live robot in the viser scene — so the user can eyeball tracking
error (anchor drift, joint deviation) at a glance.

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
    from rlworld.rl.envs.mdp.commands.motion import MotionCommand
    from rlworld.rl.envs.world import World


# Pale-cyan, fairly translucent — reads cleanly against both the live
# robot and the default viser background.
_GHOST_COLOR_RGB = (120, 200, 255)
_GHOST_OPACITY = 0.35


class MotionGhost:
    """Per-body ghost meshes + per-tick transform updates."""

    def __init__(
        self,
        server: viser.ViserServer,
        env: World,
        color: tuple[int, int, int] = _GHOST_COLOR_RGB,
        opacity: float = _GHOST_OPACITY,
    ) -> None:
        self._server = server
        self._env = env
        self._color = color
        self._opacity = opacity
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

    def update(self, env_idx: int) -> None:
        """Pull the reference body transforms for ``env_idx`` from the
        MotionCommand and write them onto the viser handles. Cheap —
        only sets ``position`` / ``wxyz`` per handle."""
        cmd = self._motion_command()
        if cmd is None or not self._handles:
            return
        body_pos = cmd.body_pos_w[env_idx].detach().cpu().numpy()
        body_quat = cmd.body_quat_w[env_idx].detach().cpu().numpy()
        for i, name in enumerate(cmd.cfg.body_names):
            handle = self._handles.get(name)
            if handle is None:
                continue
            handle.position = tuple(body_pos[i].tolist())
            handle.wxyz = tuple(body_quat[i].tolist())

    def set_visible(self, visible: bool) -> None:
        for handle in self._handles.values():
            handle.visible = visible

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
