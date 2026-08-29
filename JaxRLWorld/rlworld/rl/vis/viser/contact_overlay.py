"""Contact-force arrow overlay for the viser viewers.

Draws one arrow per tracked body of a contact group: anchored at the
body, pointing along the net contact force, length proportional to
|F|. Works on every backend because it reads the cross-sim
``contact_manager`` API (net force per tracked body) rather than any
engine's contact buffers — the per-contact-point variant is a separate,
per-backend upgrade.

Rendering follows mjviser's decor pattern: two persistent
``add_batched_meshes_simple`` handles (unit shaft cylinder + unit head
cone) whose ``batched_positions / wxyzs / scales`` are mutated in place
every frame. No handle is ever removed between frames, so the browser
never blinks, and hiding an arrow is a zero scale row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_ARROW_COLOR = (235, 100, 40)
_SHAFT_RADIUS = 0.008
_HEAD_RADIUS = 0.02
_SHAFT_RATIO = 0.75
_MAX_LENGTH_M = 1.0

_unit_shaft: trimesh.Trimesh | None = None
_unit_head: trimesh.Trimesh | None = None


def _get_unit_meshes() -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    global _unit_shaft, _unit_head
    if _unit_shaft is None:
        _unit_shaft = trimesh.creation.cylinder(radius=1.0, height=1.0)
        _unit_shaft.apply_translation(np.array([0.0, 0.0, 0.5]))
        _unit_head = trimesh.creation.cone(radius=2.0, height=1.0)
    return _unit_shaft, _unit_head


def _quats_from_z(dirs: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternions rotating +Z onto each row of ``dirs`` (unit)."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dots = dirs @ z
    crosses = np.cross(np.broadcast_to(z, dirs.shape), dirs)
    quats = np.concatenate([(1.0 + dots)[:, None], crosses], axis=1)
    # Antiparallel rows have w ~ 0 and a zero cross product; any 180-deg
    # flip about a horizontal axis works.
    flipped = quats[:, 0] < 1e-8
    quats[flipped] = np.array([0.0, 1.0, 0.0, 0.0])
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


class ContactForceOverlay:
    """GUI folder + batched arrow rendering for one contact group."""

    def __init__(self, server: viser.ViserServer, env: World):
        self._server = server
        self._env = env
        self._handles: tuple | None = None  # (shaft, head, n_rows)
        self._anchor_cache: dict[str, list[int] | None] = {}

        # Only groups that actually track forces: a fields=('found',)
        # group has nothing to draw and contact_force() refuses it.
        groups = [g for g in env.contact_manager.group_names() if "force" in env.contact_manager._groups[g].fields]
        self._available = bool(groups)
        if not self._available:
            return

        with server.gui.add_folder("Contact forces", expand_by_default=False):
            self._enable = server.gui.add_checkbox("Show", initial_value=False)
            self._group = server.gui.add_dropdown("Group", options=tuple(groups), initial_value=groups[0])
            self._scale = server.gui.add_slider("Scale (m/N)", min=0.001, max=0.02, step=0.001, initial_value=0.005)
            self._threshold = server.gui.add_slider("Min force (N)", min=0.0, max=20.0, step=0.5, initial_value=1.0)
            self._status = server.gui.add_markdown("")

    # ── anchors ──────────────────────────────────────────────────────

    def _resolve_anchors(self, group: str) -> list[int] | None:
        """Tracked names -> robot body row indices, cached per group.

        MuJoCo groups track collision geoms (``FL_foot_collision``)
        while Newton/Genesis track bodies (``FL_foot``); stripping the
        ``_collision`` suffix bridges the naming. A group whose names
        resolve to no robot body (e.g. a free object's contacts) is
        reported in the GUI and drawn as nothing rather than guessed.
        """
        if group in self._anchor_cache:
            return self._anchor_cache[group]
        rd = self._env.get_robot_data(self._env.robot_entity_name)
        tracked = self._env.contact_manager.tracked_names(group)
        ids: list[int] = []
        for name in tracked:
            candidates = [name]
            if name.endswith("_collision"):
                candidates.append(name[: -len("_collision")])
            for cand in candidates:
                try:
                    ids.append(rd.find_body_index(cand))
                    break
                except (KeyError, ValueError):
                    continue
            else:
                self._status.content = f"⚠ `{group}`: tracked name `{name}` matches no robot body — not drawn."
                self._anchor_cache[group] = None
                return None
        self._anchor_cache[group] = ids
        return ids

    # ── per-frame update ─────────────────────────────────────────────

    def update(self, env_idx: int, scene_offset: np.ndarray) -> None:
        if not self._available:
            return
        if not self._enable.value:
            self._hide()
            return

        group = self._group.value
        ids = self._resolve_anchors(group)
        if ids is None:
            self._hide()
            return
        try:
            force = self._env.contact_manager.contact_force(group)
        except (ValueError, RuntimeError) as e:
            # Belt over the dropdown filter: a group without a force
            # field (or a backend refusing the read) reports in the GUI
            # instead of killing the viewer's update loop.
            self._status.content = f"⚠ `{group}`: {e}"
            self._hide()
            return
        self._status.content = ""

        force_np = force[env_idx].detach().cpu().numpy().astype(np.float32)  # (N, 3)
        rd = self._env.get_robot_data(self._env.robot_entity_name)
        anchors = rd.body_pos_w_all[env_idx, ids].detach().cpu().numpy().astype(np.float32)
        anchors = anchors + scene_offset.astype(np.float32)

        mags = np.linalg.norm(force_np, axis=1)
        active = mags >= max(float(self._threshold.value), 1e-6)
        lengths = np.clip(mags * float(self._scale.value), 0.0, _MAX_LENGTH_M)
        lengths[~active] = 0.0

        dirs = np.where(mags[:, None] > 1e-9, force_np / np.clip(mags[:, None], 1e-9, None), [[0.0, 0.0, 1.0]])
        quats = _quats_from_z(dirs.astype(np.float32))

        shaft_len = lengths * _SHAFT_RATIO
        head_len = lengths * (1.0 - _SHAFT_RATIO)
        shaft_scales = np.stack(
            [np.full_like(lengths, _SHAFT_RADIUS), np.full_like(lengths, _SHAFT_RADIUS), shaft_len], axis=1
        )
        shaft_scales[~active] = 0.0
        head_scales = np.stack(
            [np.full_like(lengths, _HEAD_RADIUS), np.full_like(lengths, _HEAD_RADIUS), head_len], axis=1
        )
        head_scales[~active] = 0.0
        head_pos = anchors + dirs * shaft_len[:, None]

        n = len(ids)
        if self._handles is not None and self._handles[2] != n:
            self._handles[0].remove()
            self._handles[1].remove()
            self._handles = None

        if self._handles is None:
            shaft_mesh, head_mesh = _get_unit_meshes()
            shaft = self._server.scene.add_batched_meshes_simple(
                "/overlay/contact_forces/shafts",
                shaft_mesh.vertices,
                shaft_mesh.faces,
                batched_wxyzs=quats,
                batched_positions=anchors,
                batched_scales=shaft_scales,
                batched_colors=np.tile(np.array(_ARROW_COLOR, dtype=np.uint8), (n, 1)),
                lod="off",
                cast_shadow=False,
                receive_shadow=False,
            )
            head = self._server.scene.add_batched_meshes_simple(
                "/overlay/contact_forces/heads",
                head_mesh.vertices,
                head_mesh.faces,
                batched_wxyzs=quats,
                batched_positions=head_pos,
                batched_scales=head_scales,
                batched_colors=np.tile(np.array(_ARROW_COLOR, dtype=np.uint8), (n, 1)),
                lod="off",
                cast_shadow=False,
                receive_shadow=False,
            )
            self._handles = (shaft, head, n)
            return

        shaft, head, _ = self._handles
        shaft.visible = True
        head.visible = True
        shaft.batched_positions = anchors
        shaft.batched_wxyzs = quats
        shaft.batched_scales = shaft_scales
        head.batched_positions = head_pos
        head.batched_wxyzs = quats
        head.batched_scales = head_scales

    def _hide(self) -> None:
        if self._handles is not None:
            self._handles[0].visible = False
            self._handles[1].visible = False
