"""Sites: named frames rigidly attached to a body.

A MuJoCo ``<site>`` is a massless, non-colliding frame parented to a body.
It carries no dynamics; its only content is a local transform. Manipulation
needs them because the point that matters — the grasp point between the
finger pads — sits at no link origin, and hard-coding that offset into
every reward is how it drifts apart from the asset.

Only MuJoCo declares sites. Newton parses them from an MJCF into shapes
flagged ``ShapeFlags.SITE``, Genesis drops them, and URDF has no such
concept at all (its idiom is a massless link on a fixed joint, which
arrives as an ordinary body and needs nothing from this module).

So a site is stored here as what it fundamentally is — ``(parent body,
local transform)`` — and its world pose is composed the same way on every
backend:

    pos_w = body_pos_w + R(body_quat_w) @ local_pos
    vel_w = body_lin_vel_w + body_ang_vel_w x (R(body_quat_w) @ local_pos)

The velocity form is why ``body_lin_vel_w_all`` must be the velocity at the
LINK FRAME ORIGIN rather than at the centre of mass: the lever arm in the
cross product is measured from the same point the velocity refers to. All
three backends document theirs as link-origin (Newton transfers from CoM
explicitly), so the one formula holds everywhere.

The local transform comes from MuJoCo's own compiler rather than from the
XML text. ``<default>`` classes, ``angle="degree"`` and inherited frame
offsets all mean the attribute as written need not be the compiled value,
and reimplementing that resolution is a good way to be subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_mul_wxyz, quat_rotate_wxyz


@dataclass(frozen=True)
class SiteFrame:
    """One site: the body it rides on, and where it sits on it.

    Attributes:
        name: Site name as declared in the MJCF.
        body_name: Bare name of the parent body.
        local_pos: Offset from the parent body's frame origin, in that
            body's frame [m].
        local_quat_wxyz: Rotation from the parent body's frame, wxyz.
            Unused by position and velocity, kept because a site is a
            frame rather than a point and orientation-valued reads
            (tool axis, approach direction) need it.
    """

    name: str
    body_name: str
    local_pos: tuple[float, float, float]
    local_quat_wxyz: tuple[float, float, float, float]


@lru_cache(maxsize=32)
def sites_from_mjcf(mjcf_path: str) -> tuple[SiteFrame, ...]:
    """Every site in an MJCF, in the order MuJoCo assigns their ids.

    Ordered, not keyed: a caller holding pre-resolved ``site_ids`` indexes
    this sequence, and those ids have to mean the same thing as MuJoCo's
    own so the mjlab path and the other two agree on which site is which.

    Compiled rather than parsed. ``MjSpec.compile()`` resolves defaults,
    angle units and frame nesting; the raw attribute text does not.
    """
    import mujoco

    resolved = Path(mjcf_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"MJCF not found for site extraction: {resolved}")
    model = mujoco.MjSpec.from_file(str(resolved)).compile()

    frames: list[SiteFrame] = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not name:
            # Unnamed sites exist (visual markers declared in a default
            # class). They are unaddressable by name, but they still hold
            # an id, so they must occupy their slot or every later site
            # would shift.
            name = f"__unnamed_site_{site_id}"
        body_id = int(model.site_bodyid[site_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        frames.append(
            SiteFrame(
                name=name,
                body_name=body_name.rsplit("/", 1)[-1],
                local_pos=tuple(float(v) for v in model.site_pos[site_id]),
                local_quat_wxyz=tuple(float(v) for v in model.site_quat[site_id]),
            )
        )
    return tuple(frames)


def site_pos_w(body_pos_w: Tensor, body_quat_w: Tensor, local_pos: Tensor) -> Tensor:
    """World positions of sites riding on the given bodies.

    Args:
        body_pos_w: ``(num_envs, n, 3)`` parent body origins.
        body_quat_w: ``(num_envs, n, 4)`` parent body orientations, wxyz.
        local_pos: ``(n, 3)`` offsets in each parent's frame.

    Returns:
        ``(num_envs, n, 3)``.
    """
    offset_w = quat_rotate_wxyz(body_quat_w, local_pos.expand_as(body_pos_w))
    return body_pos_w + offset_w


def site_quat_w(body_quat_w: Tensor, local_quat: Tensor) -> Tensor:
    """World orientations of sites riding on the given bodies.

    A site is a frame, not a point: the tool axis a task aims along, and
    the frame an end-effector velocity is expressed in, both live in this
    rotation.

    Args:
        body_quat_w: ``(num_envs, n, 4)`` parent orientations, wxyz.
        local_quat: ``(n, 4)`` site rotations in each parent's frame, wxyz.

    Returns:
        ``(num_envs, n, 4)`` wxyz.
    """
    return quat_mul_wxyz(body_quat_w, local_quat.expand_as(body_quat_w))


def site_lin_vel_w(
    body_lin_vel_w: Tensor,
    body_ang_vel_w: Tensor,
    body_quat_w: Tensor,
    local_pos: Tensor,
) -> Tensor:
    """World linear velocities of sites riding on the given bodies.

    ``body_lin_vel_w`` must be the velocity at the body's LINK FRAME
    ORIGIN, since that is the point the lever arm below is measured from.

    Args:
        body_lin_vel_w: ``(num_envs, n, 3)``.
        body_ang_vel_w: ``(num_envs, n, 3)`` world-frame angular velocity.
        body_quat_w: ``(num_envs, n, 4)`` wxyz.
        local_pos: ``(n, 3)`` offsets in each parent's frame.

    Returns:
        ``(num_envs, n, 3)``.
    """
    offset_w = quat_rotate_wxyz(body_quat_w, local_pos.expand_as(body_lin_vel_w))
    return body_lin_vel_w + torch.cross(body_ang_vel_w, offset_w, dim=-1)


def resolve_site_ids(env, entity_name: str, site_names) -> Tensor | None:
    """Site ids for a selector, against the one shared table.

    Returns ``None`` when the selector names no sites, matching how the
    other id fields report "not restricted". Resolved here rather than
    per backend so an id means the same site everywhere — the whole point
    of taking the table from the asset instead of from each simulator.
    """
    if not site_names:
        return None
    config = env.scene_manager.config
    cfg = config.entities.get(entity_name) or config.rigid_objects.get(entity_name)
    if cfg is None or cfg.mjcf_path is None:
        raise ValueError(
            f"Entity {entity_name!r} has no MJCF, so a selector cannot name sites on it (asked for {list(site_names)})."
        )
    frames = sites_from_mjcf(cfg.mjcf_path)
    by_name = {f.name: i for i, f in enumerate(frames)}
    ids = []
    for name in site_names:
        if name not in by_name:
            declared = [f.name for f in frames if not f.name.startswith("__unnamed")]
            raise KeyError(f"Entity {entity_name!r} has no site {name!r}. Declared: {declared or 'none'}.")
        ids.append(by_name[name])
    return torch.tensor(ids, device=env.device, dtype=torch.long)


class SiteReaderMixin:
    """Site reads for a :class:`RobotData`, from body poses alone.

    A backend mixes this in and supplies :attr:`_site_frames` — the
    entity's sites in MuJoCo's own id order. Everything else is the same
    arithmetic everywhere, because the only inputs are the body pose and
    velocity accessors all three already agree on.

    Site ids index :attr:`_site_frames`. They are defined by one table so
    an id resolved once means the same site on every backend; a table
    built per simulator would order them by that parser's internals and
    quietly disagree.
    """

    # Supplied by the mixing class.
    _env: object
    _entity_name: str

    @property
    def _site_frames(self) -> tuple[SiteFrame, ...]:
        """This entity's sites, from the MJCF it was built from.

        Read from the asset rather than from each simulator's own parsed
        model: the id of a site has to mean the same thing on all three,
        and three parsers ordering their own tables is exactly how that
        stops being true. The simulators that DO keep sites are used to
        check this table, not to build it.
        """
        config = self._env.scene_manager.config
        # Articulations and passive props live in separate registries, and
        # a site is equally meaningful on either — a fixture can carry the
        # frame a task aims at.
        cfg = config.entities.get(self._entity_name) or config.rigid_objects.get(self._entity_name)
        if cfg is None:
            raise KeyError(
                f"No scene entity named {self._entity_name!r}. "
                f"Entities: {sorted(config.entities)}; rigid objects: {sorted(config.rigid_objects)}."
            )
        if cfg.mjcf_path is None:
            raise ValueError(
                f"Entity {self._entity_name!r} declares no mjcf_path, so it has no sites. "
                "Sites are an MJCF concept; a URDF expresses the same idea as a massless "
                "link on a fixed joint, which is read as an ordinary body."
            )
        return sites_from_mjcf(cfg.mjcf_path)

    def find_body_index(self, body_name: str) -> int: ...  # noqa: D102 — from RobotData

    @property
    def body_pos_w_all(self) -> Tensor: ...  # noqa: D102

    @property
    def body_quat_w_all(self) -> Tensor: ...  # noqa: D102

    @property
    def body_lin_vel_w_all(self) -> Tensor: ...  # noqa: D102

    @property
    def body_ang_vel_w_all(self) -> Tensor: ...  # noqa: D102

    # ── Resolution ───────────────────────────────────────────────────

    def find_site_index(self, site_name: str) -> int:
        """Index of a named site within this entity's site table."""
        for idx, frame in enumerate(self._site_frames):
            if frame.name == site_name:
                return idx
        available = [f.name for f in self._site_frames if not f.name.startswith("__unnamed")]
        raise KeyError(
            f"No site named {site_name!r} on this entity. Declared sites: {available or 'none'}. "
            "Sites come from the MJCF; an entity built from a URDF has none."
        )

    def _site_body_rows(self, site_ids: Tensor | list[int]) -> tuple[Tensor, Tensor]:
        """Parent-body rows and local offsets for the given site ids."""
        ids = site_ids.tolist() if isinstance(site_ids, Tensor) else list(site_ids)
        frames = [self._site_frames[int(i)] for i in ids]
        body_rows = torch.tensor(
            [self.find_body_index(f.body_name) for f in frames],
            device=self.body_pos_w_all.device,
            dtype=torch.long,
        )
        local = torch.tensor(
            [f.local_pos for f in frames],
            device=self.body_pos_w_all.device,
            dtype=self.body_pos_w_all.dtype,
        )
        return body_rows, local

    def _site_local_quats(self, site_ids: Tensor | list[int]) -> Tensor:
        """Site rotations relative to their parents, for the given ids."""
        ids = site_ids.tolist() if isinstance(site_ids, Tensor) else list(site_ids)
        return torch.tensor(
            [self._site_frames[int(i)].local_quat_wxyz for i in ids],
            device=self.body_quat_w_all.device,
            dtype=self.body_quat_w_all.dtype,
        )

    # ── Reads ────────────────────────────────────────────────────────

    def site_pos_w_by_ids(self, site_ids: Tensor) -> Tensor:
        """World positions of the sites at ``site_ids``. ``(num_envs, n, 3)``."""
        body_rows, local = self._site_body_rows(site_ids)
        return site_pos_w(
            self.body_pos_w_all[:, body_rows],
            self.body_quat_w_all[:, body_rows],
            local,
        )

    def site_lin_vel_w_by_ids(self, site_ids: Tensor) -> Tensor:
        """World linear velocities of the sites at ``site_ids``. ``(num_envs, n, 3)``."""
        body_rows, local = self._site_body_rows(site_ids)
        return site_lin_vel_w(
            self.body_lin_vel_w_all[:, body_rows],
            self.body_ang_vel_w_all[:, body_rows],
            self.body_quat_w_all[:, body_rows],
            local,
        )

    def site_quat_w_by_ids(self, site_ids: Tensor) -> Tensor:
        """World orientations of the sites at ``site_ids``. ``(num_envs, n, 4)`` wxyz."""
        body_rows, _ = self._site_body_rows(site_ids)
        return site_quat_w(self.body_quat_w_all[:, body_rows], self._site_local_quats(site_ids))

    def site_quat_w(self, names: list[str]) -> Tensor:
        """World orientations of the named sites, ordered by ``names``."""
        return self.site_quat_w_by_ids([self.find_site_index(n) for n in names])

    def site_pos_w(self, names: list[str]) -> Tensor:
        """World positions of the named sites, ordered by ``names``."""
        return self.site_pos_w_by_ids([self.find_site_index(n) for n in names])

    def site_lin_vel_w(self, names: list[str]) -> Tensor:
        """World linear velocities of the named sites, ordered by ``names``."""
        return self.site_lin_vel_w_by_ids([self.find_site_index(n) for n in names])
