"""Genesis contact sensor — simulator-agnostic ``ContactSensorCfg`` backend.

Contact-list implementation. Instead of per-link native sensors (one
``gs.sensors.Contact`` + ``gs.sensors.ContactForce`` pair per primary
link, each updated inside ``scene.step`` on every substep — a cost that
scales with sensor count and dominated the Genesis step time for
many-link groups), every group is computed from ONE batched read of the
rigid solver's global contact list per substep
(``collider.get_contacts`` + the live ``n_contacts`` counter), shared
across all groups through the per-step read cache.

The computation reproduces the native sensors' values exactly — verified
bit-exact for ``found`` and to float32 sum-order tolerance for ``force``
by ``scripts/diag/genesis_contact_list_parity_diag.py`` (standalone
native-vs-list scene) on every genesis preset's group layout:

* ``found``: an unfiltered (primary, counterpart) pair exists in the
  contact list.
* ``force``: sum of signed pair forces on the primary side (``-f`` when
  the primary link is contact side a, ``+f`` on side b), rotated into
  the link LOCAL frame with the link quaternion at capture time —
  matching ``genesis/engine/sensors/contact_force.py``.
* Rows past each env's live ``n_contacts`` counter are stale on the
  zero-copy path and are masked out via the counter, never via field
  sentinels.

Timing semantics also match the native sensors: one frame is captured
per substep (``ContactManager.advance`` drives :meth:`capture_substep`)
into per-group rings, and ``read_found`` / ``read_force`` /
``compute_history`` serve captured frames only. Values therefore change
when physics steps, not when a reset teleports state — a recompute-on-
read design would rotate the stale contact list with post-reset link
quaternions.

The agnostic config's ``secondary`` resolves to a POSITIVE counterpart
link set (the native path inverted it into a blacklist):

* ``secondary is None``            → every link counts
* ``mode="entity"``                → every link of ``secondary.entity``
* ``mode="body"``                  → only the links ``pattern`` names
* ``secondary.entity == "self"``   → scoped to the primary entity
* ``secondary.entity == "terrain"``→ scoped to the terrain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat
from genesis.utils.misc import qd_to_torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.genesis.robot_data import _per_step_read
from rlworld.rl.utils import entity_utils as eu

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@dataclass
class ContactSensorData:
    """Output of a Genesis contact sensor computation."""

    found: torch.Tensor
    """(num_envs, num_primary_links) bool — contact exists."""
    force: torch.Tensor
    """(num_envs, num_primary_links, 3) — net contact force on each primary link."""
    tracked_names: list[str]
    """Primary link names in order."""


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether ``name`` matches any of ``patterns`` (regex search)."""
    return any(re.search(p, name) for p in patterns)


class GenesisContactListReader:
    """One shared, generation-cached read of the solver's contact list.

    All contact groups of an env consume the same collider state, so the
    expensive parts — the ``get_contacts`` readback and the per-entity
    link-quaternion read — are cached on the env's cache generation
    (bumped once per substep by ``GenesisEnv._step_physics``) and paid
    once per substep regardless of the number of groups.
    """

    def __init__(self, env: GenesisEnv):
        self._env = env
        self._read_cache: dict[str, tuple[int, object]] = {}
        self._quat_generation = -1
        self._quat_by_entity: dict[int, torch.Tensor] = {}

    @_per_step_read
    def raw(self):
        """``(link_a, link_b, force, row_valid)`` for the current substep.

        ``link_a``/``link_b``: (num_envs, C) global link indices;
        ``force``: (num_envs, C, 3) world-frame contact force (applied to
        side b; side a receives ``-force``); ``row_valid``: (num_envs, C)
        bool masking each env's live rows via the collider's
        ``n_contacts`` counter (rows beyond it are stale on the zero-copy
        path).
        """
        solver = self._env.scene_manager.scene.sim.rigid_solver
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b, force = cd["link_a"], cd["link_b"], cd["force"]
        n_live = qd_to_torch(solver.collider._collider_state.n_contacts, copy=False)
        row_valid = torch.arange(link_a.shape[1], device=link_a.device)[None, :] < n_live[:, None]
        return link_a, link_b, force, row_valid

    def links_quat(self, entity) -> torch.Tensor:
        """(num_envs, entity_n_links, 4) link quaternions, generation-cached per entity."""
        gen = self._env._cache_generation
        if self._quat_generation != gen:
            self._quat_generation = gen
            self._quat_by_entity.clear()
        quat = self._quat_by_entity.get(entity.idx)
        if quat is None:
            quat = entity.get_links_quat()
            self._quat_by_entity[entity.idx] = quat
        return quat


class GenesisContactSensor:
    """Runtime contact sensor backing one ``ContactManager`` group.

    Fully resolved at construction (post-build is fine — no native scene
    objects are created). ``ContactManager.advance`` calls
    :meth:`capture_substep` once per physics substep; all read paths
    serve the captured rings.
    """

    def __init__(self, env: GenesisEnv, cfg: ContactSensorCfg, reader: GenesisContactListReader):
        if not isinstance(cfg, ContactSensorCfg):
            raise TypeError(f"GenesisContactSensor expects a ContactSensorCfg, got {type(cfg).__name__}")

        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self._reader = reader

        # ---- backend support matrix ---------------------------------
        if cfg.primary.mode == "subtree":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode='subtree' is not supported (mjlab-only)."
            )
        if cfg.primary.mode == "geom":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode='geom' is not "
                "yet supported (the contact-list reader matches on link indices); use mode='body'."
            )
        if cfg.primary.mode != "body":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode={cfg.primary.mode!r}; "
                "only 'body' is supported."
            )
        if cfg.reduce != "netforce":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} reduce={cfg.reduce!r}; only "
                "'netforce' (sum of all contacts into one net wrench) is supported."
            )
        if cfg.num_slots != 1:
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} num_slots={cfg.num_slots}; only "
                "num_slots=1 is supported."
            )
        unsupported_fields = set(cfg.fields) - {"found", "force"}
        if unsupported_fields:
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} fields={cfg.fields}; only "
                f"{{'found', 'force'}} are supported (got extra {sorted(unsupported_fields)})."
            )
        if cfg.history_length < env.decimation:
            raise ValueError(
                f"Genesis ContactSensorCfg {cfg.name!r}: history_length={cfg.history_length} < "
                f"decimation={env.decimation}. History consumers (contact_force_history) expect "
                "every substep of the last control step to be retained; set "
                "history_length=decimation (as every genesis preset does)."
            )

        # ---- resolve primary links ----------------------------------
        primary_entity_name = cfg.primary.entity or "robot"
        entity = env.scene_manager[primary_entity_name]
        self._entity = entity
        self._primary_entity_name = primary_entity_name

        primary_patterns = (
            (cfg.primary.pattern,) if isinstance(cfg.primary.pattern, str) else tuple(cfg.primary.pattern)
        )
        link_ids_local, link_names = eu.find_links(
            entity, list(primary_patterns), global_ids=False, preserve_order=True
        )
        if cfg.primary.exclude:
            kept = [
                (lid, lname)
                for lid, lname in zip(link_ids_local, link_names)
                if not _matches_any(lname, cfg.primary.exclude)
            ]
            link_ids_local = [lid for lid, _ in kept]
            link_names = [lname for _, lname in kept]
        if not link_names:
            raise ValueError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary pattern "
                f"{cfg.primary.pattern!r} (entity {primary_entity_name!r}, after exclude "
                f"{cfg.primary.exclude}) matched no links."
            )

        self._link_ids_local: list[int] = link_ids_local
        self._tracked_names: list[str] = link_names
        self._num_primary = len(link_names)
        self._primary_local = torch.tensor(link_ids_local, dtype=torch.long, device=self.device)
        self._primary_links = torch.tensor(
            [entity.link_start + lid for lid in link_ids_local], dtype=torch.long, device=self.device
        )

        # ---- resolve secondary → counterpart link set ----------------
        n_links = env.scene_manager.scene.sim.rigid_solver.n_links
        sec = cfg.secondary
        if sec is None:
            # Every link counts; skip the counterpart membership test entirely.
            self._counterpart_is_all = True
            self._counterpart_links = torch.arange(n_links, dtype=torch.long, device=self.device)
        else:
            if not sec.entity:
                # secondary.entity is None/"" but a literal pattern was given — out of scope.
                raise NotImplementedError(
                    f"Genesis backend: ContactSensorCfg {cfg.name!r} secondary with a literal "
                    "pattern (no entity scope) is not supported; use secondary.entity=<name> "
                    "or secondary.entity='self'."
                )
            # ``"self"`` keeps only intra-primary-entity contacts; ``"terrain"``
            # is a sentinel for the ground (owned by ``TerrainImporter``, not in
            # ``scene_manager.entities``); everything else looks up a named entity.
            if sec.entity == "self":
                sec_entity = entity
            elif sec.entity == "terrain":
                sec_entity = env.scene_manager.terrain.entity
            else:
                sec_entity = env.scene_manager[sec.entity]
            self._counterpart_is_all = False
            if sec.mode == "entity":
                # Whole entity: its links are contiguous in the solver.
                counterpart = list(range(sec_entity.link_start, sec_entity.link_end))
            elif sec.mode == "body":
                # Named links only. The counterpart set is a membership test, so
                # any subset works — this used to take the entity's whole range
                # and ignore the pattern, which silently widened "the left jaw"
                # into "any part of the tool" while Newton and mjlab honoured it.
                sec_patterns = (sec.pattern,) if isinstance(sec.pattern, str) else list(sec.pattern)
                link_ids, link_names = eu.find_links(
                    sec_entity, list(sec_patterns), global_ids=True, preserve_order=True
                )
                if sec.exclude:
                    kept = [
                        (lid, lname) for lid, lname in zip(link_ids, link_names) if not _matches_any(lname, sec.exclude)
                    ]
                    link_ids = [lid for lid, _ in kept]
                    link_names = [lname for _, lname in kept]
                if not link_names:
                    raise ValueError(
                        f"Genesis backend: ContactSensorCfg {cfg.name!r} secondary pattern "
                        f"{sec.pattern!r} (entity {sec.entity!r}, after exclude {sec.exclude}) "
                        f"matched no links."
                    )
                counterpart = link_ids
            else:
                raise NotImplementedError(
                    f"Genesis backend: ContactSensorCfg {cfg.name!r} secondary.mode={sec.mode!r}; "
                    f"Genesis resolves contacts by link, so only 'body' (named links) and 'entity' "
                    f"(every link of the entity) are supported."
                )
            self._counterpart_links = torch.tensor(counterpart, dtype=torch.long, device=self.device)

        # ---- per-substep capture rings (newest-first, native layout) --
        # ``fields`` is a declaration of what anything reads, and mjlab's
        # sensor already extracts only what it lists. Honouring it here
        # too is worth more than it looks: the force is a signed
        # accumulation over the contact list followed by a rotation into
        # each link's frame, and on a locomotion preset whose rewards,
        # observations and terminations all read only ``is_contact`` that
        # was two thirds of this sensor's per-substep cost — computed,
        # stored, and never looked at.
        self._name = cfg.name
        self._track_force = "force" in cfg.fields
        h = cfg.history_length
        self._found_hist = torch.zeros(self.num_envs, h, self._num_primary, dtype=torch.bool, device=self.device)
        self._force_hist = (
            torch.zeros(self.num_envs, h, self._num_primary, 3, device=self.device) if self._track_force else None
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def tracked_names(self) -> list[str]:
        return self._tracked_names

    # ------------------------------------------------------------------
    # per-substep capture
    # ------------------------------------------------------------------

    def capture_substep(self) -> None:
        """Compute this substep's ``found``/``force`` frame from the shared
        contact-list read and push it onto the rings (newest-first).

        Called by ``ContactManager.advance`` after every physics substep;
        the underlying collider read is generation-cached, so all groups
        of the env share one readback per substep.
        """
        link_a, link_b, force, row_valid = self._reader.raw()

        on_p_a = link_a.unsqueeze(-1) == self._primary_links  # (B, C, P)
        on_p_b = link_b.unsqueeze(-1) == self._primary_links
        if self._counterpart_is_all:
            a_ok = row_valid
            b_ok = row_valid
        else:
            a_ok = row_valid & (link_a.unsqueeze(-1) == self._counterpart_links).any(-1)
            b_ok = row_valid & (link_b.unsqueeze(-1) == self._counterpart_links).any(-1)
        # A pair counts for a primary link on side a iff the OTHER side (b)
        # is an allowed counterpart, and vice versa.
        pmask_a = on_p_a & b_ok.unsqueeze(-1)
        pmask_b = on_p_b & a_ok.unsqueeze(-1)

        found = (pmask_a | pmask_b).any(1)  # (B, P)

        f_local = None
        if self._track_force:
            # World-frame signed sum: the collider force applies to side b;
            # side a receives the opposite sign.
            f_world = torch.einsum("ncp,nci->npi", pmask_b.float() - pmask_a.float(), force)
            quats = self._reader.links_quat(self._entity)[:, self._primary_local]
            f_local = inv_transform_by_quat(f_world, quats)
        self.push_frame(found, f_local)

    # ------------------------------------------------------------------
    # reads (captured frames only)
    # ------------------------------------------------------------------

    def read_found(self) -> torch.Tensor:
        """(num_envs, N) bool — newest captured ``found`` frame."""
        return self._found_hist[:, 0]

    def push_frame(self, found: torch.Tensor, f_local: torch.Tensor | None) -> None:
        """Push one externally computed frame onto the rings.

        Used by :class:`GenesisContactBatch`, which computes every
        group's frame in one fused pass and hands each sensor its slice.
        ``torch.roll`` allocates a fresh tensor, so frames handed out by
        the read paths stay valid snapshots after later captures.
        """
        self._found_hist = torch.roll(self._found_hist, 1, dims=1)
        self._found_hist[:, 0] = found
        if self._track_force:
            self._force_hist = torch.roll(self._force_hist, 1, dims=1)
            self._force_hist[:, 0] = f_local

    def read_force(self) -> torch.Tensor:
        """(num_envs, N, 3) — newest captured link-local net contact force."""
        self._require_force()
        return self._force_hist[:, 0]

    def _require_force(self) -> None:
        """Refuse a read of a force this sensor was told not to track."""
        if not self._track_force:
            raise RuntimeError(
                f"Genesis contact sensor {self._name!r} was configured with fields that omit "
                '"force", so no force was computed. Add "force" to the ContactSensorCfg\'s '
                "fields to read it."
            )

    def compute(self) -> ContactSensorData:
        return ContactSensorData(
            found=self.read_found(),
            force=self.read_force(),
            tracked_names=self._tracked_names,
        )

    def compute_history(self) -> torch.Tensor:
        """(num_envs, N, H, 3) captured contact-force history, newest-first."""
        self._require_force()
        return self._force_hist.permute(0, 2, 1, 3)


class GenesisContactBatch:
    """One fused per-substep capture for every contact group.

    Each group used to run its own primary-link masks, counterpart test,
    ``einsum`` and frame rotation against the same shared contact-list
    read — with G groups that is G of everything, all launch-bound small
    kernels (measured ~0.24 ms per group per substep on go2_gait @4096).
    The batch concatenates every group's primary links into one column
    axis and runs each stage once, then hands each sensor its column
    slice via :meth:`GenesisContactSensor.push_frame`.

    Per-column math is identical to ``capture_substep``'s: ``found`` is
    bit-identical, ``force`` can differ only by the reduction scheduling
    inside the single larger einsum (float sum order).
    """

    def __init__(self, env: GenesisEnv, reader: GenesisContactListReader, sensors: list[GenesisContactSensor]):
        self._reader = reader
        self._sensors = sensors
        device = env.device

        self._all_primary = torch.cat([s._primary_links for s in sensors])
        slices, start = [], 0
        for s in sensors:
            slices.append(slice(start, start + s._num_primary))
            start += s._num_primary
        self._slices = slices

        # Distinct counterpart sets: most presets point every group at the
        # same counterpart (the terrain), so the membership test collapses
        # to one. ``None`` in the list means "every link counts".
        keys: list[tuple] = []
        self._distinct_counterparts: list[torch.Tensor | None] = []
        col_set: list[int] = []
        for s in sensors:
            key = ("all",) if s._counterpart_is_all else tuple(s._counterpart_links.tolist())
            if key not in keys:
                keys.append(key)
                self._distinct_counterparts.append(None if s._counterpart_is_all else s._counterpart_links)
            col_set.extend([keys.index(key)] * s._num_primary)
        self._uniform_counterpart = len(self._distinct_counterparts) == 1
        self._col_set = torch.tensor(col_set, dtype=torch.long, device=device)

        # Force is only computed for the columns whose sensor tracks it
        # (the fields declaration — see the ring comment in the sensor).
        self._force_sensors = [s for s in sensors if s._track_force]
        fcols: list[int] = []
        for s, sl in zip(sensors, slices):
            if s._track_force:
                fcols.extend(range(sl.start, sl.stop))
        self._force_cols = torch.tensor(fcols, dtype=torch.long, device=device) if fcols else None
        self._force_all_cols = self._force_cols is not None and len(fcols) == start

    def capture_substep(self) -> None:
        """Compute this substep's frames for every group and push them."""
        link_a, link_b, force, row_valid = self._reader.raw()

        on_a = link_a.unsqueeze(-1) == self._all_primary  # (B, C, P_total)
        on_b = link_b.unsqueeze(-1) == self._all_primary

        a_oks, b_oks = [], []
        for cp in self._distinct_counterparts:
            if cp is None:
                a_oks.append(row_valid)
                b_oks.append(row_valid)
            else:
                a_oks.append(row_valid & (link_a.unsqueeze(-1) == cp).any(-1))
                b_oks.append(row_valid & (link_b.unsqueeze(-1) == cp).any(-1))
        if self._uniform_counterpart:
            a_ok_cols = a_oks[0].unsqueeze(-1)
            b_ok_cols = b_oks[0].unsqueeze(-1)
        else:
            a_ok_cols = torch.stack(a_oks, dim=-1)[:, :, self._col_set]
            b_ok_cols = torch.stack(b_oks, dim=-1)[:, :, self._col_set]

        # A pair counts for a primary link on side a iff the OTHER side (b)
        # is an allowed counterpart, and vice versa — same rule per column
        # as the single-group capture.
        pmask_a = on_a & b_ok_cols
        pmask_b = on_b & a_ok_cols
        found = (pmask_a | pmask_b).any(1)  # (B, P_total)

        f_local = None
        if self._force_cols is not None:
            if self._force_all_cols:
                sub_a, sub_b = pmask_a, pmask_b
            else:
                sub_a = pmask_a[:, :, self._force_cols]
                sub_b = pmask_b[:, :, self._force_cols]
            f_world = torch.einsum("ncp,nci->npi", sub_b.float() - sub_a.float(), force)
            quats = torch.cat(
                [self._reader.links_quat(s._entity)[:, s._primary_local] for s in self._force_sensors],
                dim=1,
            )
            f_local = inv_transform_by_quat(f_world, quats)

        fstart = 0
        for s, sl in zip(self._sensors, self._slices):
            if s._track_force:
                n = s._num_primary
                s.push_frame(found[:, sl], f_local[:, fstart : fstart + n])
                fstart += n
            else:
                s.push_frame(found[:, sl], None)
