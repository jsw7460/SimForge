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

* ``secondary is None``          → every link counts
* ``secondary.entity == <name>`` → only that entity's links count
* ``secondary.entity == "self"`` → only the primary entity's links count
* ``secondary.entity == "terrain"`` → only the terrain links count
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
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode='subtree' "
                "is not supported (mjlab-only)."
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
            self._counterpart_links = torch.arange(
                sec_entity.link_start, sec_entity.link_end, dtype=torch.long, device=self.device
            )

        # ---- per-substep capture rings (newest-first, native layout) --
        h = cfg.history_length
        self._found_hist = torch.zeros(self.num_envs, h, self._num_primary, dtype=torch.bool, device=self.device)
        self._force_hist = torch.zeros(self.num_envs, h, self._num_primary, 3, device=self.device)

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
        # World-frame signed sum: the collider force applies to side b;
        # side a receives the opposite sign.
        f_world = torch.einsum("ncp,nci->npi", pmask_b.float() - pmask_a.float(), force)
        quats = self._reader.links_quat(self._entity)[:, self._primary_local]
        f_local = inv_transform_by_quat(f_world, quats)

        # torch.roll allocates a fresh tensor, so frames handed out by the
        # read paths below stay valid snapshots after later captures.
        self._found_hist = torch.roll(self._found_hist, 1, dims=1)
        self._found_hist[:, 0] = found
        self._force_hist = torch.roll(self._force_hist, 1, dims=1)
        self._force_hist[:, 0] = f_local

    # ------------------------------------------------------------------
    # reads (captured frames only)
    # ------------------------------------------------------------------

    def read_found(self) -> torch.Tensor:
        """(num_envs, N) bool — newest captured ``found`` frame."""
        return self._found_hist[:, 0]

    def read_force(self) -> torch.Tensor:
        """(num_envs, N, 3) — newest captured link-local net contact force."""
        return self._force_hist[:, 0]

    def compute(self) -> ContactSensorData:
        return ContactSensorData(
            found=self.read_found(),
            force=self.read_force(),
            tracked_names=self._tracked_names,
        )

    def compute_history(self) -> torch.Tensor:
        """(num_envs, N, H, 3) captured contact-force history, newest-first."""
        return self._force_hist.permute(0, 2, 1, 3)
