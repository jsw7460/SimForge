"""Genesis contact sensor — simulator-agnostic ``ContactSensorCfg`` backend.

Two code paths, chosen at construction time from the resolved
``secondary``:

``native``
    Uses Genesis's first-class link sensors (``gs.sensors.Contact`` +
    ``gs.sensors.ContactForce``), one pair per primary link.
    ``Contact`` carries a native blacklist ``filter_link_idx`` so the
    ``found`` flag is already counterpart-filtered; ``ContactForce`` has
    no filter, so the per-link force is masked by ``found`` after the
    fact (see :meth:`GenesisContactSensor.compute`). Native sensors keep
    a ring buffer when ``cfg.history_length > 0`` — that is the only
    reason to prefer this path. **Native sensors must be added to the
    scene before ``scene.build()``**, so :meth:`create_native_sensors`
    is invoked from the env's pre-build phase
    (``GenesisEnv._build_scene``) rather than from
    ``ContactManager.register_sensor`` (which runs post-build).

``get_contacts``
    Used only when ``secondary.entity == "self"`` (self-collision):
    Genesis has no native "contact with another link of the same
    entity" filter, so we keep the legacy ``entity.get_contacts(...)``
    query + per-primary-link force aggregation. This path is a runtime
    query, so it works post-build and needs no pre-registration. It has
    no substep history (``compute_history()`` returns ``None``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import genesis as gs
import torch

from rlworld.rl.configs.sensors import ContactSensorCfg
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


class GenesisContactSensor:
    """Runtime contact sensor backing one ``ContactManager`` group.

    Resolution + validation happen in ``__init__`` (safe pre- or
    post-build). For the ``native`` path the actual Genesis sensor
    objects are created later via :meth:`create_native_sensors`, which
    must run while the scene is still unbuilt.
    """

    def __init__(self, env: GenesisEnv, cfg: ContactSensorCfg):
        if not isinstance(cfg, ContactSensorCfg):
            raise TypeError(f"GenesisContactSensor expects a ContactSensorCfg, got {type(cfg).__name__}")

        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs

        # ---- backend support matrix ---------------------------------
        if cfg.primary.mode == "subtree":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode='subtree' "
                "is not supported (mjlab-only)."
            )
        if cfg.primary.mode == "geom":
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} primary.mode='geom' is not "
                "yet supported (Genesis native link sensors are link-indexed, not "
                "geom-indexed); use mode='body'."
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

        # ---- resolve secondary → choose code path -------------------
        # ``native``: Contact sensor's ``filter_link_idx`` is a BLACKLIST.
        #   - secondary is None  → no filter (everything counts).
        #   - secondary.entity == <name> (e.g. "base_entity") → only
        #     contacts with that entity should count. Native has no
        #     whitelist, so invert: blacklist every link NOT belonging to
        #     the secondary entity.
        # ``get_contacts``: secondary.entity == "self" only.
        self._mode: str
        self._filter_link_idx: tuple[int, ...] = ()
        # get_contacts-path state (only used when _mode == "get_contacts")
        self._with_entity = None
        self._primary_link_ids_global: torch.Tensor | None = None

        sec = cfg.secondary
        if sec is None:
            self._mode = "native"
            self._filter_link_idx = ()
        elif sec.entity == "self":
            self._mode = "get_contacts"
            self._with_entity = entity
            # Global (solver) link ids for the resolved primary links, in the
            # same order/count as ``self._tracked_names``.
            global_ids = [entity.link_start + lid for lid in self._link_ids_local]
            self._primary_link_ids_global = torch.tensor(global_ids, dtype=torch.int64, device=self.device)
        elif sec.entity:
            self._mode = "native"
            sec_entity = env.scene_manager[sec.entity]
            sec_links = set(range(sec_entity.link_start, sec_entity.link_end))
            n_links = env.scene_manager.scene.sim.rigid_solver.n_links
            self._filter_link_idx = tuple(sorted(set(range(n_links)) - sec_links))
        else:
            # secondary.entity is None/"" but a literal pattern was given — out of scope.
            raise NotImplementedError(
                f"Genesis backend: ContactSensorCfg {cfg.name!r} secondary with a literal "
                "pattern (no entity scope) is not supported yet; use secondary.entity=<name> "
                "or secondary.entity='self'."
            )

        # Native sensor objects, created later in create_native_sensors().
        self._contact_sensors: list = []
        self._force_sensors: list = []
        self._native_created = False

        # Pre-allocated zero buffers for empty / no-contact early returns.
        self._zeros_found = torch.zeros(self.num_envs, self._num_primary, dtype=torch.bool, device=self.device)
        self._zeros_force = torch.zeros(self.num_envs, self._num_primary, 3, device=self.device)

    # ------------------------------------------------------------------
    # native sensor creation (must be called while scene is unbuilt)
    # ------------------------------------------------------------------

    def create_native_sensors(self) -> None:
        """Add ``gs.sensors.Contact`` / ``gs.sensors.ContactForce`` to the scene.

        Only meaningful for ``_mode == 'native'``; a no-op otherwise.
        Must be called before ``scene.build()`` — ``scene.add_sensor`` is
        ``@gs.assert_unbuilt``.
        """
        if self._mode != "native":
            return
        if self._native_created:
            return

        scene = self.env.scene_manager.scene
        if scene.is_built:
            raise RuntimeError(
                f"Genesis backend: ContactSensorCfg {self.cfg.name!r}: native contact sensors "
                "must be created before scene.build(); the scene is already built. Wire "
                "GenesisContactSensor.create_native_sensors() into the env's pre-build phase."
            )

        entity_idx = self._entity.idx
        hist = self.cfg.history_length
        for l in self._link_ids_local:
            contact_sensor = scene.add_sensor(
                gs.sensors.Contact(
                    entity_idx=entity_idx,
                    link_idx_local=l,
                    filter_link_idx=self._filter_link_idx,
                    history_length=hist,
                )
            )
            force_sensor = scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=entity_idx,
                    link_idx_local=l,
                    history_length=hist,
                )
            )
            self._contact_sensors.append(contact_sensor)
            self._force_sensors.append(force_sensor)
        self._native_created = True

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def tracked_names(self) -> list[str]:
        return self._tracked_names

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    @staticmethod
    def _current_frame(t: torch.Tensor, has_history: bool, num_envs: int) -> torch.Tensor:
        """Slice the most-recent frame out of a Genesis sensor reading.

        Native sensor ``read_ground_truth()`` returns, for a per-element
        payload of dim ``D``:
          - ``history > 0``:  ``(num_envs, H, D)``  (or ``(H, D)`` if num_envs==0) — newest first.
          - ``history == 0``: ``(num_envs, D)``     (or ``(D,)`` if num_envs==0).
        This returns ``(num_envs, D)`` (or ``(D,)`` if num_envs==0): the newest frame.
        """
        if has_history:
            # newest-first along the history axis → index 0
            return t[..., 0, :] if num_envs == 0 else t[:, 0, :]
        return t

    def compute(self) -> ContactSensorData:
        if self._mode == "native":
            return self._compute_native()
        return self._compute_get_contacts()

    def _compute_native(self) -> ContactSensorData:
        if not self._native_created:
            raise RuntimeError(
                f"Genesis backend: ContactSensorCfg {self.cfg.name!r}: native sensors were "
                "never created (create_native_sensors() not called before scene.build())."
            )
        has_history = self.cfg.history_length > 0
        n = self.num_envs

        found_cols: list[torch.Tensor] = []
        force_cols: list[torch.Tensor] = []
        for cs, fs in zip(self._contact_sensors, self._force_sensors):
            # Contact: per-element payload dim 1 → current frame (n, 1) / (1,)
            c = self._current_frame(cs.read_ground_truth(), has_history, n)
            # ContactForce: per-element payload dim 3 → current frame (n, 3) / (3,)
            f = self._current_frame(fs.read_ground_truth(), has_history, n)
            if n == 0:
                c = c.unsqueeze(0)  # (1, 1)
                f = f.unsqueeze(0)  # (1, 3)
            found_cols.append(c[..., 0] != 0)  # (n,)
            force_cols.append(f)  # (n, 3)

        found = torch.stack(found_cols, dim=1)  # (num_envs, N) bool
        force = torch.stack(force_cols, dim=1)  # (num_envs, N, 3)

        # ContactForce has no counterpart filter, so its force is the
        # *total* force on the link (ground + self + anything). Contact's
        # ``found`` IS counterpart-filtered, so mask the force by it.
        # Limitation: a link simultaneously touching the secondary entity
        # AND something else still reports the combined force — accepted
        # for now (it only matters when ``found`` is true).
        force = force * found.unsqueeze(-1).to(force.dtype)

        return ContactSensorData(found=found, force=force, tracked_names=self._tracked_names)

    def _compute_get_contacts(self) -> ContactSensorData:
        """Legacy self-collision path via ``entity.get_contacts``."""
        raw = self._entity.get_contacts(
            with_entity=self._with_entity,
            exclude_self_contact=False,
        )
        if raw is None:
            return ContactSensorData(
                found=self._zeros_found.clone(),
                force=self._zeros_force.clone(),
                tracked_names=self._tracked_names,
            )

        link_a = raw["link_a"]
        link_b = raw["link_b"]
        force_a = raw["force_a"]
        force_b = raw["force_b"]
        valid_mask = raw.get("valid_mask")

        if link_a.numel() == 0:
            return ContactSensorData(
                found=self._zeros_found.clone(),
                force=self._zeros_force.clone(),
                tracked_names=self._tracked_names,
            )

        if link_a.dim() == 1:
            link_a = link_a.unsqueeze(0)
            link_b = link_b.unsqueeze(0)
            force_a = force_a.unsqueeze(0)
            force_b = force_b.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)

        primary_ids = self._primary_link_ids_global  # (N,)
        match_a = link_a.unsqueeze(-1) == primary_ids  # (B, C, N) — primary is link_a
        match_b = link_b.unsqueeze(-1) == primary_ids  # (B, C, N) — primary is link_b
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(-1)
            match_a = match_a & vm
            match_b = match_b & vm

        found = (match_a | match_b).any(dim=1)  # (B, N)
        force_from_a = torch.einsum("bci,bcn->bni", force_a, match_a.float())
        force_from_b = torch.einsum("bci,bcn->bni", force_b, match_b.float())
        force = force_from_a + force_from_b  # (B, N, 3)

        return ContactSensorData(found=found, force=force, tracked_names=self._tracked_names)

    # ------------------------------------------------------------------
    # substep history (only native + history_length > 0)
    # ------------------------------------------------------------------

    def compute_history(self) -> torch.Tensor | None:
        """Return ``(num_envs, N, H, 3)`` contact-force history, or ``None``.

        Newest-first along the H axis (Genesis ring layout). The only
        consumer (``penalize_contact_force_count``) reduces over H with
        ``.any(dim=2)``, so the order does not matter.
        """
        if self._mode != "native" or self.cfg.history_length <= 0:
            return None
        if not self._native_created:
            raise RuntimeError(
                f"Genesis backend: ContactSensorCfg {self.cfg.name!r}: native sensors were "
                "never created (create_native_sensors() not called before scene.build())."
            )
        n = self.num_envs
        cols: list[torch.Tensor] = []
        for fs in self._force_sensors:
            h = fs.read_ground_truth()  # (num_envs, H, 3) or (H, 3) when num_envs==0
            if n == 0:
                h = h.unsqueeze(0)  # (1, H, 3)
            cols.append(h)  # (n, H, 3)
        return torch.stack(cols, dim=1)  # (num_envs, N, H, 3)
