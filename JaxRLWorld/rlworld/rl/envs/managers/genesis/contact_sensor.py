"""Genesis contact sensor — simulator-agnostic ``ContactSensorCfg`` backend.

Backed by Genesis's first-class link sensors (``gs.sensors.Contact`` +
``gs.sensors.ContactForce``), one pair per primary link. Both sensors
carry the same native ``filter_link_idx`` blacklist, so ``found`` and
``force`` are counterpart-filtered consistently, and both keep a substep
ring buffer when ``cfg.history_length > 0``. Native sensors must be
added to the scene before ``scene.build()`` (``scene.add_sensor`` is
``@gs.assert_unbuilt``), so :meth:`create_native_sensors` is invoked
from the env's pre-build phase (``GenesisEnv._build_scene``) rather than
from ``ContactManager.register_sensor`` (which runs post-build).

The agnostic config's positive ``secondary`` is inverted into the
blacklist:

* ``secondary is None``          → no filter (every contact counts)
* ``secondary.entity == <name>`` → blacklist every link not in that entity
* ``secondary.entity == "self"`` → blacklist every link not in the primary
  entity (keeps robot↔robot contacts only)
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
    post-build). The actual Genesis sensor objects are created via
    :meth:`create_native_sensors`, which must run while the scene is
    still unbuilt.
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

        # ---- resolve secondary → native filter_link_idx (BLACKLIST) -------
        # ``gs.sensors.Contact`` / ``ContactForce`` take a ``filter_link_idx``
        # blacklist: contacts whose *other* participant is in this list are
        # ignored. Invert the agnostic config's positive ``secondary``:
        #   - secondary is None          → no filter
        #   - secondary.entity == <name> → blacklist every link not in that entity
        #   - secondary.entity == "self" → blacklist every link not in the primary
        #     entity (so only robot↔robot contacts survive)
        self._filter_link_idx: tuple[int, ...] = ()
        sec = cfg.secondary
        if sec is not None:
            if not sec.entity:
                # secondary.entity is None/"" but a literal pattern was given — out of scope.
                raise NotImplementedError(
                    f"Genesis backend: ContactSensorCfg {cfg.name!r} secondary with a literal "
                    "pattern (no entity scope) is not supported; use secondary.entity=<name> "
                    "or secondary.entity='self'."
                )
            # ``"self"`` keeps only intra-primary contacts; ``"terrain"``
            # is a sentinel for the ground (owned by ``TerrainImporter``,
            # not in ``scene_manager.entities``); everything else looks up
            # a named entity in the dict.
            if sec.entity == "self":
                sec_entity = entity
            elif sec.entity == "terrain":
                sec_entity = env.scene_manager.terrain.entity
            else:
                sec_entity = env.scene_manager[sec.entity]
            sec_links = set(range(sec_entity.link_start, sec_entity.link_end))
            n_links = env.scene_manager.scene.sim.rigid_solver.n_links
            self._filter_link_idx = tuple(sorted(set(range(n_links)) - sec_links))

        # Native sensor objects, created later in create_native_sensors().
        self._contact_sensors: list = []
        self._force_sensors: list = []
        self._native_created = False

    # ------------------------------------------------------------------
    # native sensor creation (must be called while scene is unbuilt)
    # ------------------------------------------------------------------

    def create_native_sensors(self) -> None:
        """Add ``gs.sensors.Contact`` / ``gs.sensors.ContactForce`` to the scene.

        Must be called before ``scene.build()`` — ``scene.add_sensor`` is
        ``@gs.assert_unbuilt`` — so the env's pre-build phase
        (``GenesisEnv._build_scene``) invokes it.
        """
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
                    filter_link_idx=self._filter_link_idx,
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
            return t[:, 0, :]
        return t

    def compute(self) -> ContactSensorData:
        has_history = self.cfg.history_length > 0
        n = self.num_envs

        found_cols: list[torch.Tensor] = []
        force_cols: list[torch.Tensor] = []
        for cs, fs in zip(self._contact_sensors, self._force_sensors):
            c = self._current_frame(cs.read_ground_truth(), has_history, n)  # Contact: payload dim 1
            f = self._current_frame(fs.read_ground_truth(), has_history, n)  # ContactForce: payload dim 3
            found_cols.append(c[..., 0] != 0)  # (n,)
            force_cols.append(f)  # (n, 3)

        found = torch.stack(found_cols, dim=1)  # (num_envs, N) bool
        force = torch.stack(force_cols, dim=1)  # (num_envs, N, 3)
        # ``Contact`` and ``ContactForce`` carry the same ``filter_link_idx``
        # blacklist, so ``force`` is already counterpart-filtered (matches ``found``).
        return ContactSensorData(found=found, force=force, tracked_names=self._tracked_names)

    # ------------------------------------------------------------------
    # substep history (only when history_length > 0)
    # ------------------------------------------------------------------

    def compute_history(self) -> torch.Tensor | None:
        """Return ``(num_envs, N, H, 3)`` counterpart-filtered contact-force history, or ``None``.

        Newest-first along the H axis (Genesis ring layout). The only
        consumer (``penalize_contact_force_count``) reduces over H with
        ``.any(dim=2)``, so the order does not matter.
        """
        if self.cfg.history_length <= 0:
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
