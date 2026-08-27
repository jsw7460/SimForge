from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.misc import qd_to_torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import (
    GenesisContactBatch,
    GenesisContactListReader,
    GenesisContactSensor,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Named-group contact manager for Genesis environments.

    Each :class:`rlworld.rl.configs.sensors.ContactSensorCfg` becomes a
    named group, backed by :class:`GenesisContactSensor` — computed from
    the rigid solver's global contact list (one shared
    ``collider.get_contacts`` read per substep via
    :class:`GenesisContactListReader`), not from native per-link scene
    sensors, so registration happens post-build like on the other
    backends.
    """

    def __init__(self, env: GenesisEnv):
        super().__init__(env=env)
        self._list_reader = GenesisContactListReader(env)
        self._sensors: dict[str, GenesisContactSensor] = {}
        # Fused per-substep capture across all groups; (re)built lazily on
        # the first advance after registration settles.
        self._batch: GenesisContactBatch | None = None

    def register_sensor(self, cfg: ContactSensorCfg) -> None:
        """Register a contact sensor config as a named group."""
        sensor = GenesisContactSensor(self.env, cfg, reader=self._list_reader)
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names, cfg.fields)
        self._batch = None

    def _capture_all(self) -> None:
        """One fused capture for every group (see GenesisContactBatch)."""
        if not self._sensors:
            return
        if self._batch is None:
            self._batch = GenesisContactBatch(self.env, self._list_reader, list(self._sensors.values()))
        self._batch.capture_substep()

    # -- per-substep capture + timing accumulation --

    def advance(self, dt: float) -> None:
        """Capture one contact frame for all groups from the shared list
        read, then run the base per-substep timing arithmetic on it.

        ``GenesisEnv._step_physics`` bumps the cache generation after
        every ``scene.step``, so the shared collider read is fresh
        exactly once per substep no matter how many groups exist.
        """
        self._capture_all()
        super().advance(dt=dt)

    # -- abstract impl --

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._sensors[group.name].read_force()

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        return self._sensors[group.name].compute_history()

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        """Contact bool = "an unfiltered pair exists in the solver's
        contact list" — the same solver-level binary signal Genesis's
        native ``gs.sensors.Contact`` reports (verified bit-exact by the
        contact-list parity diags), invariant to solver-iteration force
        jitter and aligned with mjlab's ``data.found > 0`` semantics.
        """
        return self._sensors[group.name].read_found()

    # -- post-reset refresh --

    def refresh_after_reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Recompute collision detection for the freshly reset poses.

        Genesis exposes no ``forward()``-equivalent that solves contact
        forces without integrating, but detection is a standalone
        geometric pass — so post-reset reads carry a fresh ``found`` for
        the new poses, while the force field (a constraint-solve
        product) is zeroed for the reset envs instead of leaking the
        pre-reset solve's values into the re-detected slots.
        """
        if not self._sensors:
            return
        solver = self.env.scene_manager.scene.sim.rigid_solver
        force = qd_to_torch(solver.collider._collider_state.contact_data.force, transpose=True, copy=False)
        # ``collider.clear()`` inside the detection kernel wipes the force
        # field for EVERY env; the non-reset envs must keep their last
        # solve's forces (re-detection from unchanged states reproduces
        # the same slot order), so snapshot and restore around it.
        saved_force = force.clone()
        solver._kernel_detect_collision()
        force.copy_(saved_force)
        if env_ids is None:
            force.zero_()
        else:
            force[env_ids] = 0.0
        # Detection replaced the collider state in place; drop the shared
        # list-reader cache so the capture below re-pulls it.
        self.env._invalidate_cache()
        # Sensor reads return captured frames, not live collider reads
        # (``read_found``/``read_force``), so push one frame from the
        # fresh detection — the Newton counterpart's ``_update_sensors``
        # call does the same on its side.
        self._capture_all()

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._groups:
            return ""

        rows = []
        for gname, group in self._groups.items():
            sensor = self._sensors[gname]
            cfg = sensor.cfg
            if cfg.secondary is None:
                sec = "any"
            elif cfg.secondary.entity:
                sec = cfg.secondary.entity
            else:
                sec = str(cfg.secondary.pattern)
            for idx, name in enumerate(group.tracked_names):
                rows.append([gname, idx, name, sec])

        table = create_manager_table(
            title="Contact Tracking (Genesis)",
            columns=["Group", "Idx", "Link", "Secondary"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
