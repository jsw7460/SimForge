from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactListReader, GenesisContactSensor

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

    def register_sensor(self, cfg: ContactSensorCfg) -> None:
        """Register a contact sensor config as a named group."""
        sensor = GenesisContactSensor(self.env, cfg, reader=self._list_reader)
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names)

    # -- per-substep capture + timing accumulation --

    def advance(self, dt: float) -> None:
        """Capture one contact frame per group from the shared list read,
        then run the base per-substep timing arithmetic on it.

        ``GenesisEnv._step_physics`` bumps the cache generation after
        every ``scene.step``, so the sensors' shared collider read is
        fresh exactly once per substep no matter how many groups exist.
        """
        for sensor in self._sensors.values():
            sensor.capture_substep()
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
