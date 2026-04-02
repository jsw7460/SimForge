from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactSensor

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisContactSensorCfg
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Named-group contact manager for Genesis environments.

    Each ``GenesisContactSensorCfg`` becomes a named group backed by a
    ``GenesisContactSensor`` that wraps ``entity.get_contacts()`` with
    primary/secondary filtering.
    """

    def __init__(self, env: "GenesisEnv"):
        super().__init__(env=env)
        self._sensors: dict[str, GenesisContactSensor] = {}

    def register_sensor(self, cfg: "GenesisContactSensorCfg") -> None:
        """Register a contact sensor config as a named group."""
        sensor = GenesisContactSensor(self.env, cfg)
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names)

    # -- abstract impl --

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        return self._sensors[group.name].compute().found

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._sensors[group.name].compute().force

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._groups:
            return ""

        rows = []
        for gname, group in self._groups.items():
            cfg = self._sensors[gname].cfg
            sec = cfg.secondary_entity or "any"
            for idx, name in enumerate(group.tracked_names):
                rows.append([gname, idx, name, sec])

        table = create_manager_table(
            title="Contact Tracking (Genesis)",
            columns=["Group", "Idx", "Link", "Secondary"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
