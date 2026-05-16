from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactSensor

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Named-group contact manager for Genesis environments.

    Each :class:`rlworld.rl.configs.sensors.ContactSensorCfg` becomes a
    named group, backed by :class:`GenesisContactSensor` (which prefers
    Genesis's native link sensors). Those native sensors must be created
    *before* ``scene.build()``, so they are pre-built by
    ``GenesisEnv._build_scene`` and merely *adopted* here (see
    ``env._genesis_contact_sensors``).
    """

    def __init__(self, env: GenesisEnv):
        super().__init__(env=env)
        self._sensors: dict[str, GenesisContactSensor] = {}

    def register_sensor(self, cfg: ContactSensorCfg) -> None:
        """Register a contact sensor config as a named group.

        The sensor (and, for the native path, its ``gs.sensors.*``
        objects added to the still-unbuilt scene) is pre-built by
        ``GenesisEnv._build_scene``; here we just adopt it.
        """
        pre_built = getattr(self.env, "_genesis_contact_sensors", {})
        sensor = pre_built.get(cfg.name)
        if sensor is None:
            raise RuntimeError(
                f"ContactSensorCfg {cfg.name!r} was not pre-registered before "
                "scene.build(); ensure GenesisEnv._build_scene() iterates "
                "scene_cfg.contact_sensors and calls create_native_sensors()."
            )
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names)

    # -- abstract impl --

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._sensors[group.name].compute().force

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._sensors[group.name]
        compute_history = getattr(sensor, "compute_history", None)
        if compute_history is None:
            return None
        return compute_history()

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
