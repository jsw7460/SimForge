from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactSensor
from rlworld.rl.envs.managers.genesis.legacy_contact_sensor import LegacyGenesisContactSensor

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisContactSensorCfg
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Named-group contact manager for Genesis environments.

    Each contact sensor config becomes a named group. Two config types
    are accepted for backward compatibility:

    - :class:`rlworld.rl.configs.sensors.ContactSensorCfg` — the
      simulator-agnostic config (go2_flat and newer presets). Backed by
      :class:`GenesisContactSensor`, which prefers Genesis's native link
      sensors. Those native sensors must be created *before*
      ``scene.build()``, so they are pre-built by
      ``GenesisEnv._build_scene`` and merely *adopted* here (see
      ``env._genesis_contact_sensors``); only when no pre-built sensor is
      found do we construct one on the spot (e.g. the ``get_contacts``
      self-collision path, which is build-agnostic).
    - the legacy :class:`GenesisContactSensorCfg` (g1 / t1 presets),
      backed by :class:`LegacyGenesisContactSensor` (the old
      ``entity.get_contacts``-based wrapper, untouched).
    """

    def __init__(self, env: GenesisEnv):
        super().__init__(env=env)
        self._sensors: dict[str, GenesisContactSensor | LegacyGenesisContactSensor] = {}

    def register_sensor(self, cfg: ContactSensorCfg | GenesisContactSensorCfg) -> None:
        """Register a contact sensor config as a named group.

        ``ContactSensorCfg`` sensors are pre-built (and, for the native
        path, their ``gs.sensors.*`` objects added to the still-unbuilt
        scene) by ``GenesisEnv._build_scene``; here we just adopt them.
        Legacy ``GenesisContactSensorCfg`` sensors are build-agnostic
        and constructed on the spot.
        """
        if isinstance(cfg, ContactSensorCfg):
            pre_built = getattr(self.env, "_genesis_contact_sensors", {})
            sensor = pre_built.get(cfg.name)
            if sensor is None:
                raise RuntimeError(
                    f"ContactSensorCfg {cfg.name!r} was not pre-registered before "
                    "scene.build(); ensure GenesisEnv._build_scene() iterates "
                    "scene_cfg.contact_sensors and calls create_native_sensors()."
                )
        else:
            sensor = LegacyGenesisContactSensor(self.env, cfg)
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names)

    # -- abstract impl --

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        return self._sensors[group.name].compute().found

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
            if isinstance(cfg, ContactSensorCfg):
                if cfg.secondary is None:
                    sec = "any"
                elif cfg.secondary.entity:
                    sec = cfg.secondary.entity
                else:
                    sec = str(cfg.secondary.pattern)
            else:
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
