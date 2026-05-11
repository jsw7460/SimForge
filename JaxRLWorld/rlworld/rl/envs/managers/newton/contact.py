from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.newton.contact_sensor import NewtonContactSensor

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class NewtonContactManager(BaseContactManager):
    """Named-group contact manager for Newton environments.

    Each :class:`rlworld.rl.configs.sensors.ContactSensorCfg` becomes a
    named group, backed by :class:`NewtonContactSensor` (which wraps a
    native ``newton.sensors.SensorContact`` and adds substep history).
    Those wrappers are built post scene-build by ``NewtonSceneManager``
    and stored in ``scene_manager.sensors`` keyed by ``cfg.name``; here
    we just adopt them.
    """

    def __init__(self, env: World):
        super().__init__(env)
        # group name -> NewtonContactSensor
        self._group_sensors: dict[str, NewtonContactSensor] = {}

    # -- sensor registration --

    def register_sensors(self) -> None:
        """Discover all contact sensors in the scene and register each as a group."""
        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if not isinstance(sensor, NewtonContactSensor):
                continue
            # ContactSensorCfg path — tracking names already resolved
            # (world-0 leaf names) by the wrapper.
            self._group_sensors[sensor_name] = sensor
            self._register_group(sensor_name, list(sensor.tracked_names))

    # -- abstract impl --

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._group_sensors[group.name].compute_force()

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        return self._group_sensors[group.name].compute_found()

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        return self._group_sensors[group.name].compute_history()  # (num_envs, N, H, 3) or None

    # -- per-env reset --

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        for sensor in self._group_sensors.values():
            sensor.reset(env_ids)

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._groups:
            return ""

        rows = []
        for gname, group in self._groups.items():
            for idx, name in enumerate(group.tracked_names):
                rows.append([gname, idx, name])

        table = create_manager_table(
            title="Contact Tracking (Newton)",
            columns=["Group", "Idx", "Name"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
