from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.sensor.contact_sensor import ContactSensor


class MujocoContactManager(BaseContactManager):
    """Named-group contact manager for MuJoCo/mjlab environments.

    Each mjlab ``ContactSensor`` becomes a named group (sensor cfg name = group name).
    """

    def __init__(self, env: "World"):
        super().__init__(env)
        self._group_sensors: dict[str, "ContactSensor"] = {}

    # -- sensor registration --

    def register_sensors(self) -> None:
        from mjlab.sensor.contact_sensor import ContactSensor

        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, ContactSensor):
                self._group_sensors[sensor_name] = sensor
                primary_names = list(
                    dict.fromkeys(slot.primary_name for slot in sensor._slots)
                )
                self._register_group(sensor_name, primary_names)

    # -- abstract impl --

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        sensor = self._group_sensors[group.name]
        found = sensor.data.found
        if found is None:
            return torch.zeros(
                self.num_envs, group.num_tracked, dtype=torch.bool, device=self.device
            )
        if found.dim() == 3:
            found = found.squeeze(-1)
        return found > 0

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._group_sensors[group.name]
        return sensor.data.force

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._group_sensors[group.name]
        return sensor.data.force_history  # (num_envs, N, H, 3) or None

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
            title="Contact Tracking (MuJoCo/mjlab)",
            columns=["Group", "Idx", "Name"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
