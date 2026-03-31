from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.contact import BaseContactManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.sensor.contact_sensor import ContactSensor


class MujocoContactManager(BaseContactManager):
    """Manages contact information for MuJoCo/mjlab environments.

    Tracks contact state and timing for shapes registered with mjlab ContactSensor.
    All tensors have shape ``(num_envs, num_shapes)``.
    """

    def __init__(self, env: "World"):
        super().__init__(env)
        self._contact_sensors: dict[str, "ContactSensor"] = {}
        self._shape_names: list[str] = []

    # -- backward compat aliases --

    @property
    def num_shapes(self) -> int:
        return self._num_tracked

    @property
    def shape_names(self) -> list[str]:
        return self._shape_names

    @property
    def tracked_names(self) -> list[str]:
        return self._shape_names

    # -- sensor registration --

    def register_sensors(self) -> None:
        from mjlab.sensor.contact_sensor import ContactSensor

        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, ContactSensor):
                self._contact_sensors[sensor_name] = sensor

        if not self._contact_sensors:
            return

        for sensor in self._contact_sensors.values():
            primary_names = list(
                dict.fromkeys(slot.primary_name for slot in sensor._slots)
            )
            self._shape_names.extend(primary_names)

        self._num_tracked = len(self._shape_names)
        self._init_buffers()

    # -- abstract impl --

    def _compute_is_contact(self) -> torch.Tensor:
        if self._num_tracked == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        contact_states = []
        for sensor in self._contact_sensors.values():
            found = sensor.data.found
            if found is not None:
                contact_states.append(found > 0)

        if not contact_states:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        return torch.cat(contact_states, dim=1)

    # -- MuJoCo-specific --

    @property
    def contact_force(self) -> torch.Tensor:
        """Contact force ``(num_envs, num_shapes, 3)``."""
        if self._num_tracked == 0:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)

        forces = []
        for sensor in self._contact_sensors.values():
            force = sensor.data.force
            if force is not None:
                forces.append(force)
            else:
                num_primaries = len(
                    list(
                        dict.fromkeys(
                            slot.primary_name for slot in sensor._slots
                        )
                    )
                )
                forces.append(
                    torch.zeros(
                        self.num_envs, num_primaries, 3, device=self.device
                    )
                )
        return torch.cat(forces, dim=1)

    def get_sensor(self, sensor_name: str) -> "ContactSensor":
        return self._contact_sensors[sensor_name]

    def get_sensor_air_time(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        current_times = []
        last_times = []
        for sensor in self._contact_sensors.values():
            if sensor.cfg.track_air_time:
                if sensor.data.current_air_time is not None:
                    current_times.append(sensor.data.current_air_time)
                if sensor.data.last_air_time is not None:
                    last_times.append(sensor.data.last_air_time)
        if current_times:
            return torch.cat(current_times, dim=1), torch.cat(last_times, dim=1)
        return None, None

    def get_sensor_contact_time(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        current_times = []
        last_times = []
        for sensor in self._contact_sensors.values():
            if sensor.cfg.track_air_time:
                if sensor.data.current_contact_time is not None:
                    current_times.append(sensor.data.current_contact_time)
                if sensor.data.last_contact_time is not None:
                    last_times.append(sensor.data.last_contact_time)
        if current_times:
            return torch.cat(current_times, dim=1), torch.cat(last_times, dim=1)
        return None, None

    def get_shape_indices(
        self,
        patterns: str | list[str],
        use_regex: bool = False,
        preserve_order: bool = False,
    ) -> list[int]:
        import re

        if isinstance(patterns, str):
            patterns = [patterns]

        matched_indices = []
        for i, name in enumerate(self._shape_names):
            for pattern in patterns:
                if use_regex:
                    if re.search(pattern, name):
                        matched_indices.append(i)
                        break
                else:
                    if pattern in name or pattern == name:
                        matched_indices.append(i)
                        break
        return matched_indices

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if self._num_tracked == 0:
            return ""
        rows = [[idx, name] for idx, name in enumerate(self._shape_names)]
        table = create_manager_table(
            title="Contact Tracking (MuJoCo/mjlab)",
            columns=["Idx", "Shape Name"],
            rows=rows,
            footer=f"{self._num_tracked} tracked shapes",
        )
        return table_to_string(table)
