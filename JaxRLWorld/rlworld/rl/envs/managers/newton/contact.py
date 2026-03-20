from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from newton.sensors import SensorContact
from rlworld.rl.envs.managers.common.contact import BaseContactManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World

import newton


class NewtonContactManager(BaseContactManager):
    """Manages contact information for Newton environments.

    Tracks contact state and timing for all shapes registered with SensorContact.

    CRITICAL ORDERING INVARIANT:
    Newton's SensorContact stores sensing_objs in env-major order.
    ``shape_names`` stores names for ONE env (first env's shapes).
    All output tensors have shape ``(num_envs, num_shapes)``.
    """

    def __init__(self, env: "World"):
        super().__init__(env)
        self._contact_sensors: dict[str, SensorContact] = {}
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
        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, SensorContact):
                self._contact_sensors[sensor_name] = sensor

        if not self._contact_sensors:
            return

        if len(self._contact_sensors) > 1:
            raise ValueError(
                f"NewtonContactManager currently supports only one SensorContact. "
                f"Found {len(self._contact_sensors)}: {list(self._contact_sensors.keys())}"
            )

        model: newton.Model = self.env.scene_manager.model
        sensor: SensorContact = list(self._contact_sensors.values())[0]

        obj_type = sensor.sensing_obj_type
        label_list = model.body_label if obj_type == "body" else model.shape_label

        world_count = self.env.scene_manager.model.world_count
        n_per_env = len(sensor.sensing_obj_idx) // world_count
        first_env_indices = sensor.sensing_obj_idx[:n_per_env]

        for idx in first_env_indices:
            self._shape_names.append(label_list[idx])

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
            if sensor.total_force is not None:
                total_force = wp.to_torch(sensor.total_force)
            else:
                force_matrix = wp.to_torch(sensor.force_matrix)
                total_force = force_matrix.sum(dim=1)

            force_magnitude = torch.norm(total_force, dim=-1)
            contact_states.append(force_magnitude > 1.0)

        if not contact_states:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        return torch.cat(contact_states, dim=0).reshape(
            self.num_envs, self._num_tracked
        )

    # -- Newton-specific --

    @property
    def contact_force(self) -> torch.Tensor:
        """Contact force ``(num_envs, num_shapes, 3)``."""
        if self._num_tracked == 0:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)

        sensor = list(self._contact_sensors.values())[0]
        if sensor.total_force is not None:
            total_force = wp.to_torch(sensor.total_force)
        else:
            force_matrix = wp.to_torch(sensor.force_matrix)
            total_force = force_matrix.sum(dim=1)

        return total_force.reshape(self.num_envs, self._num_tracked, 3)

    def get_shape_indices(
        self,
        patterns: str | list[str],
        use_regex: bool = False,
        preserve_order: bool = False,
    ) -> list[int]:
        from rlworld.rl.utils import string as string_utils

        if isinstance(patterns, str):
            patterns = [patterns]
        _, matched = string_utils.resolve_matching_names(
            patterns, self._shape_names, preserve_order=preserve_order
        )
        return [self._shape_names.index(n) for n in matched]

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if self._num_tracked == 0:
            return ""
        rows = [[idx, name] for idx, name in enumerate(self._shape_names)]
        table = create_manager_table(
            title="Contact Tracking (Newton)",
            columns=["Idx", "Shape Name"],
            rows=rows,
            footer=f"{self._num_tracked} tracked shapes",
        )
        return table_to_string(table)
