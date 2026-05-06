from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from newton.sensors import SensorContact

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.utils.newton.label import leaf_name

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class NewtonContactManager(BaseContactManager):
    """Named-group contact manager for Newton environments.

    Each ``SensorContact`` becomes a named group (sensor_name = group_name).
    Multiple sensors are supported. Body order follows Newton's internal
    ordering (typically alphabetical).
    """

    def __init__(self, env: World):
        super().__init__(env)
        self._group_sensors: dict[str, SensorContact] = {}

    # -- sensor registration --

    def register_sensors(self) -> None:
        """Discover all SensorContact instances and register each as a group."""
        model: newton.Model = self.env.scene_manager.model

        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if not isinstance(sensor, SensorContact):
                continue

            obj_type = sensor.sensing_obj_type
            label_list = model.body_label if obj_type == "body" else model.shape_label

            world_count = model.world_count
            n_per_env = len(sensor.sensing_obj_idx) // world_count
            first_env_indices = sensor.sensing_obj_idx[:n_per_env]

            # Canonicalise to bare leaf names so user patterns
            # (``"FR_foot"``, ``".*_foot"``) resolve uniformly across
            # URDF-flat labels (``"go2_description/FR_foot"``) and
            # MJCF-XPath labels (``"g1_29dof/worldbody/.../torso_link"``).
            names = [leaf_name(label_list[idx]) for idx in first_env_indices]

            self._group_sensors[sensor_name] = sensor
            self._register_group(sensor_name, names)

    # -- abstract impl --

    def _get_total_force(self, sensor: SensorContact, group: ContactGroup) -> torch.Tensor:
        """Get total force. Returns (num_envs, N, 3)."""
        if sensor.total_force is not None:
            total_force = wp.to_torch(sensor.total_force)
        else:
            force_matrix = wp.to_torch(sensor.force_matrix)
            total_force = force_matrix.sum(dim=1)

        return total_force.reshape(self.num_envs, group.num_tracked, 3)

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        force = self._get_total_force(self._group_sensors[group.name], group)
        return torch.norm(force, dim=-1) > 1.0

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._get_total_force(self._group_sensors[group.name], group)

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
