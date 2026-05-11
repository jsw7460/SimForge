from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from newton.sensors import SensorContact

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.newton.contact_sensor import NewtonContactSensor
from rlworld.rl.envs.utils.newton.label import leaf_name

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class NewtonContactManager(BaseContactManager):
    """Named-group contact manager for Newton environments.

    Each contact sensor becomes a named group. Two config types are
    accepted for backward compatibility:

    - :class:`rlworld.rl.configs.sensors.ContactSensorCfg` — the
      simulator-agnostic config (go2_flat and newer presets). Backed by
      :class:`NewtonContactSensor`, which wraps a native
      ``newton.sensors.SensorContact`` and adds substep history. These
      wrappers are built post scene-build by ``NewtonSceneManager`` and
      stored in ``scene_manager.sensors`` keyed by ``cfg.name``; here we
      just adopt them.
    - the legacy :class:`NewtonContactSensorConfig` (g1 / t1 presets),
      whose ``SensorContact`` is created directly in
      ``NewtonSceneManager._create_sensor``. We discover those and wrap
      them with timing-only groups (no substep history). **Untouched.**
    """

    def __init__(self, env: World):
        super().__init__(env)
        # group name -> NewtonContactSensor (new) | SensorContact (legacy)
        self._group_sensors: dict[str, NewtonContactSensor | SensorContact] = {}

    # -- sensor registration --

    def register_sensors(self) -> None:
        """Discover all contact sensors in the scene and register each as a group."""
        model: newton.Model = self.env.scene_manager.model

        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, NewtonContactSensor):
                # New ContactSensorCfg path — tracking names already
                # resolved (world-0 leaf names) by the wrapper.
                self._group_sensors[sensor_name] = sensor
                self._register_group(sensor_name, list(sensor.tracked_names))
            elif isinstance(sensor, SensorContact):
                # Legacy NewtonContactSensorConfig path — derive tracking
                # names from the native sensor's resolved indices.
                obj_type = sensor.sensing_obj_type
                label_list = model.body_label if obj_type == "body" else model.shape_label
                world_count = model.world_count
                n_per_env = len(sensor.sensing_obj_idx) // world_count
                first_env_indices = sensor.sensing_obj_idx[:n_per_env]
                names = [leaf_name(label_list[idx]) for idx in first_env_indices]
                self._group_sensors[sensor_name] = sensor
                self._register_group(sensor_name, names)

    # -- abstract impl --

    def _legacy_total_force(self, sensor: SensorContact, group: ContactGroup) -> torch.Tensor:
        """Total force for a legacy ``SensorContact``. Returns ``(num_envs, N, 3)``."""
        if sensor.total_force is not None:
            total_force = wp.to_torch(sensor.total_force)
        else:
            force_matrix = wp.to_torch(sensor.force_matrix)
            total_force = force_matrix.sum(dim=1)
        return total_force.reshape(self.num_envs, group.num_tracked, 3)

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._group_sensors[group.name]
        if isinstance(sensor, NewtonContactSensor):
            return sensor.compute_force()
        return self._legacy_total_force(sensor, group)

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        sensor = self._group_sensors[group.name]
        if isinstance(sensor, NewtonContactSensor):
            return sensor.compute_found()
        force = self._legacy_total_force(sensor, group)
        return torch.norm(force, dim=-1) > 1.0

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._group_sensors[group.name]
        if isinstance(sensor, NewtonContactSensor):
            return sensor.compute_history()  # (num_envs, N, H, 3) or None
        return None

    # -- per-env reset --

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        for sensor in self._group_sensors.values():
            if isinstance(sensor, NewtonContactSensor):
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
