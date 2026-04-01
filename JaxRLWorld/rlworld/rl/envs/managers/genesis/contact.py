from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.contact import BaseContactManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Tracks contact state and timing for Genesis environments.

    Automatically discovers all ContactSensors registered in SceneManager.

    Usage::

        contact_manager = ContactManager(env)

        # In step loop (call before reward computation):
        contact_manager.advance()

        # Access timing data:
        first_contact = contact_manager.compute_first_contact()
        air_time = contact_manager.last_air_time

        # Supports regex:
        link_indices = contact_manager.get_link_indices(".*_foot")
    """

    def __init__(self, env: "GenesisEnv"):
        super().__init__(env=env)

        # Structure: [(entity_name, link_name), ...]
        self._tracked_sensors: list[tuple[str, str]] = []
        self._discover_sensors()

        self._num_tracked = len(self._tracked_sensors)
        self._init_buffers()

    # -- backward compat aliases --
    @property
    def num_links(self) -> int:
        return self._num_tracked

    @property
    def tracked_links(self) -> list[tuple[str, str]]:
        return self._tracked_sensors

    @property
    def tracked_names(self) -> list[str]:
        return [link_name for _, link_name in self._tracked_sensors]

    # -- sensor discovery --

    def _discover_sensors(self) -> None:
        sensors = self.env.scene_manager.sensors
        import ipdb; ipdb.set_trace()
        for entity_name, entity_sensors in sensors.items():
            for link_name, link_sensors in entity_sensors.items():
                if "ContactSensor" in link_sensors:
                    self._tracked_sensors.append((entity_name, link_name))

    # -- abstract impl --

    def _compute_is_contact(self) -> torch.Tensor:
        if self._num_tracked == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
        sensors = self.env.scene_manager.sensors
        contact_states = []
        for entity_name, link_name in self._tracked_sensors:
            sensor = sensors[entity_name][link_name]["ContactSensor"]
            contact_states.append(sensor.read().squeeze(-1))
        return torch.stack(contact_states, dim=-1)

    # -- Genesis-specific: entity-aware get_link_indices --

    def get_link_indices(
        self,
        links: str | list[str],
        entity_name: str = "robot",
        preserve_order: bool = False,
    ) -> list[int]:
        entity_links = [
            link_name
            for ent, link_name in self._tracked_sensors
            if ent == entity_name
        ]
        _, matched_names = string_utils.resolve_matching_names(
            links, entity_links, preserve_order=preserve_order
        )
        return [
            self._tracked_sensors.index((entity_name, name))
            for name in matched_names
        ]

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if self._num_tracked == 0:
            return ""
        rows = [
            [idx, link_name, entity_name]
            for idx, (entity_name, link_name) in enumerate(self._tracked_sensors)
        ]
        table = create_manager_table(
            title="Contact Tracking (Genesis)",
            columns=["Idx", "Link Name", "Entity"],
            rows=rows,
            footer=f"{self._num_tracked} tracked links",
        )
        return table_to_string(table)
