from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@dataclass
class ContactManagerConfig:
    pass


class ContactManager(BaseManager):
    """Tracks contact state and timing for all links with ContactSensor.

    Automatically discovers all ContactSensors registered in SceneManager
    and tracks timing information for each.

    Maintains running timers for:
    - current_air_time: How long each link has been in the air
    - current_contact_time: How long each link has been in contact
    - last_air_time: Duration of the previous air phase (updated on landing)
    - last_contact_time: Duration of the previous contact phase (updated on liftoff)

    Usage:
        # Automatically initialized in GenesisEnv._setup_environment()
        contact_manager = ContactManager(env)

        # In step loop (call before reward computation):
        contact_manager.advance()

        # Access timing data:
        first_contact = contact_manager.compute_first_contact()
        air_time = contact_manager.last_air_time

        # In reward function (supports regex):
        link_indices = contact_manager.get_link_indices(".*_foot")
        reward = torch.sum((air_time[:, link_indices] - threshold) * first_contact[:, link_indices], dim=-1)
    """

    def __init__(self, env: "GenesisEnv"):
        """Initialize contact manager.

        Args:
            env: The Genesis environment instance.
        """
        super().__init__(env=env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt

        # Auto-discover all ContactSensors
        # Structure: [(entity_name, link_name), ...]
        self._tracked_sensors: list[tuple[str, str]] = []
        self._discover_sensors()

        self.num_links = len(self._tracked_sensors)

        if self.num_links == 0:
            return

        # Timing buffers: (num_envs, num_links)
        self.current_air_time = torch.zeros(
            self.num_envs, self.num_links, device=self.device
        )
        self.current_contact_time = torch.zeros(
            self.num_envs, self.num_links, device=self.device
        )
        self.last_air_time = torch.zeros(
            self.num_envs, self.num_links, device=self.device
        )
        self.last_contact_time = torch.zeros(
            self.num_envs, self.num_links, device=self.device
        )
        self._prev_is_contact = torch.zeros(
            self.num_envs, self.num_links, dtype=torch.bool, device=self.device
        )

    def _discover_sensors(self) -> None:
        """Discover all links with ContactSensor from SceneManager.

        Iterates through all registered sensors and collects links
        that have a ContactSensor attached.
        """
        sensors = self.env.scene_manager.sensors

        for entity_name, entity_sensors in sensors.items():
            for link_name, link_sensors in entity_sensors.items():
                if "ContactSensor" in link_sensors:
                    self._tracked_sensors.append((entity_name, link_name))

    def _get_contact_states(self) -> torch.Tensor:
        """Read contact states from all tracked sensors.

        Returns:
            Boolean tensor of shape (num_envs, num_links) indicating
            whether each tracked link is currently in contact.
        """
        sensors = self.env.scene_manager.sensors

        contact_states = []
        for entity_name, link_name in self._tracked_sensors:
            sensor = sensors[entity_name][link_name]["ContactSensor"]
            contact_state = sensor.read().squeeze(-1)  # (num_envs,)
            contact_states.append(contact_state)

        return torch.stack(contact_states, dim=-1)

    @property
    def is_contact(self) -> torch.Tensor:
        """Current contact state for all tracked links.

        Returns:
            Boolean tensor of shape (num_envs, num_links).
        """
        return self._get_contact_states()

    @property
    def tracked_links(self) -> list[tuple[str, str]]:
        """List of tracked (entity_name, link_name) pairs."""
        return self._tracked_sensors

    def get_link_names(self, indices: list[int]) -> list[str]:
        return [self._tracked_sensors[i][1] for i in indices]

    def get_link_indices(
        self,
        links: str | list[str],
        entity_name: str = "robot",
        preserve_order: bool = False,
    ) -> list[int]:
        """Get indices of links in tracked_sensors matching the pattern.

        Supports regex patterns for flexible link selection.

        Args:
            links: Link name pattern(s). Supports regex (e.g., ".*_foot").
            entity_name: Name of the entity containing the links.

        Returns:
            List of indices in tracked_sensors matching the pattern.

        Example:
            # Single regex pattern
            indices = contact_manager.get_link_indices(".*_foot")

            # Explicit list
            indices = contact_manager.get_link_indices(["FL_foot", "FR_foot"])
        """
        # Get all link names for this entity
        entity_links = [
            link_name for ent, link_name in self._tracked_sensors
            if ent == entity_name
        ]

        # Resolve matching names using regex
        _, matched_names = string_utils.resolve_matching_names(links, entity_links, preserve_order=preserve_order)

        # Convert to indices
        indices = [
            self._tracked_sensors.index((entity_name, name))
            for name in matched_names
        ]

        return indices

    def advance(self) -> None:
        """Advance contact timing based on current contact states.

        Should be called once per environment step, before reward computation.
        """
        if self.num_links == 0:
            return

        is_contact = self._get_contact_states()

        # Detect state transitions
        is_landing = ~self._prev_is_contact & is_contact  # air -> contact
        is_liftoff = self._prev_is_contact & ~is_contact  # contact -> air

        # On landing: save air time before resetting
        self.last_air_time = torch.where(
            is_landing, self.current_air_time, self.last_air_time
        )

        # On liftoff: save contact time before resetting
        self.last_contact_time = torch.where(
            is_liftoff, self.current_contact_time, self.last_contact_time
        )

        # Update current timers
        self.current_contact_time = torch.where(
            is_contact,
            self.current_contact_time + self.dt,
            torch.zeros_like(self.current_contact_time)
        )
        self.current_air_time = torch.where(
            ~is_contact,
            self.current_air_time + self.dt,
            torch.zeros_like(self.current_air_time)
        )

        # Cache for next update
        self._prev_is_contact = is_contact

    def compute_first_contact(self, abs_tol: float = 1e-6) -> torch.Tensor:
        """Detect links that just made contact within the last dt.

        Useful for rewarding foot touchdown events in locomotion.

        Args:
            abs_tol: Absolute tolerance for time comparison.

        Returns:
            Boolean tensor of shape (num_envs, num_links) where True
            indicates the link landed within the last dt seconds.
        """
        is_contact = self.current_contact_time > 0
        just_landed = self.current_contact_time < (self.dt + abs_tol)
        return is_contact & just_landed

    def compute_first_air(self, abs_tol: float = 1e-6) -> torch.Tensor:
        """Detect links that just lifted off within the last dt.

        Useful for detecting foot liftoff events in locomotion.

        Args:
            abs_tol: Absolute tolerance for time comparison.

        Returns:
            Boolean tensor of shape (num_envs, num_links) where True
            indicates the link lifted off within the last dt seconds.
        """
        is_air = self.current_air_time > 0
        just_lifted = self.current_air_time < (self.dt + abs_tol)
        return is_air & just_lifted

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset timing buffers for specified environments.

        Should be called in _reset_idx() when environments are reset.

        Args:
            env_ids: Environment indices to reset.
        """
        if self.num_links == 0 or len(env_ids) == 0:
            return

        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self._prev_is_contact[env_ids] = False

    def __str__(self) -> str:
        """Pretty print contact manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if self.num_links == 0:
            return ""

        rows = []
        for idx, (entity_name, link_name) in enumerate(self._tracked_sensors):
            rows.append([idx, link_name, entity_name])

        table = create_manager_table(
            title="Contact Tracking (Genesis)",
            columns=["Idx", "Link Name", "Entity"],
            rows=rows,
            footer=f"{self.num_links} tracked links"
        )
        return table_to_string(table)
