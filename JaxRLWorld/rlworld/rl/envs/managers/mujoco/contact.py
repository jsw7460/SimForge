from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.sensor.contact_sensor import ContactSensor


class MjlabContactManager(BaseManager):
    """Manages contact information for MuJoCo/mjlab environments.

    Tracks contact state and timing for shapes registered with mjlab ContactSensor.
    Provides rlworld-compatible interface for reward functions.

    All tensors have shape (num_envs, num_shapes) where:
    - Axis 0: environment index
    - Axis 1: shape index (matches shape_names order)

    Usage:
        contact_manager = MjlabContactManager(env)
        contact_manager.register_sensors()

        # In step loop:
        contact_manager.advance()

        # Access contact state
        is_contact = contact_manager.is_contact  # (num_envs, num_shapes)
    """

    def __init__(self, env: "World"):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt

        self._contact_sensors: dict[str, "ContactSensor"] = {}

        # Shape names for indexing
        self._shape_names: list[str] = []
        self.num_shapes: int = 0

        # Timing buffers (num_envs, num_shapes)
        self.current_air_time: torch.Tensor | None = None
        self.current_contact_time: torch.Tensor | None = None
        self.last_air_time: torch.Tensor | None = None
        self.last_contact_time: torch.Tensor | None = None
        self._prev_is_contact: torch.Tensor | None = None

    def register_sensors(self) -> None:
        """Discover and register all ContactSensor instances from scene_manager."""
        from mjlab.sensor.contact_sensor import ContactSensor

        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, ContactSensor):
                self._contact_sensors[sensor_name] = sensor

        if not self._contact_sensors:
            self.num_shapes = 0
            return

        # Collect shape names from sensors (primary names from _slots)
        for sensor in self._contact_sensors.values():
            # Get unique primary names from slots
            primary_names = list(dict.fromkeys(
                slot.primary_name for slot in sensor._slots
            ))
            self._shape_names.extend(primary_names)

        self.num_shapes = len(self._shape_names)

        if self.num_shapes == 0:
            return

        # Initialize timing buffers
        self.current_air_time = torch.zeros(
            self.num_envs, self.num_shapes, device=self.device
        )
        self.current_contact_time = torch.zeros(
            self.num_envs, self.num_shapes, device=self.device
        )
        self.last_air_time = torch.zeros(
            self.num_envs, self.num_shapes, device=self.device
        )
        self.last_contact_time = torch.zeros(
            self.num_envs, self.num_shapes, device=self.device
        )
        self._prev_is_contact = torch.zeros(
            self.num_envs, self.num_shapes, dtype=torch.bool, device=self.device
        )

    def _compute_is_contact(self) -> torch.Tensor:
        """Compute contact state from sensor data.

        Returns:
            Boolean tensor (num_envs, num_shapes) - True if in contact.
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        contact_states = []

        for sensor in self._contact_sensors.values():
            # Get found field from ContactData dataclass (0=no contact, >0=contact)
            found = sensor.data.found  # (num_envs, num_primaries)
            if found is not None:
                is_contact = found > 0
                contact_states.append(is_contact)

        if not contact_states:
            return torch.zeros(
                self.num_envs, self.num_shapes, dtype=torch.bool, device=self.device
            )

        # Concatenate across sensors
        return torch.cat(contact_states, dim=1)

    @property
    def is_contact(self) -> torch.Tensor:
        """Current contact state for all tracked shapes.

        Returns:
            Boolean tensor (num_envs, num_shapes).
        """
        return self._compute_is_contact()

    @property
    def shape_names(self) -> list[str]:
        """List of tracked shape names."""
        return self._shape_names

    @property
    def contact_force(self) -> torch.Tensor:
        """Contact force for all tracked shapes.

        Returns:
            Tensor (num_envs, num_shapes, 3) - force vectors.
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, 3, device=self.device
            )

        forces = []
        for sensor in self._contact_sensors.values():
            # Access force from ContactData dataclass
            force = sensor.data.force
            if force is not None:
                forces.append(force)  # (num_envs, num_primaries, 3)
            else:
                # If no force field, return zeros
                num_primaries = len(list(dict.fromkeys(
                    slot.primary_name for slot in sensor._slots
                )))
                forces.append(torch.zeros(
                    self.num_envs, num_primaries, 3, device=self.device
                ))

        return torch.cat(forces, dim=1)

    def get_sensor(self, sensor_name: str) -> "ContactSensor":
        """Get specific contact sensor by name.

        Args:
            sensor_name: Name of the contact sensor.

        Returns:
            The ContactSensor instance.

        Raises:
            KeyError: If sensor_name not found.
        """
        return self._contact_sensors[sensor_name]

    def get_sensor_air_time(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get air time from sensors with track_air_time=True.

        Returns:
            Tuple of (current_air_time, last_air_time) or (None, None) if not available.
        """
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

    def get_sensor_contact_time(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get contact time from sensors with track_air_time=True.

        Returns:
            Tuple of (current_contact_time, last_contact_time) or (None, None) if not available.
        """
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
        """Get indices of shapes matching the pattern.

        Args:
            patterns: Shape name pattern(s).
            use_regex: If True, use regex matching.
            preserve_order: If True, preserve order of patterns.

        Returns:
            List of indices into shape_names matching the pattern.
        """
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

    def get_link_indices(
        self,
        links: str | list[str],
        entity_name: str = "robot",
        preserve_order: bool = False,
    ) -> list[int]:
        """Genesis-compatible alias for get_shape_indices."""
        return self.get_shape_indices(links, use_regex=True, preserve_order=preserve_order)

    def advance(self) -> None:
        """Advance contact timing based on current contact states.

        Should be called once per environment step.
        """
        if self.num_shapes == 0:
            return

        is_contact = self._compute_is_contact()

        # Detect state transitions
        is_landing = ~self._prev_is_contact & is_contact
        is_liftoff = self._prev_is_contact & ~is_contact

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

        self._prev_is_contact = is_contact

    def compute_first_contact(self, abs_tol: float = 1e-6) -> torch.Tensor:
        """Detect shapes that just made contact within the last dt."""
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        is_contact = self.current_contact_time > 0
        just_landed = self.current_contact_time < (self.dt + abs_tol)
        return is_contact & just_landed

    def compute_first_air(self, abs_tol: float = 1e-6) -> torch.Tensor:
        """Detect shapes that just lifted off within the last dt."""
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        is_air = self.current_air_time > 0
        just_lifted = self.current_air_time < (self.dt + abs_tol)
        return is_air & just_lifted

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset timing buffers for specified environments."""
        if self.num_shapes == 0 or env_ids is None or len(env_ids) == 0:
            return

        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self._prev_is_contact[env_ids] = False

    def __str__(self) -> str:
        """Pretty print contact manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if self.num_shapes == 0:
            return ""

        rows = []
        for idx, shape_name in enumerate(self._shape_names):
            rows.append([idx, shape_name])

        table = create_manager_table(
            title="Contact Tracking (MuJoCo/mjlab)",
            columns=["Idx", "Shape Name"],
            rows=rows,
            footer=f"{self.num_shapes} tracked shapes"
        )
        return table_to_string(table)
