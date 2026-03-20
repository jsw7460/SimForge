from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from newton.sensors import SensorContact
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World
import newton

class NewtonContactManager(BaseManager):
    """Manages contact information for Newton environments.

    Tracks contact state and timing for all shapes registered with SensorContact.
    Provides Genesis-compatible interface for reward functions.

    CRITICAL ORDERING INVARIANT:
    ============================
    All tensors in this class follow the same ordering convention:

    1. Newton's SensorContact stores sensing_objs in env-major order:
       [env0_shape0, env0_shape1, ..., env1_shape0, env1_shape1, ...]

    2. shape_names stores names for ONE env (first env's shapes):
       [shape0_name, shape1_name, ...]

    3. All output tensors have shape (num_envs, num_shapes) where:
       - Axis 0: environment index
       - Axis 1: shape index (matches shape_names order)
       - tensor[env_idx, shape_idx] corresponds to shape_names[shape_idx]

    This ordering is guaranteed by:
    - register_sensors(): extracts names from sensing_objs[0:num_shapes_per_env]
    - _compute_is_contact(): reshapes flat sensor output to (num_envs, num_shapes)
    - All other methods index into these consistently ordered tensors

    Usage:
        # Access contact state
        is_contact = contact_manager.is_contact  # (num_envs, num_shapes)

        # Get shape indices by pattern
        foot_indices = contact_manager.get_shape_indices([".*_foot"], use_regex=True)

        # Timing information (call advance() each step)
        first_contact = contact_manager.compute_first_contact()
        air_time = contact_manager.last_air_time
    """

    def __init__(self, env: "World"):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt

        self._contact_sensors: dict[str, SensorContact] = {}

        # shape_names[i] = name of i-th shape
        # This order defines the canonical shape indexing for ALL tensors
        self._shape_names: list[str] = []
        self.num_shapes: int = 0

        # All timing buffers have shape (num_envs, num_shapes)
        # Axis 1 ordering matches self._shape_names
        self.current_air_time: torch.Tensor | None = None
        self.current_contact_time: torch.Tensor | None = None
        self.last_air_time: torch.Tensor | None = None
        self.last_contact_time: torch.Tensor | None = None
        self._prev_is_contact: torch.Tensor | None = None

    def register_sensors(self) -> None:
        """Discover and register all SensorContact instances from scene_manager.

        ORDERING: Establishes the canonical shape ordering from sensor.sensing_objs.

        Newton's sensing_objs layout (env-major order):
            [env0_shape0, env0_shape1, ..., env0_shapeN,
             env1_shape0, env1_shape1, ..., env1_shapeN,
             ...]

        We extract names from the first env's shapes (indices 0 to num_shapes_per_env-1).
        This ordering is preserved in all subsequent tensor operations.
        """
        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if isinstance(sensor, SensorContact):
                self._contact_sensors[sensor_name] = sensor

        if not self._contact_sensors:
            self.num_shapes = 0
            return

        if len(self._contact_sensors) > 1:
            raise ValueError(
                f"NewtonContactManager currently supports only one SensorContact. "
                f"Found {len(self._contact_sensors)}: {list(self._contact_sensors.keys())}"
            )

        model: newton.Model = self.env.scene_manager.model
        sensor: SensorContact = list(self._contact_sensors.values())[0]

        # sensing_obj_idx: flat list, sensing_obj_type: "body" | "shape"
        obj_type = sensor.sensing_obj_type  # "body" or "shape"
        label_list = model.body_label if obj_type == "body" else model.shape_label

        # sensing_obj_idx already preserves per-world order (world 0 = first N entries)
        world_count = self.env.scene_manager.model.world_count
        n_per_env = len(sensor.sensing_obj_idx) // world_count

        first_env_indices = sensor.sensing_obj_idx[:n_per_env]

        for idx in first_env_indices:
            self._shape_names.append(label_list[idx])

        self.num_shapes = len(self._shape_names)

        if self.num_shapes == 0:
            return

        # Initialize timing buffers with shape (num_envs, num_shapes)
        # ORDERING: [:, i] corresponds to shape_names[i] for all buffers
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
        """Compute contact state from sensor net_force.

        ORDERING: sensor.net_force is stored in env-major order:
            [env0_shape0, env0_shape1, ..., env1_shape0, env1_shape1, ...]

        After reshape(num_envs, num_shapes):
            result[env_idx, shape_idx] = contact state for shape_names[shape_idx] in env_idx

        This matches the canonical ordering established in register_sensors().

        Returns:
            Boolean tensor (num_envs, num_shapes) - True if in contact.
            Axis 1 ordering matches self.shape_names.
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        # Aggregate contact states from all sensors
        contact_states = []

        for sensor in self._contact_sensors.values():
            if sensor.total_force is not None:
                total_force = wp.to_torch(sensor.total_force)  # shape: (n_sensing_objs, 3)
            else:
                # force_matrix: (n_sensing_objs, n_counterparts, 3)
                force_matrix = wp.to_torch(sensor.force_matrix)
                total_force = force_matrix.sum(dim=1)

            force_magnitude = torch.norm(total_force, dim=-1)

            # Contact if force > small threshold
            is_contact = force_magnitude > 1.0  # (n_sensing_objs,)
            contact_states.append(is_contact)

        if not contact_states:
            return torch.zeros(
                self.num_envs, self.num_shapes, dtype=torch.bool, device=self.device
            )

        # Stack and reshape to (num_envs, num_shapes)
        all_contacts = torch.cat(contact_states, dim=0)
        # Reshape based on num_envs
        return all_contacts.reshape(self.num_envs, self.num_shapes)

    @property
    def is_contact(self) -> torch.Tensor:
        """Current contact state for all tracked shapes.

        Returns:
            Boolean tensor (num_envs, num_shapes).
            ORDERING: [:, i] corresponds to shape_names[i].
        """
        return self._compute_is_contact()

    @property
    def shape_names(self) -> list[str]:
        """List of tracked shape names.

        ORDERING: This list defines the canonical shape ordering.
        shape_names[i] corresponds to column i in all output tensors.
        """
        return self._shape_names

    @property
    def contact_force(self) -> torch.Tensor:
        """Contact force for all tracked shapes.

        Returns:
            Tensor (num_envs, num_shapes, 3) - force vectors.
            ORDERING: [:, i, :] corresponds to shape_names[i].
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, 3, device=self.device
            )

        sensor = list(self._contact_sensors.values())[0]
        if sensor.total_force is not None:
            total_force = wp.to_torch(sensor.total_force)  # (n_sensing_objs, 3)
        else:
            force_matrix = wp.to_torch(sensor.force_matrix)
            total_force = force_matrix.sum(dim=1)

        return total_force.reshape(self.num_envs, self.num_shapes, 3)

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
            ORDERING: These indices can be used to index axis 1 of any output tensor.
            Example: is_contact[:, indices] gives contact states for matched shapes.
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        _, matched_names = string_utils.resolve_matching_names(
            patterns, self._shape_names, preserve_order=preserve_order
        )

        return [self._shape_names.index(name) for name in matched_names]

    def get_link_indices(
        self,
        links: str | list[str],
        entity_name: str = "robot",
        preserve_order: bool = False,
    ) -> list[int]:
        """Genesis-compatible alias for get_shape_indices.

        ORDERING: Returns indices compatible with all tensor outputs.
        """
        return self.get_shape_indices(links, use_regex=True, preserve_order=preserve_order)

    def advance(self) -> None:
        """Advance contact timing based on current contact states.

        Should be called once per environment step, before reward computation.

        ORDERING: All tensor operations preserve the (num_envs, num_shapes) layout
        where axis 1 matches shape_names order.
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
        """Detect shapes that just made contact within the last dt.

        Returns:
            Boolean tensor (num_envs, num_shapes).
            ORDERING: [:, i] corresponds to shape_names[i].
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        is_contact = self.current_contact_time > 0
        just_landed = self.current_contact_time < (self.dt + abs_tol)
        return is_contact & just_landed

    def compute_first_air(self, abs_tol: float = 1e-6) -> torch.Tensor:
        """Detect shapes that just lifted off within the last dt.

        Returns:
            Boolean tensor (num_envs, num_shapes).
            ORDERING: [:, i] corresponds to shape_names[i].
        """
        if self.num_shapes == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )

        is_air = self.current_air_time > 0
        just_lifted = self.current_air_time < (self.dt + abs_tol)
        return is_air & just_lifted

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset timing buffers for specified environments.
        """
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
            title="Contact Tracking (Newton)",
            columns=["Idx", "Shape Name"],
            rows=rows,
            footer=f"{self.num_shapes} tracked shapes"
        )
        return table_to_string(table)
