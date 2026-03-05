from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from newton.sensors import SensorContact
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World
import newton

class NewtonContactManager(BaseManager):
    """Manages contact information for Newton environments (JAX-native)."""

    def __init__(self, env: "World"):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt

        self._contact_sensors: dict[str, SensorContact] = {}

        self._shape_names: list[str] = []
        self.num_shapes: int = 0
        self._include_total: bool = True

        self.current_air_time: jax.Array | None = None
        self.current_contact_time: jax.Array | None = None
        self.last_air_time: jax.Array | None = None
        self.last_contact_time: jax.Array | None = None
        self._prev_is_contact: jax.Array | None = None

    def register_sensors(self) -> None:
        """Discover and register all SensorContact instances from scene_manager."""
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

        self._include_total = True

        num_sensing_objs = len(sensor.sensing_objs)
        num_shapes_per_env = num_sensing_objs // self.num_envs

        first_env_objs = []
        for i, (idx, match_kind) in enumerate(sensor.sensing_objs):
            first_env_objs.append((idx, match_kind, i))

        first_env_objs.sort(key=lambda x: x[0])

        for idx, match_kind, _ in first_env_objs[:num_shapes_per_env]:
            if match_kind == SensorContact.ObjectType.BODY:
                name = model.body_label[idx]
            elif match_kind == SensorContact.ObjectType.SHAPE:
                name = model.shape_label[idx]
            else:
                continue
            self._shape_names.append(name)

        self.num_shapes = len(self._shape_names)

        if self.num_shapes == 0:
            return

        self.current_air_time = jnp.zeros((self.num_envs, self.num_shapes))
        self.current_contact_time = jnp.zeros((self.num_envs, self.num_shapes))
        self.last_air_time = jnp.zeros((self.num_envs, self.num_shapes))
        self.last_contact_time = jnp.zeros((self.num_envs, self.num_shapes))
        self._prev_is_contact = jnp.zeros((self.num_envs, self.num_shapes), dtype=jnp.bool_)

    def _compute_is_contact(self) -> jax.Array:
        """Compute contact state from sensor net_force."""
        if self.num_shapes == 0:
            return jnp.zeros((self.num_envs, 0), dtype=jnp.bool_)

        contact_states = []

        for sensor in self._contact_sensors.values():
            net_force = wp_to_jax(sensor.net_force)

            if self._include_total:
                total_force = net_force[:, 0, :]
            else:
                total_force = net_force.sum(axis=1)

            force_magnitude = jnp.linalg.norm(total_force, axis=-1)

            is_contact = force_magnitude > 1.0
            contact_states.append(is_contact)

        if not contact_states:
            return jnp.zeros((self.num_envs, self.num_shapes), dtype=jnp.bool_)

        all_contacts = jnp.concatenate(contact_states, axis=0)
        return all_contacts.reshape(self.num_envs, self.num_shapes)

    @property
    def is_contact(self) -> jax.Array:
        """Current contact state for all tracked shapes."""
        return self._compute_is_contact()

    @property
    def shape_names(self) -> list[str]:
        return self._shape_names

    @property
    def contact_force(self) -> jax.Array:
        """Contact force for all tracked shapes."""
        if self.num_shapes == 0:
            return jnp.zeros((self.num_envs, 0, 3))

        sensor = list(self._contact_sensors.values())[0]
        net_force = wp_to_jax(sensor.net_force)

        if self._include_total:
            total_force = net_force[:, 0, :]
        else:
            total_force = net_force.sum(axis=1)

        return total_force.reshape(self.num_envs, self.num_shapes, 3)

    def get_shape_indices(
        self,
        patterns: str | list[str],
        use_regex: bool = False,
        preserve_order: bool = False,
    ) -> list[int]:
        """Get indices of shapes matching the pattern."""
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
        return self.get_shape_indices(links, use_regex=True, preserve_order=preserve_order)

    def advance(self) -> None:
        """Advance contact timing based on current contact states."""
        if self.num_shapes == 0:
            return

        is_contact = self._compute_is_contact()

        is_landing = ~self._prev_is_contact & is_contact
        is_liftoff = self._prev_is_contact & ~is_contact

        self.last_air_time = jnp.where(
            is_landing, self.current_air_time, self.last_air_time
        )

        self.last_contact_time = jnp.where(
            is_liftoff, self.current_contact_time, self.last_contact_time
        )

        self.current_contact_time = jnp.where(
            is_contact,
            self.current_contact_time + self.dt,
            jnp.zeros_like(self.current_contact_time)
        )
        self.current_air_time = jnp.where(
            ~is_contact,
            self.current_air_time + self.dt,
            jnp.zeros_like(self.current_air_time)
        )

        self._prev_is_contact = is_contact

    def compute_first_contact(self, abs_tol: float = 1e-6) -> jax.Array:
        """Detect shapes that just made contact within the last dt."""
        if self.num_shapes == 0:
            return jnp.zeros((self.num_envs, 0), dtype=jnp.bool_)

        is_contact = self.current_contact_time > 0
        just_landed = self.current_contact_time < (self.dt + abs_tol)
        return is_contact & just_landed

    def compute_first_air(self, abs_tol: float = 1e-6) -> jax.Array:
        """Detect shapes that just lifted off within the last dt."""
        if self.num_shapes == 0:
            return jnp.zeros((self.num_envs, 0), dtype=jnp.bool_)

        is_air = self.current_air_time > 0
        just_lifted = self.current_air_time < (self.dt + abs_tol)
        return is_air & just_lifted

    def reset(self, env_ids) -> None:
        """Reset timing buffers for specified environments."""
        if self.num_shapes == 0 or env_ids is None or len(env_ids) == 0:
            return

        self.current_air_time = self.current_air_time.at[env_ids].set(0.0)
        self.current_contact_time = self.current_contact_time.at[env_ids].set(0.0)
        self.last_air_time = self.last_air_time.at[env_ids].set(0.0)
        self.last_contact_time = self.last_contact_time.at[env_ids].set(0.0)
        self._prev_is_contact = self._prev_is_contact.at[env_ids].set(False)

    def __str__(self) -> str:
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
