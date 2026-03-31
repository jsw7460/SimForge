from dataclasses import dataclass, field
from typing import Any, Callable

from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class SensorConfig:
    """Configuration for a sensor (declaration phase).

    Uses entity names (strings) instead of indices, making it easier to
    define sensors before the scene is built.

    ``sensor_class`` accepts a class or ``"module:ClassName"`` string.
    Automatically serialized to string by ``recursive_to_dict()``.
    """

    entity_name: str  # Name of the entity (e.g., "robot")
    link_name: str = "base"  # Name of the link (e.g., "base", "hand")
    sensor_class: Callable | str | None = None  # gs.sensors.IMU or "genesis.engine.sensors:IMU"
    sensor_params: dict[str, Any] = field(default_factory=dict)

    def create_sensor(self, scene, entity):
        """Create the actual sensor object after the scene is built."""
        link = entity.get_link(self.link_name)

        sensor_cls = self.sensor_class
        if isinstance(sensor_cls, str):
            sensor_cls = resolve_callable(sensor_cls)

        sensor_options = sensor_cls(
            entity_idx=entity.idx,
            link_idx_local=link.idx_local,
            **self.sensor_params,
        )

        sensor = scene.add_sensor(sensor_options)
        return sensor
