from dataclasses import dataclass, field
from typing import Any, Type

import genesis as gs
from genesis.engine.sensors.base_sensor import Sensor


@dataclass
class SensorConfig:
    """
    Configuration for a sensor (declaration phase).

    This config uses entity names (strings) instead of indices,
    making it easier to define sensors before the scene is built.
    """

    entity_name: str  # Name of the entity (e.g., "robot")
    link_name: str = "base"  # Name of the link (e.g., "base", "hand")
    sensor_class: Type[gs.sensors.SensorOptions] = None  # Sensor class (IMU, Contact, etc.)
    sensor_params: dict[str, Any] = field(default_factory=dict)  # Additional parameters

    def create_sensor(
        self,
        scene: gs.Scene,
        entity
    ) -> Sensor:
        """
        Create the actual sensor object after the scene is built.

        This method is called after scene.build(), when entity_idx
        and link_idx_local are available.

        Parameters
        ----------
        scene : gs.Scene
            The built scene
        entity : RigidEntity
            The entity to attach the sensor to

        Returns
        -------
        sensor : gs.sensors.Sensor
            The created sensor object
        """
        # Get the link from the entity
        link = entity.get_link(self.link_name)

        # Create Genesis sensor options with actual indices
        sensor_options = self.sensor_class(
            entity_idx=entity.idx,
            link_idx_local=link.idx_local,
            **self.sensor_params  # User-provided additional parameters
        )

        # Add sensor to the scene
        sensor = scene.add_sensor(sensor_options)

        return sensor
