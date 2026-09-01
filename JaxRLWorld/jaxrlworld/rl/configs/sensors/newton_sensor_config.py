"""Newton Sensor Configuration.

This module defines configuration for sensors in Newton environments.
Newton supports IMU, Contact, FrameTransform, and TiledCamera sensors.

Sensors in Newton are attached to "sites" which are created on bodies during
entity registration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import newton


class NewtonSensorType(Enum):
    """Available sensor types in Newton."""

    IMU = "imu"
    FRAME_TRANSFORM = "frame_transform"
    TILED_CAMERA = "tiled_camera"


@dataclass
class NewtonSensorConfig:
    """Base configuration for a Newton sensor.

    This is an abstract base config. Use specific sensor configs below.

    Example:
        imu_config = NewtonIMUSensorConfig(
            sensor_name="base_imu",
            entity_name="robot",
            site_names=["imu_site"],
        )
    """

    sensor_name: str  # Unique name for this sensor
    entity_name: str  # Entity this sensor is attached to
    sensor_type: NewtonSensorType = NewtonSensorType.IMU


@dataclass
class NewtonIMUSensorConfig(NewtonSensorConfig):
    """Configuration for Newton IMU sensor.

    IMU sensors measure linear acceleration and angular velocity at specified sites.
    The sites must be created on the entity during registration.

    Example:
        imu_config = NewtonIMUSensorConfig(
            sensor_name="base_imu",
            entity_name="robot",
            site_names=["base_imu_site"],  # Must match site defined in NewtonEntityConfig
        )
    """

    sensor_type: NewtonSensorType = NewtonSensorType.IMU
    site_names: list[str] = field(default_factory=list)  # Names of sites to attach IMU

    @staticmethod
    def create_sensor(model: "newton.Model", site_indices: list[int]) -> "newton.sensors.SensorIMU":
        """Create the actual IMU sensor object.

        Args:
            model: The Newton model
            site_indices: List of site shape indices

        Returns:
            The created SensorIMU object
        """
        import newton

        return newton.sensors.SensorIMU(model, site_indices)


@dataclass
class NewtonFrameTransformSensorConfig(NewtonSensorConfig):
    """Configuration for Newton FrameTransform sensor.

    Measures the transform (position and orientation) of specified sites.
    """

    sensor_type: NewtonSensorType = NewtonSensorType.FRAME_TRANSFORM
    site_names: list[str] = field(default_factory=list)

    @staticmethod
    def create_sensor(model: "newton.Model", site_indices: list[int]) -> "newton.sensors.SensorFrameTransform":
        """Create the actual FrameTransform sensor object."""
        import newton

        return newton.sensors.SensorFrameTransform(model, site_indices)
