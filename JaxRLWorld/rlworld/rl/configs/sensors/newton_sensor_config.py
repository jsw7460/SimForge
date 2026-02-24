"""Newton Sensor Configuration.

This module defines configuration for sensors in Newton environments.
Newton supports IMU, Contact, FrameTransform, Raycast, and TiledCamera sensors.

Sensors in Newton are attached to "sites" which are created on bodies during
entity registration.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import newton


class NewtonSensorType(Enum):
    """Available sensor types in Newton."""
    IMU = "imu"
    CONTACT = "contact"
    FRAME_TRANSFORM = "frame_transform"
    RAYCAST = "raycast"
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
class NewtonContactSensorConfig(NewtonSensorConfig):
    """Configuration for Newton Contact sensor.

    Contact sensors detect contacts between specified geometries.
    Exactly one of sensing_obj_bodies or sensing_obj_shapes must be specified.

    Example:
        # Body-based (recommended for URDF)
        contact_config = NewtonContactSensorConfig(
            sensor_name="foot_contacts",
            entity_name="robot",
            sensing_obj_bodies=".*_foot",
            use_regex=True,
        )

        # Shape-based (for USD or custom shapes)
        contact_config = NewtonContactSensorConfig(
            sensor_name="foot_contacts",
            entity_name="robot",
            sensing_obj_shapes=".*_foot.*",
            use_regex=True,
        )
    """
    sensor_type: NewtonSensorType = NewtonSensorType.CONTACT

    # Sensing objects (exactly one must be specified)
    sensing_obj_bodies: str | list[str] | None = None
    sensing_obj_shapes: str | list[str] | None = None

    # Counterparts (optional, at most one)
    counterpart_bodies: str | list[str] | None = None
    counterpart_shapes: str | list[str] | None = None

    include_total: bool = True
    use_regex: bool = False


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


@dataclass
class NewtonRaycastSensorConfig(NewtonSensorConfig):
    """Configuration for Newton Raycast sensor.

    Casts rays from specified sites and returns hit information.
    """
    sensor_type: NewtonSensorType = NewtonSensorType.RAYCAST
    site_names: list[str] = field(default_factory=list)
    ray_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)  # Local direction
    max_distance: float = 10.0

    def create_sensor(self, model: "newton.Model", site_indices: list[int]) -> "newton.sensors.SensorRaycast":
        """Create the actual Raycast sensor object."""
        import newton
        import warp as wp
        return newton.sensors.SensorRaycast(
            model,
            site_indices,
            ray_direction=wp.vec3(*self.ray_direction),
            max_distance=self.max_distance,
        )