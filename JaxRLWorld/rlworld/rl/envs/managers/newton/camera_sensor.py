"""Newton backend for :class:`CameraSensorCfg`.

Newton renders by raytracing a batch of cameras against the scene's
BVHs. Two things it does NOT do, which mjlab does for free, and which
this adapter therefore has to do by hand:

* **carry the camera along with the link.** MuJoCo moves an MJCF
  ``<camera>`` element with the body it lives in. Newton takes a
  camera-to-world transform per world and renders exactly where it is
  told, so the transform is recomputed from the link's pose every frame.
* **keep the BVHs current.** Ray hits are found against acceleration
  structures built for the state at finalize time. Rendering without
  refitting them first returns a picture of where the robot USED to be —
  a plausible image of the wrong moment, which no shape check catches.

The output is presented the way mjlab presents it — ``(num_envs, H, W,
1)`` of forward-projected metres with 0.0 for "hit nothing" — so an
observation term reads the same tensor on either backend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from newton import ShapeFlags

from rlworld.rl.configs.sensors.camera_sensor_config import CameraOffset, CameraSensorCfg, MjcfCameraPlacement
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.utils.quat_utils import quat_mul_wxyz, quat_rotate_wxyz, wxyz_to_xyzw

if TYPE_CHECKING:
    from rlworld.rl.envs import World

__all__ = ["NewtonCameraSensor", "NewtonCameraData"]


def _ray_optics(cfg: CameraSensorCfg, optics: MjcfCameraPlacement) -> dict:
    """The arguments that make Newton's rays match what mjwarp renders.

    An MJCF camera can state its optics as a sensor size and a focal
    length instead of an angle, and the D405 on the YAM does: 3.896 x
    2.14 mm behind a 1.93 mm lens, which is 90.5 degrees across and 58.0
    down. A real sensor is not square.

    mjwarp does not render that. Where the image's aspect ratio differs
    from the sensor's it CROPS the sensor to match
    (``render_util.compute_ray``), so a 32x32 request turns the 3.896 mm
    width into 2.14 mm and the horizontal field of view collapses from
    90.5 to 58 degrees. Handing Newton the sensor's true width instead
    made the two backends disagree noticeably WORSE, which is what
    identified this: the mismatch is not in the asset, it is in what
    mjwarp does with it.

    So the crop is reproduced here rather than the datasheet. The image
    is then the one both backends actually produce — and for a square
    image that is the vertical angle applied both ways, which is why
    passing the fovy alone had looked right.
    """
    if optics.sensorsize is None or optics.focal is None:
        return {"camera_fovs": math.radians(optics.fovy)}
    if optics.focal[0] != optics.focal[1]:
        raise NotImplementedError(
            f"Camera {cfg.name!r} has different focal lengths across and down "
            f"({optics.focal}); Newton's pinhole helper takes a single focal length."
        )

    sensor_width, sensor_height = optics.sensorsize
    target_aspect = cfg.width / cfg.height
    sensor_aspect = sensor_width / sensor_height
    if target_aspect > sensor_aspect:
        sensor_height = sensor_width / target_aspect
    elif target_aspect < sensor_aspect:
        sensor_width = sensor_height * target_aspect

    return {
        "focal_length": optics.focal[1],
        "horizontal_aperture": sensor_width,
        "vertical_aperture": sensor_height,
    }


class NewtonCameraData:
    """The mjlab-shaped view of one camera's rendered channels."""

    def __init__(self):
        self.depth: torch.Tensor | None = None
        self.rgb: torch.Tensor | None = None


class NewtonCameraSensor:
    """One camera riding on one link, rendered by ``SensorTiledCamera``."""

    def __init__(
        self,
        env: World,
        cfg: CameraSensorCfg,
        link_name: str,
        offset: CameraOffset,
        optics: MjcfCameraPlacement,
    ):
        if offset.convention != "opengl":
            raise NotImplementedError(
                f"Camera {cfg.name!r} asks for the {offset.convention!r} convention. Newton's ray kernel "
                "takes the camera forward as -Z, which is the 'opengl' convention and the only one supported."
            )
        unsupported = set(cfg.data_types) - {"depth"}
        if unsupported:
            raise NotImplementedError(
                f"Camera {cfg.name!r} asks Newton for {sorted(unsupported)}; only 'depth' is wired up here."
            )

        self.cfg = cfg
        self.data = NewtonCameraData()
        self._env = env
        self._scene = env.scene_manager
        self._entity_name = cfg.entity_name
        # Resolved on the first render: cameras are created while the
        # scene is still being built, and the per-entity robot data that
        # knows body names does not exist until after that.
        self._link_name = link_name
        self._body_index: int | None = None

        device = env.device
        self._offset_pos = torch.tensor(offset.pos, dtype=torch.float32, device=device)
        self._offset_quat = torch.tensor(offset.quat, dtype=torch.float32, device=device)

        self._apply_visibility(cfg.visible_geometry)
        self._sensor = newton.sensors.SensorTiledCamera(self._scene.model)
        utils = self._sensor.utils
        self._rays = utils.compute_camera_rays_pinhole(cfg.width, cfg.height, **_ray_optics(cfg, optics))
        self._forward_depth = utils.create_forward_depth_image_output(cfg.width, cfg.height, camera_count=1)

        # (camera_count, world_count), the order SensorTiledCamera.update
        # documents for transforms — note the outputs are the other way
        # round, (world_count, camera_count, H, W).
        self._transform_torch = torch.zeros((1, env.num_envs, 7), dtype=torch.float32, device=device)
        self._transforms = wp.from_torch(self._transform_torch.view(-1, 7), dtype=wp.transformf).reshape(
            (1, env.num_envs)
        )

        # (worlds, cameras, H, W) -> (num_envs, H, W, 1), mjlab's layout.
        depth_torch = wp.to_torch(self._forward_depth)
        self.data.depth = depth_torch[:, 0].unsqueeze(-1)

    def _apply_visibility(self, visible_geometry: str) -> None:
        """Show the camera the same shapes mjlab is told to draw.

        MuJoCo assets describe a robot twice — visual meshes that cannot
        collide, and coarse colliders — and Newton renders whatever
        carries the VISIBLE flag, which after an MJCF import is both.
        mjlab picks between them by geom group; here the same choice is
        made on the flag Newton already keeps, so the classification is
        exact rather than a convention about group numbers.

        This changes the model, which every camera and viewer shares, so
        two cameras asking for different geometry is refused rather than
        silently resolved in favour of whichever was built last.
        """
        if visible_geometry == "all":
            return

        model = self._scene.model
        previous = getattr(model, "_rlworld_camera_visibility", None)
        if previous is not None and previous != visible_geometry:
            raise ValueError(
                f"Camera {self.cfg.name!r} wants to see {visible_geometry!r} geometry, but another camera "
                f"already set the model's shapes to {previous!r}. Newton's visibility flag is per shape, "
                "not per camera, so the two cannot both be honoured."
            )
        model._rlworld_camera_visibility = visible_geometry

        flags = wp.to_torch(model.shape_flags)
        collides = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        should_see = collides if visible_geometry == "collision" else ~collides
        flags |= int(ShapeFlags.VISIBLE)
        flags[~should_see] &= ~int(ShapeFlags.VISIBLE)

    def _write_camera_transforms(self) -> None:
        """Place the camera on the link, this frame."""
        robot_data = self._env.get_robot_data(self._entity_name)
        if self._body_index is None:
            # Newton labels bodies by the MJCF's own body names, which is
            # what resolve_mjcf_camera returns. Resolved through the cache
            # rather than find_body_index because that one keeps the first
            # of several matches, and a camera silently riding the wrong
            # one of two identically-named bodies is the whole failure
            # mode this sensor has.
            indices = get_cache(self._env).get_body_indices(self._link_name)
            if len(indices) != 1:
                raise ValueError(
                    f"Camera {self.cfg.name!r} rides body {self._link_name!r}, which matches "
                    f"{len(indices)} bodies in the Newton model. It must name exactly one."
                )
            self._body_index = indices[0]
        link_pos = robot_data.body_pos_w_all[:, self._body_index]
        link_quat = robot_data.body_quat_w_all[:, self._body_index]

        camera_pos = link_pos + quat_rotate_wxyz(link_quat, self._offset_pos.expand_as(link_pos))
        camera_quat = quat_mul_wxyz(link_quat, self._offset_quat.expand_as(link_quat))

        # warp transforms carry the quaternion as xyzw; the protocol
        # hands out wxyz.
        self._transform_torch[0, :, 0:3] = camera_pos
        self._transform_torch[0, :, 3:7] = wxyz_to_xyzw(camera_quat)

    def render(self) -> None:
        """Refit the BVHs to the current state, then raytrace."""
        state = self._scene.state_0
        self._write_camera_transforms()
        self._scene.model.bvh_refit_shapes(state)
        self._sensor.update(
            state,
            self._transforms,
            self._rays,
            forward_depth_image=self._forward_depth,
        )
