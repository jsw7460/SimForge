"""Genesis backend for :class:`CameraSensorCfg`.

Genesis renders batched cameras through Madrona, which has to be asked
for at scene construction: an ordinary Genesis scene has no batch
renderer and a camera added to it would be drawn one environment at a
time. It also insists every camera in the scene share a resolution.

Unlike Newton, Genesis carries a camera along with a link itself —
``attach`` takes the link and a 4x4 offset, and ``move_to_attach``
recomputes the pose for every environment. So this adapter does not
compose transforms; it hands over the same offset the other backends
read out of the MJCF and asks Genesis to follow.

Depth comes back forward-projected: with the raytracer Genesis converts
ray distance to plane distance in ``distance_center_to_plane``, and the
rasterizer produces plane distance to begin with. A ray that hits
nothing comes back as +inf, where mjlab and Newton report 0.0, so it is
converted here. The output is presented as mjlab presents it —
``(num_envs, H, W, 1)`` in metres, 0.0 for no hit — so an observation
term reads the same tensor on any backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from rlworld.rl.configs.sensors.camera_sensor_config import CameraOffset, CameraSensorCfg, MjcfCameraPlacement

if TYPE_CHECKING:
    from rlworld.rl.envs import World

__all__ = ["GenesisCameraSensor", "GenesisCameraData"]

_NEAR_PLANE = 0.01
"""Metres. Below anything the observation's own near clip discards, so
that clip is the only rule deciding what is too close to report."""

_FAR_PLANE = 1000.0
"""Metres. mjlab and Newton raytrace the ground plane out to hundreds of
metres; a nearer far plane would make Genesis disagree with them about
the sky rather than about the scene."""


class GenesisCameraData:
    """The mjlab-shaped view of one camera's rendered channels."""

    def __init__(self):
        self.depth: torch.Tensor | None = None
        self.rgb: torch.Tensor | None = None


def _offset_matrix(offset: CameraOffset) -> np.ndarray:
    """The 4x4 link-to-camera transform Genesis's ``attach`` expects.

    Built in numpy rather than with the torch quaternion helpers: this
    is a constant computed once at build time, and Genesis sets torch's
    default device to the GPU, so a tensor made here would land there
    and refuse to become an array.
    """
    w, x, y, z = offset.quat
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = offset.pos
    return matrix


class GenesisCameraSensor:
    """One camera riding on one link, drawn by Genesis's batch renderer."""

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
                f"Camera {cfg.name!r} asks for the {offset.convention!r} convention; only 'opengl' is supported."
            )
        unsupported = set(cfg.data_types) - {"depth"}
        if unsupported:
            raise NotImplementedError(
                f"Camera {cfg.name!r} asks Genesis for {sorted(unsupported)}; only 'depth' is wired up here."
            )

        self.cfg = cfg
        self.data = GenesisCameraData()
        self._env = env
        self._scene = env.scene_manager

        entity = self._scene.entities[cfg.entity_name]
        link = entity.get_link(link_name)

        # fov is the VERTICAL field of view in degrees, the same thing
        # MuJoCo writes as fovy, so the value resolved from the MJCF
        # carries over unchanged.
        # near/far are stated rather than left to add_camera's defaults
        # of 0.1 and 20 m. 0.1 sits ABOVE the near clip the observation
        # applies, so Genesis alone would report nothing between the two
        # and the backends would differ by configuration; and mjlab and
        # Newton see the ground plane out to hundreds of metres, which a
        # 20 m far plane would cut off.
        self._camera = self._scene.scene.add_camera(
            res=(cfg.width, cfg.height),
            fov=optics.fovy,
            near=_NEAR_PLANE,
            far=_FAR_PLANE,
            GUI=False,
        )
        self._camera.attach(link, _offset_matrix(offset))

        # Allocated now, not on the first render: the runner asks for the
        # observation's shape while it is building its networks, which is
        # before anything has been rendered or even reset. mjlab and
        # Newton both hand out a zeroed buffer from construction, so a
        # camera that has None there is the odd one out and fails only on
        # the training path, never in a diag that resets first.
        self.data.depth = torch.zeros((env.num_envs, cfg.height, cfg.width, 1), dtype=torch.float32, device=env.device)

    def render(self) -> None:
        """Follow the link, then draw.

        Two separate staleness traps, both keyed on the scene clock and
        both silent: the visualizer skips ``move_to_attach`` when time
        has not advanced (``visualizer.py:225``), and the batch renderer
        returns a cached image on the same condition
        (``batch_renderer.py:378``). One leaves the camera behind, the
        other leaves the picture behind.
        """
        self._camera.move_to_attach()
        # force_render because the batch renderer only redraws when the
        # scene clock has advanced (``batch_renderer.py:378``) and hands
        # back a cached image otherwise. An observation is produced
        # between physics steps — and the cross-sim diag imposes state
        # without stepping at all — so without this the camera returns
        # the picture it drew for an earlier state, which is a plausible
        # image of the wrong moment and nothing about its shape or range
        # gives it away.
        _, depth, _, _ = self._camera.render(rgb=False, depth=True, force_render=True)
        depth_tensor = depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
        # Genesis reports a ray that hit nothing as +inf; mjlab and
        # Newton report 0.0, and the observation term's contract is 0.0.
        # Left alone it poisons everything downstream — a mean becomes
        # inf, a quantile becomes nan, and the normalised image saturates
        # a whole region to 1.0 for a reason no shape check would show.
        depth_tensor = torch.where(torch.isfinite(depth_tensor), depth_tensor, torch.zeros_like(depth_tensor))
        # (num_envs, H, W) -> (num_envs, H, W, 1), mjlab's layout.
        self.data.depth = depth_tensor.reshape(self._env.num_envs, self.cfg.height, self.cfg.width, 1)
