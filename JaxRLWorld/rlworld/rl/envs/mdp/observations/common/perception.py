"""Observation terms that read a camera instead of the state vector.

Ported from mjlab's ``tasks/manipulation/mdp/observations.py``. The
normalisation is theirs, value for value, so a policy trained here and a
policy trained there see the same picture.

An image term returns ``(B, C, H, W)`` — channel first, the layout every
CNN in torch expects — while every other term in this package returns
``(B, D)``. It therefore cannot share an observation group with the
proprioception terms: put it in its own group, whose ``concatenate_dim``
is the channel axis, so several cameras (or a depth map and a mask)
stack into one tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def camera_depth(
    env: World,
    sensor_name: str,
    cutoff_distance: float,
    min_depth: float = 0.01,
    near_clip: float = 0.0,
) -> torch.Tensor:
    """Depth in CNN-compatible format, normalised to [0, 1].

    Depth arrives forward-projected — the distance along the camera's
    view axis, not along the ray to the pixel — which is what a real
    D405 reports, so the numbers a policy learns here are the numbers it
    reads on hardware.

    ``cutoff_distance`` is the far plane: everything beyond it saturates
    at 1.0. Set it to roughly the depth of the workspace, not to the
    room — the resolution a policy gets is spread across this range.

    ``near_clip`` discards surfaces nearer than the real sensor can
    measure — a D405 cannot report anything closer than about 7 cm, so
    below that both simulators are inventing, and they invent
    differently: one renders the camera's own housing where the other
    clips it away. Applied here rather than in a backend so BOTH see the
    same rule, which is what makes their images comparable and what a
    policy would meet on hardware.

    A pixel that hits nothing comes back as 0.0 and is lifted to
    ``min_depth``, i.e. it reads as "something right at the lens" rather
    than "nothing out there". mjlab's own wrist-camera task is unharmed
    by that because the table fills the frame at all times; a camera that
    can see past the workspace needs the far plane handled deliberately.

    Returns:
        ``(B, 1, H, W)``, freshly allocated — clamp is out of place, so
        the result never aliases the renderer's buffer, which is
        overwritten on the next render.
    """
    sensor = env.scene_manager.sensors[sensor_name]
    depth_data = sensor.data.depth  # (B, H, W, 1)
    if depth_data is None:
        raise ValueError(f"Camera {sensor_name!r} has no depth data. Add 'depth' to its data_types.")
    depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
    if near_clip > 0.0:
        depth_data = torch.where(depth_data < near_clip, torch.zeros_like(depth_data), depth_data)
    depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
    return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


def camera_rgb(env: World, sensor_name: str) -> torch.Tensor:
    """RGB in CNN-compatible format, normalised to [0, 1].

    Returns:
        ``(B, 3, H, W)``, freshly allocated by the divide.
    """
    sensor = env.scene_manager.sensors[sensor_name]
    rgb_data = sensor.data.rgb  # (B, H, W, 3)
    if rgb_data is None:
        raise ValueError(f"Camera {sensor_name!r} has no RGB data. Add 'rgb' to its data_types.")
    return rgb_data.permute(0, 3, 1, 2).float() / 255.0
