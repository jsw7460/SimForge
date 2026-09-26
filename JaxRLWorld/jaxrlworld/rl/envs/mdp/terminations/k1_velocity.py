"""Termination terms of the Booster K1 velocity recipe.

The one term the shared library lacks is a fall condition that fires
probabilistically rather than the instant the tilt limit is crossed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from jaxrlworld.rl.configs.terminations import TerminationResult

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def tilt_angle(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Angle between the root link's up axis and world up, in radians.

    Zero when upright, pi when inverted. The body-frame projected gravity is
    the negated up vector, so its z component is the cosine of this angle.
    """
    gravity_b = env.get_entity_data(asset_cfg.name).projected_gravity_b
    return torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0)).abs()


def stochastic_bad_orientation(
    env: World,
    limit_angle: float,
    probability: float = 0.01,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> TerminationResult:
    """Terminate with a fixed per-step probability while tilted past the limit.

    A deterministic tilt limit makes the episode end the first control step the
    robot crosses it, which cuts off any recovery the policy might still have
    learned and puts a hard discontinuity at the boundary. Drawing once per
    environment per step instead turns the limit into a survival rate: a
    persistently over-tilted robot lasts an expected ``1 / probability`` steps,
    while one that rights itself is immediately out of danger again. No
    countdown is carried between steps.

    Args:
        limit_angle: Tilt from upright, in radians, past which the draw happens.
        probability: Per-step termination probability while over the limit.
            ``1.0`` makes the term deterministic and skips the draw, which is
            what the parity diagnostics use to strip this randomness.

    Returns:
        ``TerminationResult`` of shape ``(num_envs,)``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    tilted = tilt_angle(env, asset_cfg) > limit_angle
    if probability >= 1.0:
        return TerminationResult(tilted)
    draw = torch.rand(tilted.shape, device=tilted.device) < probability
    return TerminationResult(tilted & draw)
