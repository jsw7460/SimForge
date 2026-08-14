"""Termination terms for the K1 joystick task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.configs.terminations import TerminationResult

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def gravity_z_positive(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> TerminationResult:
    """Terminate when the trunk tilts past horizontal.

    Upstream condition ``upvector_z < 0``; since the body-frame
    projected gravity is the negated upvector, this is
    ``projected_gravity_b.z > 0``.
    """
    g = env.get_entity_data(asset_cfg.name).projected_gravity_b
    return TerminationResult(g[:, 2] > 0.0)
