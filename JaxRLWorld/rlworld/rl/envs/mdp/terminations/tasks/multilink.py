from __future__ import annotations

from typing import TYPE_CHECKING

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.configs.terminations import TerminationResult

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def end_effector_below_ground(
    env: GenesisEnv, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR, z_threshold: float = 0.0
) -> TerminationResult:
    """Terminate if a selected link goes below ground.

    ``asset_cfg.body_names`` selects the link(s) to check; an environment
    terminates if **any** selected link's z drops below ``z_threshold``.
    If the selector specifies no ``body_names`` the entity's last link is
    used (legacy default).

    Args:
        env: The environment.
        asset_cfg: Selector identifying the robot entity / link(s).
        z_threshold: Z coordinate threshold (default 0.0 for ground level).

    Returns:
        TerminationResult indicating which environments should be terminated.
    """
    entity = env.scene_manager[asset_cfg.name]

    if asset_cfg.body_ids is not None:
        z_coord = entity.get_links_pos(links_idx_local=asset_cfg.body_ids.tolist())[:, :, 2]
        return TerminationResult((z_coord < z_threshold).any(dim=1))

    z_coord = entity.get_links_pos()[:, -1, 2]  # last link's z
    return TerminationResult(z_coord < z_threshold)
