from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv

from rlworld.rl.configs.terminations import TerminationResult


def end_effector_below_ground(
    env: GenesisEnv, entity_name: str = "robot", link_name: str = None, z_threshold: float = 0.0
) -> TerminationResult:
    """Terminate if the end effector (last link) goes below ground.

    Args:
        env: The environment.
        entity_name: Name of the robot entity.
        link_name: Name of the link to check. If None, uses the last link.
        z_threshold: Z coordinate threshold (default 0.0 for ground level).

    Returns:
        TerminationResult indicating which environments should be terminated.
    """
    entity = env.scene_manager[entity_name]

    if link_name is not None:
        link = entity.get_link(link_name)
        link_pos = link.get_pos()  # (num_envs, 3)
        z_coord = link_pos[:, 2]
    else:
        links_pos = entity.get_links_pos()  # (num_envs, num_links, 3)
        z_coord = links_pos[:, -1, 2]  # Last link's z

    return TerminationResult(z_coord < z_threshold)
