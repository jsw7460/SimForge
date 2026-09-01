"""Which point on an entity an MDP term should aim at.

An object's frame origin is the obvious answer and, for a cube, the right
one: it is the middle of the thing, it is where a gripper closes, and it
is what "bring the cube here" means. For anything shaped, it stops being
either. A pair of tongs has its origin at the pivot, which is one end of
the tool, is where the two arms already touch, and — lying on a bench —
sits a few millimetres off the surface, so a reward aiming a gripper
there is aiming it through the bench.

So a term takes a SELECTOR rather than a name. Name a site on the entity
and that site is the point; name none and the point is the root, which is
what every existing task was already doing. The distinction is the
object's, not the term's: "reach it", "bring it somewhere" and "has it
fallen" all want the same answer for a given object, and they should not
each be told separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World


def entity_point_w(env: World, entity_cfg: ResolvedEntity) -> torch.Tensor:
    """World position of the point that represents ``entity_cfg``.

    The selector's site when it names exactly one, the entity's root
    otherwise.

    Args:
        env: The environment.
        entity_cfg: Selector naming the entity, and optionally one site
            on it.

    Returns:
        ``(num_envs, 3)``.

    Raises:
        ValueError: If the selector names more than one site. Two sites
            are two points and there is no defensible way to pick one —
            an average would be a third point that is on neither.
    """
    data = env.get_entity_data(entity_cfg.name)
    if entity_cfg.site_ids is None:
        return data.root_link_pos_w
    if len(entity_cfg.site_ids) != 1:
        raise ValueError(
            f"Selector for entity {entity_cfg.name!r} names {len(entity_cfg.site_ids)} sites; "
            "a term that aims at a point needs exactly one, or none for the root."
        )
    return data.site_pos_w_by_ids(entity_cfg.site_ids)[:, 0]
