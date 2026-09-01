"""Observation terms for reaching for an object and bringing it somewhere.

Ported from mjlab's ``tasks/manipulation/mdp/observations.py``.

Everything here is a RELATIVE quantity, expressed in a frame that moves
with the robot — the vector from the grasp point to the object, the
vector from the object to its goal. None of them is a world position.
That is deliberate: a policy given world coordinates learns where the
table is in this scene, and stops working when the arm is bolted down
somewhere else. A policy given "the object is 8 cm ahead of your hand"
transfers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from jaxrlworld.rl.envs.mdp.entity_points import entity_point_w
from jaxrlworld.rl.utils.quat_utils import quat_inv_wxyz, quat_rotate_wxyz

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def _site(env: World, asset_cfg: ResolvedEntity) -> tuple[torch.Tensor, torch.Tensor]:
    """World position and orientation of the selector's single site."""
    if asset_cfg.site_ids is None:
        raise ValueError(
            f"Observation term needs a site on entity {asset_cfg.name!r}: pass "
            "SceneEntitySelector(name=..., site_names=('grasp_site',))."
        )
    data = env.get_entity_data(asset_cfg.name)
    return (
        data.site_pos_w_by_ids(asset_cfg.site_ids)[:, 0],
        data.site_quat_w_by_ids(asset_cfg.site_ids)[:, 0],
    )


def ee_to_object_distance(
    env: World,
    object_cfg: ResolvedEntity,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Vector from the grasp point to the object, in the robot's base frame."""
    ee_pos_w, _ = _site(env, asset_cfg)
    obj_pos_w = entity_point_w(env, object_cfg)
    base_quat_w = env.get_entity_data(asset_cfg.name).root_link_quat_w
    return quat_rotate_wxyz(quat_inv_wxyz(base_quat_w), obj_pos_w - ee_pos_w)


def object_to_goal_distance(
    env: World,
    object_cfg: ResolvedEntity,
    command_name: str,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Vector from the object to its goal, in the robot's base frame."""
    command = env.command_manager.get_term(command_name)
    obj_pos_w = entity_point_w(env, object_cfg)
    base_quat_w = env.get_entity_data(asset_cfg.name).root_link_quat_w
    return quat_rotate_wxyz(quat_inv_wxyz(base_quat_w), command.target_pos - obj_pos_w)


def ee_velocity(
    env: World,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Grasp-point linear velocity, in the grasp point's own frame."""
    if asset_cfg.site_ids is None:
        raise ValueError(
            f"ee_velocity needs a site on entity {asset_cfg.name!r}: pass "
            "SceneEntitySelector(name=..., site_names=('grasp_site',))."
        )
    data = env.get_entity_data(asset_cfg.name)
    vel_w = data.site_lin_vel_w_by_ids(asset_cfg.site_ids)[:, 0]
    quat_w = data.site_quat_w_by_ids(asset_cfg.site_ids)[:, 0]
    return quat_rotate_wxyz(quat_inv_wxyz(quat_w), vel_w)


def target_position(
    env: World,
    command_name: str,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Vector from the grasp point to the goal, in the grasp point's frame."""
    command = env.command_manager.get_term(command_name)
    ee_pos_w, ee_quat_w = _site(env, asset_cfg)
    return quat_rotate_wxyz(quat_inv_wxyz(ee_quat_w), command.target_pos - ee_pos_w)


def object_height(
    env: World,
    object_cfg: ResolvedEntity,
    reference_height: float,
) -> torch.Tensor:
    """How far the object is above a reference height, e.g. the table top.

    Not in mjlab's set, where the object starts on the floor and its own
    world z answers this. Here the table's height would otherwise appear
    as a constant offset the policy has to learn to subtract.
    """
    return entity_point_w(env, object_cfg)[:, 2:3] - reference_height
