"""Reward terms for picking an object up and bringing it somewhere.

Ported from mjlab's ``tasks/manipulation/mdp/rewards.py``. Simulator-
agnostic: everything is read through ``RobotData`` / ``RigidObjectData``
and the site frames, so the same terms run on Newton, Genesis and mjlab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def _ee_pos_w(env: World, asset_cfg: ResolvedEntity) -> torch.Tensor:
    """World position of the selector's single site — the grasp point."""
    if asset_cfg.site_ids is None:
        raise ValueError(
            f"Reward term needs a site on entity {asset_cfg.name!r}: pass "
            "SceneEntitySelector(name=..., site_names=('grasp_site',))."
        )
    return env.get_entity_data(asset_cfg.name).site_pos_w_by_ids(asset_cfg.site_ids)[:, 0]


def staged_position_reward(
    env: World,
    command_name: str,
    object_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Reaching, multiplied by one plus bringing.

    Two Gaussian kernels: how near the grasp point is to the object, and
    how near the object is to its goal. Multiplied rather than added, so
    the bringing bonus is worth nothing until the arm is at the object.

    That ordering is the point. Bringing an object to a goal is a reward
    a policy cannot collect by accident, and a sum would let it collect
    the reaching half forever without ever closing the gripper. The
    product gives a gradient that only leads somewhere by going through
    the object first.
    """
    command = env.command_manager.get_term(command_name)
    ee_pos_w = _ee_pos_w(env, asset_cfg)
    obj_pos_w = env.get_entity_data(object_name).root_link_pos_w

    reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
    reaching = torch.exp(-reach_error / reaching_std**2)

    position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
    bringing = torch.exp(-position_error / bringing_std**2)

    return reaching * (1.0 + bringing)


def bring_object_reward(
    env: World,
    command_name: str,
    object_name: str,
    std: float,
) -> torch.Tensor:
    """How near the object is to its goal, as a Gaussian kernel."""
    command = env.command_manager.get_term(command_name)
    obj_pos_w = env.get_entity_data(object_name).root_link_pos_w
    position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
    return torch.exp(-position_error / std**2)


def joint_velocity_hinge_penalty(
    env: World,
    max_vel: float,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Squared penalty on joint speed ABOVE a limit, zero below it.

    A plain velocity penalty taxes every motion, including the fast
    approach the task wants. This one is silent until a joint exceeds
    ``max_vel`` and then grows quadratically, which bounds the speed
    without discouraging moving at all.
    """
    joint_vel = env.get_entity_data(asset_cfg.name).joint_vel
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)


def object_dropped(
    env: World,
    object_name: str,
    min_height: float,
) -> torch.Tensor:
    """1.0 where the object has fallen below ``min_height``.

    Not in mjlab's set. Added because this scene has a table: an object
    knocked off it lands on the floor and stays there, and without a term
    that notices, the episode runs to its time-out collecting a reaching
    reward for hovering over an empty table.
    """
    return (env.get_entity_data(object_name).root_link_pos_w[:, 2] < min_height).float()
