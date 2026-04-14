"""Getup / fall-recovery reward terms (sim-agnostic).

These terms mirror the fall-recovery reward shaping used by
``mjlab_playground/getup/mdp/rewards.py``. They read state exclusively
through ``env.get_robot_data(entity_name)`` and ``env.act_manager``, so
they work uniformly on Newton, Genesis, and MuJoCo.

Exposed symbols:
  - :func:`orientation_upright`     — full 3D upright orientation reward
  - :func:`height_to_target`         — exponential ramp body-height reward
  - :class:`GatedPostureTracker`    — posture matching gated by upright
  - :class:`GetupSuccessTracker`    — sticky binary success metric
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def orientation_upright(
    env: "World",
    std: float = 0.707,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for full-3D upright orientation (all axes, not just xy).

    Unlike :func:`rewards.common.reward_terms.flat_orientation` which only
    penalizes the xy components of ``projected_gravity_b`` (and therefore
    only distinguishes tilts from upright), this term measures the full
    error between the body-frame projected gravity and the target
    ``[0, 0, -1]`` — so it remains meaningful when the robot is upside-
    down, on its side, or in any arbitrary orientation. That is required
    for the getup task where the robot starts in a random orientation.

    Formula (matches mjlab_playground getup ``orientation_reward``):

        err = sum((UP - projected_gravity_b)^2)      with UP = [0, 0, -1]
        return exp(-err / std^2)

    Args:
        env: Any environment with ``get_robot_data``.
        std: Standard deviation of the exponential kernel. The default of
            ``0.707 ≈ 1/sqrt(2)`` matches the common ``exp(-2 * err)``
            form from mjlab_playground when err is the squared L2
            distance to the upright target.
        entity_name: Name of the entity to query.

    Returns:
        Tensor of shape ``(num_envs,)``, in ``[0, 1]``.
    """
    gravity_b = env.get_robot_data(entity_name).projected_gravity_b
    up = torch.tensor([0.0, 0.0, -1.0], device=gravity_b.device)
    err = torch.sum(torch.square(up - gravity_b), dim=-1)
    return torch.exp(-err / (std ** 2))


def height_to_target(
    env: "World",
    desired_height: float,
    body_name: str | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Exponential ramp reward for raising a body towards a target height.

    Matches mjlab_playground getup ``height_reward``:

        h_clamped = min(h, desired_height)
        return (exp(h_clamped) - 1) / (exp(desired_height) - 1)

    The reward is ``0`` when the body is on the ground and ``1`` at the
    target, with a convex ramp in between — it provides meaningful
    gradient from fallen states while saturating once standing, so it
    does not incentivize overshoot.

    Args:
        env: Any environment with ``get_robot_data``.
        desired_height: Target world-z height (m).
        body_name: Name of the body whose z-height is rewarded. If
            ``None``, the root link's z is used. Pass the trunk body
            name for a humanoid; for T1 the getup task uses both
            ``"Trunk"`` (0.67 m) and ``"Waist"`` (0.55 m) with separate
            term instances to break the "sit-down" local minimum.
        entity_name: Name of the entity to query.

    Returns:
        Tensor of shape ``(num_envs,)``, in ``[0, 1]``.
    """
    rd = env.get_robot_data(entity_name)
    if body_name is None:
        h = rd.root_link_pos_w[:, 2]
    else:
        h = rd.body_pos_w([body_name])[:, 0, 2]
    h_clamped = torch.clamp(h, max=desired_height)
    denom = float(torch.exp(torch.tensor(desired_height)) - 1.0)
    return (torch.exp(h_clamped) - 1.0) / denom


class GatedPostureTracker:
    """Stateful reward: posture tracking gated by upright orientation.

    Matches mjlab_playground getup ``gated_posture_reward``:

        gate  = 1 if orientation_error < gate_threshold else 0
        err   = mean(((q - q_default) / std_per_joint)^2)
        reward = gate * exp(-err)

    The gate ensures the policy is only rewarded for matching the
    default pose *after* it has already stood up — this prevents the
    degenerate "curl into a ball that looks like default pose while
    lying down" strategy. The ``orientation_error`` is the same L2
    distance used by :func:`orientation_upright` (``sum((UP - g_b)^2)``).

    Per-joint std is supplied as a dict of regex → float pairs, resolved
    against ``act_manager.actuated_joint_names`` at construction time
    (same idiom as :class:`VariablePostureTracker`). The default-pose
    tensor comes from ``act_manager.offset`` so Newton/Genesis/MuJoCo
    all share the same source of truth.

    Args:
        env: Any environment with ``get_robot_data`` and ``act_manager``.
        std_dict: Mapping of joint-name regex to std value. Every
            actuated joint must match at least one pattern.
        gate_threshold: Upper bound on ``sum((UP - g_b)^2)`` for the
            gate to open (mjlab default: ``0.01``, very strict).
        entity_name: Entity to query for state.
    """

    __name__ = "GatedPostureTracker"

    def __init__(
        self,
        env: "World",
        std_dict: "dict[str, float]",
        gate_threshold: float = 0.01,
        entity_name: str = "robot",
    ) -> None:
        self._env = env
        self._entity_name = entity_name
        self._gate_threshold = gate_threshold

        joint_names = list(env.act_manager.actuated_joint_names)
        _, _, std_vals = string_utils.resolve_matching_names_values(
            std_dict, joint_names
        )
        self._std = torch.tensor(
            std_vals, device=env.device, dtype=torch.float32
        )

    def __call__(self, env: "World") -> torch.Tensor:
        rd = env.get_robot_data(self._entity_name)
        gravity_b = rd.projected_gravity_b
        up = torch.tensor([0.0, 0.0, -1.0], device=gravity_b.device)
        orient_err = torch.sum(torch.square(up - gravity_b), dim=-1)
        gate = (orient_err < self._gate_threshold).float()

        current = rd.joint_pos
        default = env.act_manager.offset
        err_sq = torch.square((current - default) / self._std)
        return gate * torch.exp(-torch.mean(err_sq, dim=1))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


class GetupSuccessTracker:
    """Stateful sticky success metric for the getup task.

    Matches mjlab_playground getup ``getup_success``: once the robot
    has reached both the upright orientation threshold and the target
    height within tolerance, the per-env flag is latched to ``1.0`` for
    the remainder of the episode. The flag is cleared on episode reset.

    Conditions (both must hold in the same step to latch):

        sum((UP - g_b)^2) < orient_threshold
        |h_target - min(h, h_target)| < height_tolerance

    The "min" in the height check mirrors the height reward's clamp so
    that overshooting the target still counts as success.

    Intended usage: register with ``weight=0.0`` and log as an extra in
    the reward manager, or manually read ``tracker.success`` for wandb
    logging. The ``__call__`` return is always the latched tensor so it
    is safe to use as a regular reward term.

    Args:
        env: Any environment with ``get_robot_data``.
        desired_height: Target body z-height (same value as the height
            reward).
        body_name: Body whose z is measured, or ``None`` for root link.
        orient_threshold: Upper bound on ``sum((UP - g_b)^2)``. mjlab
            default: ``0.05`` (~18° from vertical).
        height_tolerance: Upper bound on ``|h_target - clamp(h)|`` (m).
            mjlab default: ``0.02`` m.
        entity_name: Entity to query for state.
    """

    __name__ = "GetupSuccessTracker"

    def __init__(
        self,
        env: "World",
        desired_height: float,
        body_name: str | None = None,
        orient_threshold: float = 0.05,
        height_tolerance: float = 0.02,
        entity_name: str = "robot",
    ) -> None:
        self._env = env
        self._desired_height = desired_height
        self._body_name = body_name
        self._orient_threshold = orient_threshold
        self._height_tolerance = height_tolerance
        self._entity_name = entity_name
        self.success = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )

    def __call__(self, env: "World") -> torch.Tensor:
        rd = env.get_robot_data(self._entity_name)
        gravity_b = rd.projected_gravity_b
        up = torch.tensor([0.0, 0.0, -1.0], device=gravity_b.device)
        orient_err = torch.sum(torch.square(up - gravity_b), dim=-1)
        orient_ok = orient_err < self._orient_threshold

        if self._body_name is None:
            h = rd.root_link_pos_w[:, 2]
        else:
            h = rd.body_pos_w([self._body_name])[:, 0, 2]
        h_clamped = torch.clamp(h, max=self._desired_height)
        height_ok = (self._desired_height - h_clamped) < self._height_tolerance

        just_succeeded = (orient_ok & height_ok).float()
        self.success = torch.maximum(self.success, just_succeeded)
        return self.success

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.success[env_ids] = 0.0
