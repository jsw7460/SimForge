"""Reward terms of the Booster K1 velocity recipe that the shared library lacks.

Everything else the recipe uses already exists on all three backends; these five
are the terms whose formula has no equivalent:

- :func:`track_lin_vel_relative_std` — velocity tracking whose tolerance grows
  with the commanded speed, blended with a linear progress term.
- :func:`variable_upright` — upright tracking whose tilt tolerance widens across
  three speed regimes.
- :func:`standing_pose_deviation_l1` — L1 pose deviation that fades in as the
  commanded speed falls to zero.
- :class:`upper_body_posture_penalty` — per-joint quadratic pose error with a
  three-regime per-joint std.
- :func:`self_collision_substep_count` — self-collision priced by how many
  physics substeps it lasted, not by whether it happened.

All five read state only through ``env.get_entity_data``, ``env.contact_manager`` and
``env.command_manager``, so one implementation serves MuJoCo, Newton and
Genesis.

SIGN CONVENTION: penalties return a NEGATIVE value and carry a POSITIVE weight,
which is this repository's convention and the opposite of the source recipe's
(positive penalty, negative weight). The product is identical; only the two
factors swap sign. Weights in the preset are therefore the absolute values of
the source recipe's.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict

import torch

from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def _commanded_speed(env: World) -> torch.Tensor:
    """``|v_xy| + |w_z|`` of the current velocity command, shape ``(num_envs,)``.

    The source recipe sums a norm and an absolute value rather than taking the
    norm of all three, so the three speed thresholds are calibrated against
    this quantity specifically.
    """
    lin = torch.stack([env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1)
    return torch.norm(lin, dim=1) + torch.abs(env.command_manager.ang_vel)


def _regime_blend(
    speed: torch.Tensor,
    standing: torch.Tensor | float,
    walking: torch.Tensor | float,
    running: torch.Tensor | float,
    walking_threshold: float,
    running_threshold: float,
) -> torch.Tensor:
    """Select one of three per-regime std values by commanded speed.

    Written as a masked sum rather than a branch so it stays a single fused
    expression under the compiled reward chain.
    """
    standing_mask = (speed < walking_threshold).float()
    walking_mask = ((speed >= walking_threshold) & (speed < running_threshold)).float()
    running_mask = (speed >= running_threshold).float()
    if isinstance(standing, torch.Tensor) and standing.dim() == 1:
        # Per-joint std vectors: broadcast the masks over the joint axis.
        standing_mask = standing_mask.unsqueeze(1)
        walking_mask = walking_mask.unsqueeze(1)
        running_mask = running_mask.unsqueeze(1)
    return standing * standing_mask + walking * walking_mask + running * running_mask


def track_lin_vel_relative_std(
    env: World,
    std: float,
    std_at_rest: float,
    relative_std: float = 0.75,
    progress_weight: float = 0.0,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Linear-velocity tracking with a speed-relative tolerance and a progress blend.

    Two departures from plain exponential tracking, both from the source recipe:

    1. The exponential's sigma is ``clamp(relative_std * |cmd_xy|, std_at_rest,
       std)`` instead of a constant. At rest the tolerance is tight, so standing
       still when commanded to move scores far below the peak; at speed it
       relaxes to ``std``, so the term does not demand impossible precision.
    2. With ``progress_weight > 0`` the result mixes in a linear term that is 1
       when the xy error is zero and falls to 0 when the error reaches the
       commanded speed. The exponential alone is nearly flat far from the
       target, which leaves a standing policy without a gradient toward moving;
       the linear term restores one.

    The commanded z velocity is taken to be zero and its error enters the
    exponential, but the progress term scores the xy error alone.

    Args:
        std: Upper bound on the tolerance, reached at ``|cmd_xy| >= std /
            relative_std``.
        std_at_rest: Lower bound on the tolerance, used for a zero command.
        relative_std: Tolerance per unit of commanded speed.
        progress_weight: Mixing weight of the linear progress term in ``[0, 1)``.
            Zero reduces this to plain exponential tracking.

    Returns:
        Reward in ``(0, 1]``, shape ``(num_envs,)``.
    """
    target = torch.stack([env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1)
    actual = env.get_entity_data(asset_cfg.name).root_link_lin_vel_b

    xy_error_sq = torch.sum(torch.square(target - actual[:, :2]), dim=1)
    z_error_sq = torch.square(actual[:, 2])

    cmd_norm = torch.norm(target, dim=1)
    sigma = torch.clamp(relative_std * cmd_norm, min=std_at_rest, max=std)
    tracking = torch.exp(-(xy_error_sq + z_error_sq) / torch.square(sigma))
    if progress_weight <= 0.0:
        return tracking

    xy_error = torch.sqrt(xy_error_sq)
    progress = torch.clamp(1.0 - xy_error / torch.clamp(cmd_norm, min=std_at_rest), 0.0, 1.0)
    return (1.0 - progress_weight) * tracking + progress_weight * progress


def variable_upright(
    env: World,
    std_standing: float,
    std_walking: float | None = None,
    std_running: float | None = None,
    walking_threshold: float = 0.05,
    running_threshold: float = 1.5,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Upright tracking whose tilt tolerance widens with the commanded speed.

    The tilt measure is the squared xy magnitude of gravity in the root link's
    frame, zero when perfectly upright. Only the tolerance changes between
    regimes: standing demands the most level trunk, running the least, because
    a running gait leans.

    This robot's root link is its trunk, so the root-frame gravity the shared
    :class:`RobotData` exposes is the trunk-frame gravity the source recipe
    computes from the trunk's world quaternion.

    Returns:
        Reward in ``(0, 1]``, shape ``(num_envs,)``.
    """
    if std_walking is None:
        std_walking = std_standing
    if std_running is None:
        std_running = std_walking

    speed = _commanded_speed(env)
    std = _regime_blend(speed, std_standing, std_walking, std_running, walking_threshold, running_threshold)

    gravity_b = env.get_entity_data(asset_cfg.name).projected_gravity_b
    xy_squared = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    return torch.exp(-xy_squared / torch.square(std))


def standing_pose_deviation_l1(
    env: World,
    command_threshold: float = 0.05,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Mean L1 joint deviation from the default pose, faded in as the command stops.

    The gate is a ramp, not a switch: the penalty scales by
    ``1 - clamp(speed / command_threshold, max=1)``, so it reaches full strength
    only at a dead-zero command and vanishes once the commanded speed reaches the
    threshold. A walking policy is therefore never pulled toward the home pose,
    while a standing one is held there.

    Deviation is measured against ``default_joint_pos`` (the nominal standing
    pose) rather than the action-space offset; the two coincide for this recipe
    but differ under joint-limit action mapping.

    Returns:
        Penalty in ``(-inf, 0]``, shape ``(num_envs,)``.
    """
    rd = env.get_entity_data(asset_cfg.name)
    ids = asset_cfg.joint_ids
    if ids is None:
        current, default = rd.joint_pos, rd.default_joint_pos
    else:
        current, default = rd.joint_pos[:, ids], rd.default_joint_pos[..., ids]

    deviation = torch.mean(torch.abs(current - default), dim=1)
    standing_scale = 1.0 - torch.clamp(_commanded_speed(env) / command_threshold, max=1.0)
    return -(deviation * standing_scale)


def self_collision_substep_count(
    env: World,
    contact_group: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Number of physics substeps in which the robot was touching itself.

    The shared library offers two self-collision penalties and neither is this
    one. One counts how many BODIES are in contact; the other returns a plain
    0 or 1. The source recipe's term instead reduces over bodies with ``any``
    and then SUMS over the substeps of the control step, so a brush that lasts
    one substep costs a quarter of a contact that persists through all four.
    With a decimation of 4 the two shared variants are off by up to 4x.

    Without substep history the term falls back to the instantaneous contact,
    which caps the count at 1 and makes a persistent contact cost the same as a
    momentary one. Whether a backend provides the history is visible in the
    cross-sim diagnostic, where this term is one of the compared values.

    Returns:
        Penalty in ``[-num_substeps, 0]``, shape ``(num_envs,)``.
    """
    history = env.contact_manager.contact_force_history(contact_group)
    if history is not None:
        # (B, N, H, 3) -> any over the N tracked columns -> sum over substeps.
        force_mag = torch.norm(history, dim=-1)
        return -(force_mag > force_threshold).any(dim=1).sum(dim=-1).float()

    forces = env.contact_manager.contact_force(contact_group)
    force_mag = torch.norm(forces, dim=-1)
    return -(force_mag > force_threshold).any(dim=-1).float()


class upper_body_posture_penalty:
    """Quadratic pose error over a joint subset, with a per-joint three-regime std.

    The shared ``variable_posture`` term computes ``exp(-mean(err^2 / std^2))``,
    a bounded reward. This one is the un-exponentiated cost the source recipe
    uses for the arms and head, which grows without bound as the posture drifts
    and so keeps pulling at large deviations where the exponential has
    saturated. The two are not interchangeable.

    Each std argument maps a joint-name regex to a value and must resolve every
    selected joint exactly once; an unmatched or doubly-matched joint raises at
    construction rather than silently taking a neighbour's tolerance.

    Args:
        asset_cfg: Selector for the scored joints, already resolved.
        std_standing: Per-joint regex -> std, standing regime.
        std_walking: Same, walking regime.
        std_running: Same, running regime.
        walking_threshold: Commanded speed at or above which "walking" starts.
        running_threshold: Commanded speed at or above which "running" starts.

    Returns:
        Penalty in ``(-inf, 0]``, shape ``(num_envs,)``.
    """

    __name__ = "upper_body_posture_penalty"

    def __init__(
        self,
        env: World,
        asset_cfg: ResolvedEntity,
        std_standing: Dict[str, float],
        std_walking: Dict[str, float],
        std_running: Dict[str, float],
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> None:
        if asset_cfg.joint_ids is None:
            raise ValueError(
                "upper_body_posture_penalty needs an explicit joint subset; " "pass joint_names on the selector."
            )
        self._name = asset_cfg.name
        self._ids = asset_cfg.joint_ids
        self._walking_threshold = walking_threshold
        self._running_threshold = running_threshold

        names = [n.rsplit("/", 1)[-1] for n in asset_cfg.joint_names]
        if len(names) != len(self._ids):
            raise ValueError(f"selector resolved {len(self._ids)} joint ids but {len(names)} names")
        self._std_standing = self._resolve(std_standing, names, "std_standing", env.device)
        self._std_walking = self._resolve(std_walking, names, "std_walking", env.device)
        self._std_running = self._resolve(std_running, names, "std_running", env.device)

    @staticmethod
    def _resolve(table: Dict[str, float], names: list[str], label: str, device) -> torch.Tensor:
        values = []
        for name in names:
            hits = {v for pattern, v in table.items() if re.fullmatch(pattern, name)}
            if len(hits) != 1:
                raise ValueError(f"{label} for joint {name!r} resolves to {sorted(hits)!r} (need exactly one)")
            values.append(hits.pop())
        return torch.tensor(values, device=device, dtype=torch.float32)

    def __call__(self, env: World) -> torch.Tensor:
        rd = env.get_entity_data(self._name)
        current = rd.joint_pos[:, self._ids]
        default = rd.default_joint_pos[..., self._ids]

        std = _regime_blend(
            _commanded_speed(env),
            self._std_standing,
            self._std_walking,
            self._std_running,
            self._walking_threshold,
            self._running_threshold,
        )
        error_squared = torch.square(current - default)
        return -torch.mean(error_squared / torch.square(std), dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        pass
