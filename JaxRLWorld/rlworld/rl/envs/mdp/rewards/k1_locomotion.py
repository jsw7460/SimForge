"""Reward terms for the K1 joystick task, formula-exact to upstream.

Every function transcribes the corresponding upstream mujoco_playground
K1 reward verbatim (including its quirks — see ``feet_slip_base_vel``),
reading state through the sim-agnostic RobotData / manager interfaces.
Upstream's ``filtered_linvel``/``filtered_angvel`` use filter
coefficients (1.0, 0.0), i.e. no filtering, so the tracking terms read
the raw body-frame velocities directly.

Default joint positions come from ``env.act_manager.offset`` — the
cross-sim source equal to upstream's ``_default_pose``.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


# ── Tracking (sigma is the DIVISOR of the squared error, as upstream) ─


def track_lin_vel_xy_exp(
    env: World,
    tracking_sigma: float = 0.25,
    command_term: str = "velocity",
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """``exp(-sum((cmd_xy - v_xy)^2) / sigma)`` with body-frame linvel."""
    cmd = env.command_manager.get_term(command_term).command
    v = env.get_robot_data(asset_cfg.name).root_link_lin_vel_b
    err = torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1)
    return torch.exp(-err / tracking_sigma)


def track_ang_vel_z_exp(
    env: World,
    tracking_sigma: float = 0.25,
    command_term: str = "velocity",
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """``exp(-(cmd_yaw - gyro_z)^2 / sigma)``."""
    cmd = env.command_manager.get_term(command_term).command
    w = env.get_robot_data(asset_cfg.name).root_link_ang_vel_b
    err = torch.square(cmd[:, 2] - w[:, 2])
    return torch.exp(-err / tracking_sigma)


# ── Base costs ────────────────────────────────────────────────────────


def ang_vel_xy_l2(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """``sum(gyro_xy^2)``."""
    w = env.get_robot_data(asset_cfg.name).root_link_ang_vel_b
    return torch.sum(torch.square(w[:, :2]), dim=1)


def orientation_l2(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """``sum(upvector_xy^2)``; equal to projected-gravity xy squared."""
    g = env.get_robot_data(asset_cfg.name).projected_gravity_b
    return torch.sum(torch.square(g[:, :2]), dim=1)


# ── Feet ──────────────────────────────────────────────────────────────


class K1FeetAirTime:
    """Stateful air-time reward with upstream's control-rate bookkeeping.

    Keeps per-foot ``air_time`` / ``last_contact`` buffers updated once
    per control step (NOT from the contact manager's physics-rate
    buffers) so the values and the OR-filtered first-contact detection
    match upstream exactly:

        contact_filt  = contact | last_contact
        first_contact = (air_time > 0) & contact_filt
        air_time     += control_dt
        reward        = sum(clip(air_time - t_min, max=t_max - t_min)
                            * first_contact) * (|cmd| > cmd_threshold)
        air_time     *= ~contact ; last_contact = contact

    The public :attr:`air_time` is the post-increment value, which is
    what upstream feeds to its privileged observation.
    """

    __name__ = "K1FeetAirTime"

    def __init__(
        self,
        env: World,
        contact_group: str,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
        command_term: str = "velocity",
        command_threshold: float = 0.1,
    ) -> None:
        self._contact_group = contact_group
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        self._command_term = command_term
        self._command_threshold = command_threshold
        num_feet = env.contact_manager.is_contact(contact_group).shape[1]
        self.air_time = torch.zeros(env.num_envs, num_feet, device=env.device)
        self._last_contact = torch.zeros(env.num_envs, num_feet, dtype=torch.bool, device=env.device)

    def __call__(self, env: World) -> torch.Tensor:
        contact = env.contact_manager.is_contact(self._contact_group)
        contact_filt = contact | self._last_contact
        first_contact = (self.air_time > 0.0) & contact_filt
        self.air_time += env.control_dt

        clipped = torch.clamp(
            self.air_time - self._threshold_min,
            max=self._threshold_max - self._threshold_min,
        )
        reward = torch.sum(clipped * first_contact, dim=1)
        cmd = env.command_manager.get_term(self._command_term).command
        reward = reward * (cmd.norm(dim=1) > self._command_threshold)

        self.air_time = self.air_time * (~contact)
        self._last_contact = contact
        return reward

    def reset(self, env_ids: torch.Tensor) -> None:
        self.air_time[env_ids] = 0.0
        self._last_contact[env_ids] = False


def feet_slip_base_vel(
    env: World,
    contact_group: str,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Upstream quirk preserved: penalizes the BASE planar speed (not the
    foot speed) once per foot in contact:

        sum_f ||base_linvel_xy_world|| * contact_f
    """
    base_speed = env.get_robot_data(asset_cfg.name).root_link_lin_vel_w[:, :2].norm(dim=1)
    contact = env.contact_manager.is_contact(contact_group)
    return base_speed * contact.float().sum(dim=1)


def _bezier_rz(phase: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Upstream ``gait.get_rz``: cubic-bezier swing profile over phase."""

    def bez(y_start: float, y_end: float, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        curve = x**3 + 3.0 * (x**2 * (1.0 - x))
        return y_start + y_diff * curve

    x = (phase + math.pi) / (2.0 * math.pi)
    stance = bez(0.0, swing_height, 2.0 * x)
    swing = bez(swing_height, 0.0, 2.0 * x - 1.0)
    return torch.where(x <= 0.5, stance, swing)


def feet_phase_bezier(
    env: World,
    command_term: str = "gait_phase",
    swing_height: float = 0.12,
    sigma: float = 0.01,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """``exp(-sum((foot_z - rz(phi))^2) / sigma)``.

    ``asset_cfg.body_names`` must list the feet in the same order as the
    phase columns (left, right). Reads the LIVE phase (pre-advance at
    reward time), matching upstream's within-step pairing. No zero-
    command gate: upstream has it commented out.
    """
    phase = env.command_manager.get_term(command_term).command
    foot_z = env.get_robot_data(asset_cfg.name).body_pos_w_by_ids(asset_cfg.body_ids)[..., 2]
    rz = _bezier_rz(phase, swing_height)
    err = torch.sum(torch.square(foot_z - rz), dim=1)
    return torch.exp(-err / sigma)


def feet_distance_lateral(
    env: World,
    min_distance: float = 0.2,
    cap: float = 0.1,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """``clip(min_distance - lateral_feet_distance, 0, cap)`` in the
    base-yaw frame. ``asset_cfg.body_names`` = (left foot, right foot).
    """
    feet = env.get_robot_data(asset_cfg.name).body_pos_w_by_ids(asset_cfg.body_ids)
    quat = env.get_robot_data(asset_cfg.name).root_link_quat_w  # wxyz
    w, x, y, z = quat.unbind(dim=1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    dy = feet[:, 0, 1] - feet[:, 1, 1]
    dx = feet[:, 0, 0] - feet[:, 1, 0]
    distance = torch.abs(torch.cos(yaw) * dy - torch.sin(yaw) * dx)
    return torch.clamp(min_distance - distance, min=0.0, max=cap)


def contact_pair_penalty(env: World, contact_group: str) -> torch.Tensor:
    """1.0 while any tracked pair of the group is in contact (upstream
    ``collision`` cost: left foot box touching the right foot box)."""
    return env.contact_manager.is_contact(contact_group).any(dim=1).float()


# ── Pose ──────────────────────────────────────────────────────────────


class K1PoseCost:
    """``sum(w * (q - q_default)^2)`` with a fixed per-joint weight table.

    ``weights`` maps joint-name regex (fullmatch on the bare leaf name)
    to a weight; resolved once against the act-manager joint order.
    """

    __name__ = "K1PoseCost"

    def __init__(self, env: World, weights: dict[str, float]) -> None:
        names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
        resolved = []
        for name in names:
            hits = {v for pat, v in weights.items() if re.fullmatch(pat, name)}
            if len(hits) != 1:
                raise ValueError(f"pose weight for joint {name!r} resolves to {hits!r} (need exactly one)")
            resolved.append(hits.pop())
        self._weights = torch.tensor(resolved, device=env.device)

    def __call__(self, env: World) -> torch.Tensor:
        q = env.get_robot_data().joint_pos
        q0 = env.act_manager.offset
        return torch.sum(self._weights * torch.square(q - q0), dim=1)


def joint_deviation_l1(
    env: World,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
    command_term: str = "velocity",
    gate_column: int | None = None,
    gate_threshold: float = 0.1,
) -> torch.Tensor:
    """``sum(|q_sel - q0_sel|)``, optionally gated on ``|cmd[:, col]| >
    threshold`` (upstream hip deviation gates on lateral command)."""
    ids = asset_cfg.joint_ids
    q = env.get_robot_data(asset_cfg.name).joint_pos[:, ids]
    q0 = env.act_manager.offset[:, ids]
    cost = torch.sum(torch.abs(q - q0), dim=1)
    if gate_column is not None:
        cmd = env.command_manager.get_term(command_term).command
        cost = cost * (torch.abs(cmd[:, gate_column]) > gate_threshold)
    return cost


def dof_pos_limits_soft(
    env: World,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Out-of-soft-limit magnitude.

    Reads ``RobotData.soft_joint_pos_limits`` (available on all three
    backends; mjlab exposes ONLY soft limits) — the same
    ``mid ± 0.5 * range * factor`` derivation as upstream, with the
    factor coming from ``ArticulationCfg.soft_joint_pos_limit_factor``
    (the preset pins 0.95, the upstream value)."""
    rd = env.get_robot_data(asset_cfg.name)
    soft_lo, soft_hi = rd.soft_joint_pos_limits
    q = rd.joint_pos
    out = torch.clamp(soft_lo - q, min=0.0) + torch.clamp(q - soft_hi, min=0.0)
    return torch.sum(out, dim=1)
