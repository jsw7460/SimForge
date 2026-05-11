"""Unified reward terms using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(asset_cfg.name)``, making them simulator-agnostic.

Selector convention: every term takes
:class:`~rlworld.rl.configs.scene.entity_selector.ResolvedEntity` as
``asset_cfg``.  The default is :data:`_DEFAULT_SELECTOR` which points at
the ``"robot"`` entity with no subset filter — RewardManager auto-resolves
the default at setup time, so presets only need to specify ``asset_cfg``
when they want a non-default selector (e.g. specific joints/bodies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


# ── Quadruped leg geometry helpers ───────────────────────────────────────

# Nominal x/y sign for each leg in body frame (x=forward, y=left).
_LEG_NOMINAL_SIGNS = {
    "FL": (+1.0, +1.0),  # Front-Left:  +x, +y (URDF: left = +y)
    "FR": (+1.0, -1.0),  # Front-Right: +x, -y (URDF: right = -y)
    "RL": (-1.0, +1.0),  # Rear-Left:   -x, +y
    "RR": (-1.0, -1.0),  # Rear-Right:  -x, -y
}


def get_leg_xy_signs(foot_names: tuple[str, ...] | list[str]) -> list[tuple[float, float]]:
    """Return (x_sign, y_sign) for each foot, matching foot_names order.

    Parses FL/FR/RL/RR substring from each name.
    """
    signs = []
    for name in foot_names:
        matched = [key for key in _LEG_NOMINAL_SIGNS if key in name]
        if len(matched) != 1:
            raise ValueError(
                f"Cannot identify leg from foot name '{name}'. "
                f"Expected exactly one of {list(_LEG_NOMINAL_SIGNS)} as substring."
            )
        signs.append(_LEG_NOMINAL_SIGNS[matched[0]])
    return signs


def track_lin_vel(
    env: World,
    std: float = 0.25,
    penalize_z: bool = False,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Reward for tracking commanded linear velocity in xy plane."""
    target = torch.stack([env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1)
    actual = env.get_robot_data(asset_cfg.name).root_link_lin_vel_b
    xy_error = torch.sum(torch.square(target - actual[:, :2]), dim=1)
    if penalize_z:
        xy_error = xy_error + torch.square(actual[:, 2])
    return torch.exp(-xy_error / std**2)


def track_ang_vel(
    env: World,
    std: float = 0.25,
    penalize_xy: bool = False,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw)."""
    actual = env.get_robot_data(asset_cfg.name).root_link_ang_vel_b
    z_error = torch.square(env.command_manager.ang_vel - actual[:, 2])
    if penalize_xy:
        z_error = z_error + torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-z_error / std**2)


def action_rate_l2(env: World) -> torch.Tensor:
    """Penalty for sudden action changes (L2 squared).

    Returns:
        Tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.square(env.act_manager.prev_processed_actions - env.act_manager.processed_actions),
        dim=1,
    )


def flat_orientation(
    env: World,
    std: float | None = None,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Penalty for non-flat orientation (roll/pitch deviation from upright)."""
    gravity_b = env.get_robot_data(asset_cfg.name).projected_gravity_b
    xy_squared = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    if std is not None:
        return torch.exp(-xy_squared / (std**2))
    return -xy_squared


# ── Walk-These-Ways reward terms ─────────────────────────────────────────


def penalize_lin_vel_z(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalize z-axis base linear velocity. WTW: _reward_lin_vel_z."""
    vel_z = env.get_robot_data(asset_cfg.name).root_link_lin_vel_b[:, 2]
    return -torch.square(vel_z)


def penalize_ang_vel_xy(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalize xy-axis base angular velocity. WTW: _reward_ang_vel_xy."""
    ang_vel_xy = env.get_robot_data(asset_cfg.name).root_link_ang_vel_b[:, :2]
    return -torch.sum(torch.square(ang_vel_xy), dim=1)


def penalize_dof_vel(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalize joint velocities. WTW: _reward_dof_vel."""
    return -torch.sum(torch.square(env.get_robot_data(asset_cfg.name).joint_vel), dim=1)


def similar_to_default(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalty for deviating from default joint positions."""
    return -torch.sum(
        torch.abs(env.get_robot_data(asset_cfg.name).joint_pos - env.act_manager.offset),
        dim=1,
    )


def reward_alive(env: World) -> torch.Tensor:
    """Constant alive reward (1.0 per env).

    Returns:
        Tensor of shape (num_envs,) on the default device. Matches the
        original sim-specific implementations exactly: ``torch.ones((num_envs,))``.
    """
    return torch.ones((env.num_envs,))


def base_height_penalty(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalty for deviating from target base height."""
    height_z = env.get_robot_data(asset_cfg.name).root_link_pos_w[:, 2]
    return -torch.square(height_z - env.command_manager.base_height)


def penalize_angular_momentum_l2(
    env: World,
    sensor_name: str | None = None,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Penalize whole-body angular momentum (L2 squared, sim-agnostic).

    Reads ``RobotData.angular_momentum_w(sensor_name)`` and returns
    ``-sum(square(...))``. Each simulator implements
    ``angular_momentum_w`` differently:

    - Newton: manual ``sum_i I_i @ omega_i`` over all bodies
      (``sensor_name`` ignored).
    - mjlab: reads MuJoCo's built-in ``subtreeangmom`` sensor data,
      identified by ``sensor_name``.
    - Genesis: not implemented (raises NotImplementedError).

    The two implementations are NOT bit-identical to each other —
    they compute physically related but mathematically distinct
    quantities (manual sum-of-body-momenta vs subtree angular momentum
    via the composite mass matrix). Each was already in use by its own
    sim's preset before this migration; the unification preserves
    those values exactly within each sim.

    Args:
        env: Any environment whose RobotData implements
            ``angular_momentum_w``.
        sensor_name: Sensor identifier for sims that need one (mjlab).
            Required for mjlab; ignored by Newton.
        asset_cfg: Selector identifying the robot entity.

    Returns:
        Tensor of shape ``(num_envs,)``.
    """
    angmom = env.get_robot_data(asset_cfg.name).angular_momentum_w(sensor_name=sensor_name)
    return -torch.sum(torch.square(angmom), dim=-1)


def raw_action_rate_l2(env: World) -> torch.Tensor:
    """Penalize the rate of change of raw actions (L2 squared, sim-agnostic).

    Reads ``env.act_manager.raw_actions`` and ``prev_raw_actions``, both
    pure act_manager state with no scene-state dependency. The result is
    bit-identical across all simulators because it touches no
    physics/RobotData state.

    Returns:
        Tensor of shape ``(num_envs,)`` — negative L2-squared difference.
    """
    return -torch.sum(
        torch.square(env.act_manager.raw_actions - env.act_manager.prev_raw_actions),
        dim=1,
    )


def penalize_body_ang_vel_xy(
    env: World,
    body_name: str,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity of a single body (sim-agnostic).

    Reads world-frame angular velocity for the named body via
    ``RobotData.find_body_index`` and ``RobotData.body_ang_vel_w``, then
    returns ``-sum(square(ang_vel[:, :2]))``. The yaw component (index 2)
    is intentionally NOT penalized — only roll/pitch are.

    Matches the behavior of mjlab's ``body_angular_velocity_penalty``
    exactly, which the legacy sim-specific implementations also matched.

    Args:
        env: Any environment whose RobotData implements the body-level
            accessors (Newton, Genesis, MuJoCo).
        body_name: Name of the body. Format depends on the simulator's
            naming convention (Newton uses prefixed names like
            ``"g1_29dof/torso_link"``, Genesis and mjlab use bare names
            like ``"torso_link"``).
        asset_cfg: Selector identifying the robot entity.

    Returns:
        Tensor of shape ``(num_envs,)``.
    """
    rd = env.get_robot_data(asset_cfg.name)
    body_idx = rd.find_body_index(body_name)
    ang_vel = rd.body_ang_vel_w(body_idx)
    return -torch.sum(torch.square(ang_vel[:, :2]), dim=1)


# ── Feet rewards (mjlab-style) ───────────────────────────────────────────
#
# NOTE (PR3a+b scope): these functions still take ``body_names`` /
# ``site_names`` / ``entity_name`` because the underlying RobotData
# accessors (``body_pos_w(names)``, ``site_pos_w(names)``) require
# *name* lists, not the *id* tensors that ``ResolvedEntity`` carries.
# Migrating them to the selector pattern requires either extending
# ResolvedEntity with resolved name lists or teaching RobotData to
# accept ids — both are out of scope for this PR.  Tracked as a
# follow-up under the reward-side selector adoption work.


def _command_active(env: World, command_threshold: float) -> torch.Tensor:
    """Return a (num_envs,) float mask for command magnitude > threshold."""
    cmd = torch.stack(
        [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y, env.command_manager.ang_vel],
        dim=1,
    )
    linear_norm = torch.norm(cmd[:, :2], dim=1)
    angular_norm = torch.abs(cmd[:, 2])
    total = linear_norm + angular_norm
    return (total > command_threshold).float()


def _foot_pos_vel(
    env: World,
    body_names: list[str] | None,
    site_names: list[str] | None,
    entity_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (foot_pos_w, foot_lin_vel_w) for either body or site names."""
    if (body_names is None) == (site_names is None):
        raise ValueError(
            f"Pass exactly one of body_names or site_names (got body_names={body_names!r}, site_names={site_names!r})"
        )
    rd = env.get_robot_data(entity_name)
    if body_names is not None:
        return rd.body_pos_w(body_names), rd.body_lin_vel_w(body_names)
    return rd.site_pos_w(site_names), rd.site_lin_vel_w(site_names)


def penalize_feet_clearance(
    env: World,
    target_height: float,
    command_threshold: float = 0.01,
    body_names: list[str] | None = None,
    site_names: list[str] | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize deviation from target foot clearance, weighted by foot xy speed.

    Matches mjlab's ``feet_clearance`` reward exactly. Active only when
    the commanded velocity magnitude exceeds ``command_threshold``.

    Args:
        env: Any environment with a RobotData implementation.
        target_height: Target foot clearance during swing.
        command_threshold: Minimum command magnitude to activate.
        body_names: Foot body / link names (Newton, Genesis).
        site_names: Foot site names (MuJoCo).
        entity_name: Entity to query.
    """
    foot_pos, foot_vel = _foot_pos_vel(env, body_names, site_names, entity_name)
    foot_z = foot_pos[..., 2]
    vel_norm = torch.norm(foot_vel[..., :2], dim=-1)
    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)
    return -cost * _command_active(env, command_threshold)


def penalize_feet_slip(
    env: World,
    contact_group: str,
    command_threshold: float = 0.05,
    body_names: list[str] | None = None,
    site_names: list[str] | None = None,
    contact_order: list[str] | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize foot xy speed while in contact with the ground.

    Matches mjlab's ``feet_slip`` reward exactly. The contact tensor and
    foot velocity tensor are aligned column-wise via ``contact_order``;
    if ``contact_order`` is not given it defaults to ``body_names``
    (Newton/Genesis case). For sites the caller must ensure the contact
    group's natural order matches ``site_names``.

    Args:
        env: Any environment with a RobotData + contact_manager.
        contact_group: Name of the registered contact group.
        command_threshold: Minimum command magnitude to activate.
        body_names: Foot body / link names (Newton, Genesis).
        site_names: Foot site names (MuJoCo).
        contact_order: Optional explicit contact reorder list.
        entity_name: Entity to query.
    """
    _, foot_vel = _foot_pos_vel(env, body_names, site_names, entity_name)
    vel_xy_norm_sq = torch.sum(torch.square(foot_vel[..., :2]), dim=-1)

    if contact_order is None and body_names is not None:
        contact_order = list(body_names)
    is_contact = env.contact_manager.is_contact(contact_group, order=contact_order)

    cost = torch.sum(vel_xy_norm_sq * is_contact.float(), dim=1)
    return -cost * _command_active(env, command_threshold)


def penalize_contact_force_count(
    env: World,
    contact_group: str,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Count tracked bodies whose contact-force magnitude exceeds a threshold.

    Returns the negated count (penalty). When the backend supports
    substep history (mjlab when ``history_length > 0``), the threshold
    check runs across all substeps and a body counts as a hit if **any**
    substep crossed the threshold. Otherwise we fall back to the
    instantaneous ``contact_force`` array.

    This unifies three legacy functions across simulators:

    - mjlab ``self_collision_cost`` (history-aware)
    - mjlab ``wtw_collision`` (history-aware)
    - Newton/Genesis ``wtw_collision`` (instantaneous only — those
      backends always return ``None`` from ``contact_force_history``)

    The math is identical: ``-sum((force_mag > threshold).float())`` over
    the N tracked bodies of the contact group.

    Args:
        env: Any environment with a contact_manager.
        contact_group: Name of the registered contact group.
        force_threshold: Force magnitude (Newtons) above which a body
            counts as one hit.

    Returns:
        Tensor of shape ``(num_envs,)`` — negative hit count.
    """
    history = env.contact_manager.contact_force_history(contact_group)
    if history is not None:
        # (B, N, H, 3) → (B, N, H) → (B, N) via any-substep-over-threshold
        force_mag = torch.norm(history, dim=-1)
        hit = (force_mag > force_threshold).any(dim=2)
        return -hit.float().sum(dim=-1)

    forces = env.contact_manager.contact_force(contact_group)
    force_mag = torch.norm(forces, dim=-1)  # (B, N)
    return -(force_mag > force_threshold).float().sum(dim=-1)


def penalize_soft_landing(
    env: World,
    contact_group: str,
    command_threshold: float = 0.05,
    contact_order: list[str] | None = None,
) -> torch.Tensor:
    """Penalize impact force at first foot contact (sum over feet).

    Matches mjlab's ``soft_landing`` reward exactly. Because the cost is
    summed over feet, the order does not affect the result; the
    ``contact_order`` parameter is preserved for symmetry with the other
    feet rewards but is rarely needed.
    """
    forces = env.contact_manager.contact_force(contact_group, order=contact_order)
    fmag = torch.norm(forces, dim=-1)
    first = env.contact_manager.compute_first_contact(contact_group, order=contact_order)
    cost = torch.sum(fmag * first.float(), dim=1)
    return -cost * _command_active(env, command_threshold)


class FeetSwingHeightTracker:
    """Stateful penalty: tracks per-foot peak height during swing, evaluates at landing.

    Matches mjlab's ``feet_swing_height`` reward exactly. The error is
    ``peak_h / target_h - 1``; ``use_squared_error`` toggles between
    squared (mjlab default) and absolute (used by an older Walk-These-
    Ways variant). Per-env peak heights are reset on landing and on
    explicit episode reset.

    Args:
        env: Any environment with a RobotData + contact_manager.
        contact_group: Name of the registered contact group.
        target_height: Target swing peak height.
        command_threshold: Minimum command magnitude to activate.
        body_names: Foot body / link names (Newton, Genesis).
        site_names: Foot site names (MuJoCo).
        contact_order: Optional explicit contact reorder list. Defaults
            to ``body_names`` when bodies are used.
        entity_name: Entity to query.
        use_squared_error: ``True`` for ``error**2`` (mjlab); ``False``
            for ``abs(error)`` (older WTW variant).
        reset_mode: Per-env reset behavior. ``"zero"`` (Genesis legacy),
            ``"current_foot_height"`` (Newton legacy), or ``"none"``
            (MuJoCo legacy — peaks persist across episode resets).
    """

    __name__ = "FeetSwingHeightTracker"

    def __init__(
        self,
        env: World,
        contact_group: str,
        target_height: float,
        command_threshold: float = 0.05,
        body_names: list[str] | None = None,
        site_names: list[str] | None = None,
        contact_order: list[str] | None = None,
        entity_name: str = "robot",
        use_squared_error: bool = True,
        reset_mode: str = "zero",
    ) -> None:
        if (body_names is None) == (site_names is None):
            raise ValueError(
                f"Pass exactly one of body_names or site_names (got body_names={body_names!r}, site_names={site_names!r})"
            )
        self._env = env
        self._contact_group = contact_group
        self._target_height = target_height
        self._command_threshold = command_threshold
        self._body_names = list(body_names) if body_names is not None else None
        self._site_names = list(site_names) if site_names is not None else None
        self._entity_name = entity_name
        self._use_squared_error = use_squared_error
        if reset_mode not in ("zero", "current_foot_height", "none"):
            raise ValueError(f"reset_mode must be one of 'zero', 'current_foot_height', 'none' (got {reset_mode!r})")
        self._reset_mode = reset_mode

        if contact_order is None and self._body_names is not None:
            contact_order = list(self._body_names)
        self._contact_order = list(contact_order) if contact_order is not None else None

        num_feet = len(self._body_names if self._body_names is not None else self._site_names)
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device, dtype=torch.float32)

    def _foot_heights(self, env: World) -> torch.Tensor:
        rd = env.get_robot_data(self._entity_name)
        if self._body_names is not None:
            return rd.body_pos_w(self._body_names)[..., 2]
        return rd.site_pos_w(self._site_names)[..., 2]

    def __call__(self, env: World) -> torch.Tensor:
        foot_heights = self._foot_heights(env)
        is_contact = env.contact_manager.is_contact(self._contact_group, order=self._contact_order)
        in_air = ~is_contact

        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = env.contact_manager.compute_first_contact(self._contact_group, order=self._contact_order)

        active = _command_active(env, self._command_threshold)
        error = self.peak_heights / self._target_height - 1.0
        if self._use_squared_error:
            err_term = torch.square(error)
        else:
            err_term = torch.abs(error)
        cost = torch.sum(err_term * first_contact.float(), dim=1) * active

        # Reset peaks for feet that just landed.
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return -cost

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._reset_mode == "none" or self.peak_heights is None:
            return
        if self._reset_mode == "zero":
            self.peak_heights[env_ids] = 0.0
            return
        # "current_foot_height": Newton legacy — re-seed peak with current z.
        foot_heights = self._foot_heights(self._env)
        self.peak_heights[env_ids] = foot_heights[env_ids]


class VariablePostureTracker:
    """Stateful penalty: speed-dependent posture tracking with per-joint std.

    Matches mjlab's ``variable_posture`` exactly. The reward is

        exp(-mean((q - q_default)² / std²))

    where ``std`` switches between three per-joint vectors based on the
    current commanded velocity magnitude:

      - ``std_standing`` when ``total_speed < walking_threshold``
      - ``std_walking`` when ``walking_threshold <= total_speed < running_threshold``
      - ``std_running`` when ``total_speed >= running_threshold``

    The three legacy implementations (Newton, Genesis, MuJoCo) all share
    this math but resolve joint state differently — Newton/Genesis use
    ``env.act_manager.actuated_joint_names`` plus ``env.robot_data.joint_pos``
    and a 1-D ``act_manager.offset``, while MuJoCo uses
    ``robot.find_joints(asset_cfg.joint_names)`` plus a sliced
    ``robot.data.joint_pos[:, joint_ids]`` and a 2-D
    ``robot.data.default_joint_pos[:, joint_ids]``. To accommodate both
    without leaking sim-specific code into common, the caller passes the
    pre-resolved joint name list, a callable that returns the current
    joint position tensor on each step, and the default joint position
    tensor (1-D or 2-D, both broadcast correctly).

    Args:
        env: Any environment with a ``command_manager`` exposing
            ``lin_vel_x``, ``lin_vel_y``, ``ang_vel`` columns.
        joint_names: Resolved per-joint name list, in the same order as
            the tensors returned by ``get_current_joint_pos``. Used to
            expand the std-dict regex patterns.
        std_standing: Mapping of joint-name regex → std value (standing
            regime).
        std_walking: Same, walking regime.
        std_running: Same, running regime.
        get_current_joint_pos: Callable taking ``env`` and returning the
            current joint position tensor of shape ``(num_envs, N)``,
            where ``N == len(joint_names)``.
        default_joint_pos: Default-pose tensor of shape ``(N,)`` or
            ``(num_envs, N)``. Subtracted from current to form the error.
        walking_threshold: Speed below this is "standing".
        running_threshold: Speed at or above this is "running".
    """

    __name__ = "VariablePostureTracker"

    def __init__(
        self,
        env: World,
        joint_names: list[str],
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        get_current_joint_pos,
        default_joint_pos: torch.Tensor,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> None:
        self._env = env
        self._walking_threshold = walking_threshold
        self._running_threshold = running_threshold
        self._get_current_joint_pos = get_current_joint_pos
        self._default_joint_pos = default_joint_pos

        names = list(joint_names)
        _, _, std_standing_vals = string_utils.resolve_matching_names_values(std_standing, names)
        _, _, std_walking_vals = string_utils.resolve_matching_names_values(std_walking, names)
        _, _, std_running_vals = string_utils.resolve_matching_names_values(std_running, names)
        self.std_standing = torch.tensor(std_standing_vals, device=env.device, dtype=torch.float32)
        self.std_walking = torch.tensor(std_walking_vals, device=env.device, dtype=torch.float32)
        self.std_running = torch.tensor(std_running_vals, device=env.device, dtype=torch.float32)

    def __call__(self, env: World) -> torch.Tensor:
        cmd = torch.stack(
            [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y, env.command_manager.ang_vel],
            dim=1,
        )
        linear_speed = torch.norm(cmd[:, :2], dim=1)
        angular_speed = torch.abs(cmd[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self._walking_threshold).float()
        walking_mask = ((total_speed >= self._walking_threshold) & (total_speed < self._running_threshold)).float()
        running_mask = (total_speed >= self._running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current = self._get_current_joint_pos(env)
        error_squared = torch.square(current - self._default_joint_pos)
        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


# ── Getup rewards moved to rewards/common/getup.py ──────────────
# (orientation_upright, height_to_target, GatedPostureTracker,
# GetupSuccessTracker). Import from ``rewards.common.getup``.


def penalize_joint_pos_limits_l1(
    env: World,
    soft_limit_factor: float = 1.0,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits (L1, sim-agnostic).

    Matches the math of mjlab's ``joint_pos_limits`` exactly:

        out = max(lower - q, 0) + max(q - upper, 0)
        return -sum(out, dim=-1)

    Where ``lower``, ``upper`` are the *soft* limits, computed as
    ``hard_lower * soft_limit_factor`` and ``hard_upper * soft_limit_factor``.

    Reads ``RobotData.joint_pos`` and ``RobotData.joint_pos_limits``, both
    in canonical actuated joint order.

    Args:
        env: Any environment whose ``RobotData`` implements
            ``joint_pos_limits`` (Newton, Genesis). Note: not callable on
            MuJoCo, which uses its own ``joint_pos_limits`` reward function
            in ``mdp/rewards/mujoco/reward_terms.py``.
        soft_limit_factor: Multiplicative factor on the hard limits.
            ``1.0`` (the active default in current presets) means
            penalize whenever the joint exceeds its hard limit.
        asset_cfg: Selector identifying the robot entity.

    Returns:
        Tensor of shape ``(num_envs,)`` — negative sum of soft-limit
        violations across joints.
    """
    rd = env.get_robot_data(asset_cfg.name)
    dof_pos = rd.joint_pos
    lower, upper = rd.joint_pos_limits
    lower = lower * soft_limit_factor
    upper = upper * soft_limit_factor
    out_of_limits = -(dof_pos - lower).clamp(max=0.0)
    out_of_limits += (dof_pos - upper).clamp(min=0.0)
    return -torch.sum(out_of_limits, dim=-1)
