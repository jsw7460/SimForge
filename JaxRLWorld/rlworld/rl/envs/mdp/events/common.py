"""Sim-agnostic event / reset terms.

These functions work on any ``World`` subclass by reading state through
``env.get_robot_data()`` and writing through
``env.get_robot_state_writer()``. They replace the per-simulator
``push_robot`` / ``reset_root_state_uniform`` implementations that
lived in ``newton_event_terms.py``, ``event_terms.py`` (Genesis), and
``mujoco.py``.

Quaternion convention
---------------------
All quaternion parameters (``default_quat_wxyz``) and internal
arithmetic use **wxyz**. Newton presets must convert their native xyzw
values before passing.

Subset convention
-----------------
All writer calls use subset-shaped tensors + ``env_ids``, matching the
``RobotStateWriterProtocol`` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_mul_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")

# ── Sampling utility ─────────────────────────────────────────────────


def _sample_uniform(
    lower: torch.Tensor,
    upper: torch.Tensor,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(shape, device=device) + lower


# ── Push ─────────────────────────────────────────────────────────────


def push_by_setting_velocity(
    env: World,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> None:
    """Add a random velocity perturbation to the robot's root link.

    Reads the current world-frame root velocity via ``RobotData``,
    samples a uniform perturbation, and writes the result through the
    ``RobotStateWriter``. No pose change — only velocity.

    Works identically across Newton, Genesis, and MuJoCo.
    """
    if len(env_ids) == 0:
        return

    rd = env.get_robot_data(asset_cfg.name)
    writer = env.get_robot_state_writer(asset_cfg.name)
    device = env.device
    n = len(env_ids)

    lin_vel = rd.root_link_lin_vel_w[env_ids].clone()
    ang_vel = rd.root_link_ang_vel_w[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, key in enumerate(keys):
        lo, hi = velocity_range.get(key, (0.0, 0.0))
        if lo == 0.0 and hi == 0.0:
            continue
        delta = torch.empty(n, device=device).uniform_(lo, hi)
        if i < 3:
            lin_vel[:, i] += delta
        else:
            ang_vel[:, i - 3] += delta

    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)


# ── Reset root state ────────────────────────────────────────────────


def reset_root_state_uniform(
    env: World,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    default_pos: tuple[float, ...] = (0.0, 0.0, 0.34),
    default_quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> None:
    """Reset root pose + velocity with uniform random perturbations.

    Works identically across Newton, Genesis, and MuJoCo. The
    sim-specific ``reset_root_state_uniform`` implementations can now
    be replaced by thin preset entries that pass the right
    ``default_pos`` / ``default_quat_wxyz`` from the robot config.

    **mjlab env_origins**: if ``env.scene_manager`` has a ``.scene``
    with an ``env_origins`` tensor (mjlab's multi-env offset), the
    function adds ``env_origins[env_ids]`` to the position
    automatically. Newton and Genesis ignore this because their scene
    managers have no ``scene.env_origins`` attribute.

    Args:
        env: Any environment satisfying the RobotData + Writer APIs.
        env_ids: Environments to reset.
        pose_range: Per-axis ``(min, max)`` perturbation for position
            (``x/y/z``) and orientation (``roll/pitch/yaw`` in radians).
        velocity_range: Optional per-axis ``(min, max)`` for initial
            root velocity. ``None`` → zero velocity.
        default_pos: Default root position ``(x, y, z)`` before
            perturbation. Comes from robot config.
        default_quat_wxyz: Default root orientation ``(w, x, y, z)``
            before perturbation. **wxyz** convention.
        entity_name: Entity to reset.
    """
    if len(env_ids) == 0:
        return

    writer = env.get_robot_state_writer(asset_cfg.name)
    device = env.device
    n = len(env_ids)

    # ── Sample pose perturbation ──────────────────────────────────
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = _sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)

    # ── Position: default + perturbation ──────────────────────────
    default_pos_t = torch.tensor(default_pos, device=device, dtype=torch.float32)
    pos = default_pos_t.unsqueeze(0).expand(n, -1) + pose_samples[:, 0:3]

    # Auto-detect mjlab env_origins offset
    _scene = getattr(env.scene_manager, "scene", None)
    env_origins = getattr(_scene, "env_origins", None)
    if env_origins is not None:
        pos = pos + env_origins[env_ids]

    # ── Orientation: default quat * delta quat (all wxyz) ─────────
    default_quat_t = torch.tensor(default_quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)

    # Euler perturbation → quaternion delta
    roll, pitch, yaw = pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=device)
    axis_y = torch.tensor([0.0, 1.0, 0.0], device=device)
    axis_z = torch.tensor([0.0, 0.0, 1.0], device=device)
    q_roll = quat_from_angle_axis_wxyz(roll, axis_x)
    q_pitch = quat_from_angle_axis_wxyz(pitch, axis_y)
    q_yaw = quat_from_angle_axis_wxyz(yaw, axis_z)
    delta_quat = quat_mul_wxyz(quat_mul_wxyz(q_yaw, q_pitch), q_roll)

    quat_wxyz = quat_mul_wxyz(default_quat_t, delta_quat)

    # ── Velocity ──────────────────────────────────────────────────
    lin_vel = torch.zeros((n, 3), device=device)
    ang_vel = torch.zeros((n, 3), device=device)
    if velocity_range:
        vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
        vel_ranges = torch.tensor(vel_range_list, device=device)
        vel_samples = _sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device)
        lin_vel = vel_samples[:, 0:3]
        ang_vel = vel_samples[:, 3:6]

    # ── Write ─────────────────────────────────────────────────────
    writer.set_root_pose(pos, quat_wxyz, env_ids=env_ids)
    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)


# ── Reset joint state ──────────────────────────────────────────────


def reset_joints_by_offset(
    env: World,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> None:
    """Reset actuated joint positions/velocities with uniform noise.

    Uses ``act_manager.offset`` as the default joint positions (the
    canonical cross-sim source for default DOF values) and
    ``joint_pos_limits`` from ``RobotData`` for clamping.

    Works identically across Newton, Genesis, and MuJoCo.

    Args:
        env: Any environment satisfying the RobotData + Writer APIs.
        env_ids: Environments to reset.
        position_range: ``(min, max)`` uniform noise added to defaults.
        velocity_range: ``(min, max)`` uniform noise for velocities.
            Defaults to ``(0.0, 0.0)`` (zero velocity).
        entity_name: Entity to reset.
    """
    if len(env_ids) == 0:
        return

    writer = env.get_robot_state_writer(asset_cfg.name)
    device = env.device
    n = len(env_ids)

    # Default joint positions from action manager offset.
    # offset shape: (num_envs, num_actuated) — slice by env_ids.
    default_pos = env.act_manager.offset[env_ids].clone()
    num_joints = default_pos.shape[-1]

    # Ensure default_pos is always 2-D (n, num_joints) even for 1 env.
    if default_pos.dim() == 1:
        default_pos = default_pos.unsqueeze(0)

    # Add position noise
    if position_range != (0.0, 0.0):
        noise = torch.empty(n, num_joints, device=device).uniform_(
            position_range[0],
            position_range[1],
        )
        default_pos = default_pos + noise

    # Joint velocities
    if velocity_range == (0.0, 0.0):
        joint_vel = torch.zeros(n, num_joints, device=device)
    else:
        joint_vel = torch.empty(n, num_joints, device=device).uniform_(
            velocity_range[0],
            velocity_range[1],
        )

    writer.set_dof_positions(default_pos, env_ids=env_ids)
    writer.set_dof_velocities(joint_vel, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)


# ── Encoder bias DR (cross-sim) ─────────────────────────────────────


def randomize_encoder_bias(
    env: World,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float] = (-0.015, 0.015),
) -> None:
    """Sample a per-env per-joint encoder bias.

    Mirrors mjlab's ``dr.encoder_bias`` — writes a uniform random bias
    in ``bias_range`` into ``env.act_manager._encoder_bias`` so the
    biased observation (:func:`dof_pos_biased`) reflects a static
    calibration offset for each episode. Typically registered with
    mode ``"reset_dr"`` so the bias is resampled on each env reset.

    Works uniformly on Newton/Genesis/MuJoCo because it only touches
    the cross-sim ``act_manager`` state.
    """
    if len(env_ids) == 0:
        return
    device = env.device
    n = len(env_ids)
    num_joints = env.act_manager.offset.shape[-1]
    lo, hi = bias_range
    bias = torch.empty((n, num_joints), device=device).uniform_(lo, hi)
    env.act_manager.set_encoder_bias(bias, env_ids=env_ids)


# ── Getup: mixed fallen / standing reset ────────────────────────────


def _sample_uniform_quaternion_wxyz(n: int, device: torch.device) -> torch.Tensor:
    """Uniformly sample unit quaternions on the 3-sphere (Shoemake 1992).

    Returns a ``(n, 4)`` tensor in **wxyz** convention (scalar first).
    This is the standard uniform distribution over ``SO(3)``.
    """
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device)
    u3 = torch.rand(n, device=device)
    two_pi = 2.0 * torch.pi
    s1 = torch.sqrt(1.0 - u1)
    s2 = torch.sqrt(u1)
    w = s1 * torch.sin(two_pi * u2)
    x = s1 * torch.cos(two_pi * u2)
    y = s2 * torch.sin(two_pi * u3)
    z = s2 * torch.cos(two_pi * u3)
    return torch.stack([w, x, y, z], dim=1)


def reset_fallen_or_standing(
    env: World,
    env_ids: torch.Tensor,
    fallen_prob: float = 0.6,
    fall_height: float = 0.8,
    fall_velocity_range: tuple[float, float] = (-0.5, 0.5),
    fall_joint_noise_range: tuple[float, float] | str = "soft_limit",
    standing_z_offset: float = 0.02,
    default_pos: tuple[float, ...] = (0.0, 0.0, 0.665),
    default_quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    default_joint_pos_dict: dict[str, float] | None = None,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> None:
    """Reset robots to a mix of fallen (random orientation) and standing poses.

    Inspired by mjlab_playground getup ``reset_fallen_or_standing``. A
    per-env Bernoulli draw with probability ``fallen_prob`` decides
    which branch each environment takes:

    - **Fallen** (``fallen_prob`` of envs):
        * root z = ``fall_height`` (dropped from above)
        * root orientation = uniform random quaternion on S³
          (Shoemake sampling — truly isotropic over SO(3))
        * joint positions = ``act_manager.offset`` + uniform noise in
          ``fall_joint_noise_range`` (additive, per-joint). This is a
          simplification of mjlab which samples uniformly over the
          full soft joint limit range; sampling from an absolute
          range avoids a cross-sim abstraction for hard/soft limits
          (MuJoCo's RobotData does not expose hard limits directly).
          The skeleton MVP uses ±0.8 rad which gives a visibly fallen
          pose for humanoids without risking extreme self-penetration;
          Phase K tuning can widen to full soft limits via a new
          ``rd.soft_joint_pos_limits`` accessor if needed.
        * root linear + angular velocity and joint velocity uniformly
          sampled in ``fall_velocity_range``

    - **Standing** (``1 - fallen_prob`` of envs):
        * root pose = ``default_pos + (0, 0, standing_z_offset)``
          and ``default_quat_wxyz``
        * joint positions = ``default_joint_pos_dict`` (regex→float
          dict resolved against actuated joint names) when provided,
          otherwise ``act_manager.offset``
        * all velocities = 0

    The two branches are computed on the full ``env_ids`` tensor and
    merged via a boolean mask, so the writer only needs one call per
    quantity (position/velocity/orientation). Works identically across
    Newton, Genesis, and MuJoCo through the shared RobotStateWriter +
    RobotData interface.

    Args:
        env: Any environment with RobotData + Writer APIs.
        env_ids: Environments to reset.
        fallen_prob: Probability (per env) of selecting the fallen
            branch. mjlab default: ``0.6``.
        fall_height: World-z height at which fallen robots are dropped.
        fall_velocity_range: ``(min, max)`` uniform bounds applied to
            every root linear/angular vel component and joint vel in
            the fallen branch.
        fall_joint_noise_range: ``(min, max)`` uniform noise added to
            default joint positions in the fallen branch. Not clipped
            to joint limits — the simulator will clamp on apply.
        standing_z_offset: Small z offset applied to standing pose so
            the robot does not spawn intersecting the ground plane.
        default_pos: Standing-branch root ``(x, y, z)``.
        default_quat_wxyz: Standing-branch root quaternion (wxyz).
        default_joint_pos_dict: Regex→float dict of default joint
            angles (same format as ``RobotConfig.default_joint_angles``).
            Resolved against ``act_manager.actuated_joint_names`` at
            call time. When ``None``, falls back to
            ``act_manager.offset`` (correct for absolute-action
            presets but **all-zero** for relative-action presets that
            force ``use_zero_offset=True``).
        entity_name: Entity to reset.
    """
    if len(env_ids) == 0:
        return

    writer = env.get_robot_state_writer(asset_cfg.name)
    device = env.device
    n = len(env_ids)

    # Auto-detect mjlab multi-env offsets (same convention as
    # ``reset_root_state_uniform``).
    _scene = getattr(env.scene_manager, "scene", None)
    env_origins = getattr(_scene, "env_origins", None)

    # Per-env branch mask.
    is_fallen = torch.rand(n, device=device) < fallen_prob
    is_fallen_3 = is_fallen.unsqueeze(-1)  # for broadcasting to (n, 3)

    # ── Root pose ─────────────────────────────────────────────────
    default_pos_t = torch.tensor(default_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    standing_pos = default_pos_t.clone()
    standing_pos[:, 2] = standing_pos[:, 2] + standing_z_offset

    fallen_pos = torch.zeros_like(standing_pos)
    fallen_pos[:, 2] = fall_height

    pos = torch.where(is_fallen_3, fallen_pos, standing_pos)
    if env_origins is not None:
        pos = pos + env_origins[env_ids]

    default_quat_t = torch.tensor(default_quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    random_quat = _sample_uniform_quaternion_wxyz(n, device)
    quat_wxyz = torch.where(is_fallen.unsqueeze(-1), random_quat, default_quat_t)

    # ── Root velocity ─────────────────────────────────────────────
    lo, hi = fall_velocity_range
    fallen_lin_vel = torch.empty((n, 3), device=device).uniform_(lo, hi)
    fallen_ang_vel = torch.empty((n, 3), device=device).uniform_(lo, hi)
    zero_vel = torch.zeros((n, 3), device=device)
    lin_vel = torch.where(is_fallen_3, fallen_lin_vel, zero_vel)
    ang_vel = torch.where(is_fallen_3, fallen_ang_vel, zero_vel)

    # ── Joint state ───────────────────────────────────────────────
    # Build default joint positions from the explicit dict when
    # provided, falling back to act_manager.offset for backward
    # compatibility. Relative-action presets (getup) zero the
    # action offset, so relying on it produces an all-zero home
    # pose instead of the real standing configuration.
    if default_joint_pos_dict is not None:
        rd = env.get_robot_data(asset_cfg.name)
        default_joint_pos = rd.default_joint_pos.unsqueeze(0).expand(n, -1).clone()
    else:
        default_joint_pos = env.act_manager.offset[env_ids].clone()
        if default_joint_pos.dim() == 1:
            default_joint_pos = default_joint_pos.unsqueeze(0)
    num_joints = default_joint_pos.shape[-1]

    if fall_joint_noise_range == "soft_limit":
        # mjlab_playground-faithful path: sample uniform over the
        # full soft joint limit range so fallen poses span the whole
        # reachable configuration space. Requires
        # ``rd.soft_joint_pos_limits`` (all 3 sims implement it).
        rd = env.get_robot_data(asset_cfg.name)
        jp_lo, jp_hi = rd.soft_joint_pos_limits
        u = torch.rand((n, num_joints), device=device)
        fallen_joint_pos = jp_lo + u * (jp_hi - jp_lo)
    else:
        # Additive-noise path (simpler fallback, keeps fallen pose
        # close to the default — useful for robots whose soft limits
        # haven't been validated).
        jn_lo, jn_hi = fall_joint_noise_range
        joint_noise = torch.empty((n, num_joints), device=device).uniform_(jn_lo, jn_hi)
        fallen_joint_pos = default_joint_pos + joint_noise

    is_fallen_j = is_fallen.unsqueeze(-1).expand(-1, num_joints)
    joint_pos = torch.where(is_fallen_j, fallen_joint_pos, default_joint_pos)

    fallen_joint_vel = torch.empty((n, num_joints), device=device).uniform_(lo, hi)
    zero_joint_vel = torch.zeros((n, num_joints), device=device)
    joint_vel = torch.where(is_fallen_j, fallen_joint_vel, zero_joint_vel)

    # ── Write ─────────────────────────────────────────────────────
    writer.set_root_pose(pos, quat_wxyz, env_ids=env_ids)
    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)
    writer.set_dof_positions(joint_pos, env_ids=env_ids)
    writer.set_dof_velocities(joint_vel, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)
