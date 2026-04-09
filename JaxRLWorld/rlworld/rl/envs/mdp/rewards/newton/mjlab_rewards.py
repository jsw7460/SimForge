"""mjlab-compatible reward functions for Newton environments.

These functions produce identical outputs to mjlab rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
    get_bodies_quat,
)
from rlworld.rl.envs.mdp.observations.newton.state import (
    _quat_rotate,
    _quat_rotate_inverse,
    base_quat,
)
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    FeetSwingHeightTracker,
    penalize_angular_momentum_l2,
    penalize_body_ang_vel_xy,
    penalize_contact_force_count,
    penalize_feet_clearance,
    penalize_feet_slip,
    penalize_joint_pos_limits_l1,
    penalize_soft_landing,
    raw_action_rate_l2,
)
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


# ============================================================
# track_lin_vel_mjlab
# ============================================================

def track_lin_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> torch.Tensor:
    """Reward for tracking commanded base linear velocity.

    Matches mjlab.tasks.velocity.mdp.track_linear_velocity exactly.
    Includes z velocity penalty (commanded z is assumed to be zero).

    Args:
        env: Newton environment.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], dim=1)  # (num_envs, 2)

    actual = env.robot_data.root_link_lin_vel_b  # (num_envs, 3)

    xy_error = torch.sum(torch.square(command - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    return torch.exp(-lin_vel_error / (std ** 2))


# ============================================================
# track_ang_vel_mjlab
# ============================================================

def track_ang_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity.

    Matches mjlab.tasks.velocity.mdp.track_angular_velocity exactly.
    Includes xy angular velocity penalty (commanded xy is assumed to be zero).

    Args:
        env: Newton environment.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command_z = env.command_manager.ang_vel  # (num_envs,)

    actual = env.robot_data.root_link_ang_vel_b  # (num_envs, 3)

    z_error = torch.square(command_z - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error

    return torch.exp(-ang_vel_error / (std ** 2))


# ============================================================
# flat_orientation_mjlab
# ============================================================

def flat_orientation_mjlab(
    env: "NewtonEnv",
    std: float,
    body_name: str | None = None,
) -> torch.Tensor:
    if body_name is not None:
        result = get_bodies_quat(env, body_name)
        body_quat_xyzw = result.data[:, 0, :]  # (num_envs, 4)

        # Cache normalized gravity vector
        if not hasattr(env, '_gravity_normalized_cache'):
            env._gravity_normalized_cache = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=env.device,
                dtype=torch.float32
            ).expand(env.num_envs, -1).contiguous()

        projected_gravity_b = _quat_rotate_inverse(body_quat_xyzw, env._gravity_normalized_cache)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        projected_gravity_b = env.robot_data.projected_gravity_b  # (num_envs, 3)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / (std ** 2))


# ============================================================
# body_ang_vel_penalty_mjlab
# ============================================================

def body_ang_vel_penalty_mjlab(
    env: "NewtonEnv",
    body_name: str,
) -> torch.Tensor:
    """Penalize excessive body angular velocities (xy only).

    Delegates to ``common.penalize_body_ang_vel_xy``. The Newton
    implementation of ``RobotData.find_body_index`` calls the same
    ``body_cache.get_body_indices(body_name)`` that the legacy code
    used, and ``RobotData.body_ang_vel_w`` reads the same
    ``state.body_qd[:, body_idx, 3:6]`` slice — so the result is
    bit-identical to the legacy direct-access path.

    Args:
        env: Newton environment.
        body_name: Name of the body to penalize.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    return penalize_body_ang_vel_xy(env, body_name=body_name)


# ============================================================
# feet_air_time_mjlab
# ============================================================

def feet_air_time_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time.

    Matches mjlab.tasks.velocity.mdp.feet_air_time exactly.

    Args:
        env: Newton environment.
        feet_bodies: Foot body name pattern(s).
        threshold_min: Minimum air time to receive reward.
        threshold_max: Maximum air time to receive reward.
        command_threshold: Minimum command velocity to activate reward.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_indices = result.contact_indices

    current_air_time = env.contact_manager.current_air_time("foot_contact")[:, contact_indices]

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    scale = (total_command > command_threshold).float()
    return reward * scale


# ============================================================
# feet_clearance_mjlab
# ============================================================

def feet_clearance_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    target_height: float,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_feet_clearance``.

    Bit-identical to the legacy direct-access path: the common helper
    reads ``RobotData.body_pos_w/body_lin_vel_w`` which on Newton both
    pull from the same ``state.body_q/body_qd`` views the legacy code
    used. ``feet_bodies`` is forwarded as the ``body_names`` list.
    """
    names = [feet_bodies] if isinstance(feet_bodies, str) else list(feet_bodies)
    return penalize_feet_clearance(
        env,
        target_height=target_height,
        command_threshold=command_threshold,
        body_names=names,
    )


# ============================================================
# feet_slip_mjlab
# ============================================================

def feet_slip_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_feet_slip``.

    Bit-identical: the common helper passes ``order=feet_bodies`` to the
    contact manager, which reorders ``is_contact("foot_contact")`` by
    name — equivalent to the legacy ``contact_indices`` cache lookup.
    """
    names = [feet_bodies] if isinstance(feet_bodies, str) else list(feet_bodies)
    return penalize_feet_slip(
        env,
        contact_group="foot_contact",
        command_threshold=command_threshold,
        body_names=names,
    )


# ============================================================
# soft_landing_mjlab
# ============================================================

def soft_landing_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_soft_landing``.

    Bit-identical: the cost is summed over feet so reordering does not
    affect the result. We pass ``contact_order=feet_bodies`` only to
    keep the call-site explicit about which group elements are intended;
    the legacy code's ``contact_indices`` slicing reorders the same
    columns before the sum.
    """
    names = [feet_bodies] if isinstance(feet_bodies, str) else list(feet_bodies)
    return penalize_soft_landing(
        env,
        contact_group="foot_contact",
        command_threshold=command_threshold,
        contact_order=names,
    )


# ============================================================
# joint_pos_limits_mjlab
# ============================================================

def joint_pos_limits_mjlab(
    env: "NewtonEnv",
    soft_limit_factor: float = 1.0,
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits.

    Delegates to ``common.penalize_joint_pos_limits_l1`` which reads
    ``RobotData.joint_pos`` and ``RobotData.joint_pos_limits``. The
    Newton implementation of ``joint_pos_limits`` reads the same
    ``model.joint_limit_lower/upper`` arrays indexed by
    ``newton_qd_indices`` that the legacy code accessed, so the result
    is bit-identical to the legacy direct-access path.

    Args:
        env: Newton environment.
        soft_limit_factor: Factor to scale hard limits to get soft limits.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    return penalize_joint_pos_limits_l1(env, soft_limit_factor=soft_limit_factor)


# ============================================================
# action_rate_l2_mjlab
# ============================================================

def raw_action_rate_l2_mjlab(env: "NewtonEnv") -> torch.Tensor:
    """Penalize the rate of change of raw actions (L2 squared).

    Delegates to ``common.raw_action_rate_l2``. Bit-identical: same
    pure-Python ``act_manager`` arithmetic, no scene-state involved.
    """
    return raw_action_rate_l2(env)

def processed_action_rate_l2_mjlab(env: "NewtonEnv") -> torch.Tensor:
    """Penalize the rate of change of actions using L2 squared kernel.

    Matches mjlab.envs.mdp.action_rate_l2 exactly.

    Args:
        env: Newton environment.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions),
        dim=1
    )


# ============================================================
# variable_posture
# ============================================================

class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.

    Uses per-joint standard deviations to control how much each joint can deviate
    from default pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))

    Three speed regimes (based on linear + angular command velocity):
      - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
      - std_walking (walking_threshold <= speed < running_threshold): Moderate.
      - std_running (speed >= running_threshold): Loose tolerance for large motion.

    Matches mjlab.tasks.velocity.mdp.variable_posture exactly.
    """

    __name__ = "variable_posture"

    def __init__(
        self,
        env: "NewtonEnv",
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ):
        self.env = env
        self.walking_threshold = walking_threshold
        self.running_threshold = running_threshold

        joint_names = env.act_manager.actuated_joint_names

        _, _, std_standing_values = string_utils.resolve_matching_names_values(
            std_standing, joint_names
        )
        self.std_standing = torch.tensor(
            std_standing_values, device=env.device, dtype=torch.float32
        )

        _, _, std_walking_values = string_utils.resolve_matching_names_values(
            std_walking, joint_names
        )
        self.std_walking = torch.tensor(
            std_walking_values, device=env.device, dtype=torch.float32
        )

        _, _, std_running_values = string_utils.resolve_matching_names_values(
            std_running, joint_names
        )
        self.std_running = torch.tensor(
            std_running_values, device=env.device, dtype=torch.float32
        )

        self.default_joint_pos = env.act_manager.offset

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        command = torch.stack([lin_vel_x, lin_vel_y, ang_vel], dim=1)

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self.walking_threshold).float()
        walking_mask = (
            (total_speed >= self.walking_threshold) & (total_speed < self.running_threshold)
        ).float()
        running_mask = (total_speed >= self.running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = env.robot_data.joint_pos
        error_squared = torch.square(current_joint_pos - self.default_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std ** 2), dim=1))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


# ============================================================
# feet_swing_height_mjlab
# ============================================================

class feet_swing_height_mjlab:
    """Thin wrapper around ``common.FeetSwingHeightTracker`` (Newton legacy reset).

    Preserves bit-identity by setting ``reset_mode="current_foot_height"``,
    which re-seeds peak heights to the current foot z on episode reset
    (Newton's original behavior — different from Genesis/MuJoCo).
    """

    __name__ = "feet_swing_height_mjlab"

    def __init__(
        self,
        env: "NewtonEnv",
        feet_bodies: str | list[str],
        target_height: float,
        command_threshold: float = 0.05,
    ):
        names = [feet_bodies] if isinstance(feet_bodies, str) else list(feet_bodies)
        self._impl = FeetSwingHeightTracker(
            env=env,
            contact_group="foot_contact",
            target_height=target_height,
            command_threshold=command_threshold,
            body_names=names,
            use_squared_error=True,
            reset_mode="current_foot_height",
        )

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        return self._impl(env)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._impl.reset(env_ids)


# ============================================================
# feet_swing_height (alias for backward compatibility)
# ============================================================

class feet_swing_height:
    """Walk-These-Ways variant: absolute (not squared) error.

    Identical state machinery to ``feet_swing_height_mjlab`` but
    ``use_squared_error=False``. Kept distinct because the cross-sim
    comparison test scripts reference both classes.
    """

    __name__ = "feet_swing_height"

    def __init__(
        self,
        env: "NewtonEnv",
        feet_bodies: str | list[str],
        target_height: float,
        command_threshold: float = 0.05,
    ):
        names = [feet_bodies] if isinstance(feet_bodies, str) else list(feet_bodies)
        self._impl = FeetSwingHeightTracker(
            env=env,
            contact_group="foot_contact",
            target_height=target_height,
            command_threshold=command_threshold,
            body_names=names,
            use_squared_error=False,
            reset_mode="current_foot_height",
        )

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        return self._impl(env)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._impl.reset(env_ids)


# ============================================================
# angular_momentum_penalty
# ============================================================

def angular_momentum_penalty(
    env: "NewtonEnv",
) -> torch.Tensor:
    """Penalize whole-body angular momentum.

    Delegates to ``common.penalize_angular_momentum_l2``. Bit-identical
    to the legacy implementation: ``RobotData.angular_momentum_w`` for
    Newton uses the same ``model.body_inertia`` warp array, the same
    ``state.body_q/qd`` reshape, and the same xyzw quaternion rotation
    helpers (``_quat_rotate_inverse`` / ``_quat_rotate``) that this
    function previously called inline.
    """
    return penalize_angular_momentum_l2(env)