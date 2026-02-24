from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MjlabEnv


# =============================================================================
# Entity Configuration (simplified from mjlab's SceneEntityCfg)
# =============================================================================

@dataclass
class EntityCfg:
    """Configuration specifying which entity and components to operate on."""
    name: str = "robot"
    joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))
    body_ids: list[int] | slice = field(default_factory=lambda: slice(None))
    geom_ids: list[int] | slice = field(default_factory=lambda: slice(None))
    site_ids: list[int] | slice = field(default_factory=lambda: slice(None))
    actuator_ids: list[int] | slice = field(default_factory=lambda: slice(None))

    # Named components (resolved to IDs at runtime)
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    geom_names: tuple[str, ...] | None = None
    site_names: tuple[str, ...] | None = None


def _resolve_entity_cfg(env: "MjlabEnv", cfg: EntityCfg) -> EntityCfg:
    """Resolve named components to IDs."""
    entity = env.scene_manager.get_entity(cfg.name)

    # mjlab find_* methods return (indices, names) tuples
    if cfg.joint_names is not None:
        cfg.joint_ids, _ = entity.find_joints(cfg.joint_names)
    if cfg.body_names is not None:
        cfg.body_ids, _ = entity.find_bodies(cfg.body_names)
    if cfg.geom_names is not None:
        cfg.geom_ids, _ = entity.find_geoms(cfg.geom_names)
    if cfg.site_names is not None:
        cfg.site_ids, _ = entity.find_sites(cfg.site_names)

    return cfg


# =============================================================================
# Sampling utilities
# =============================================================================

def _sample_uniform(
    lower: torch.Tensor,
    upper: torch.Tensor,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Sample from uniform distribution."""
    return (upper - lower) * torch.rand(shape, device=device) + lower


# =============================================================================
# Reset events
# =============================================================================

def reset_scene_to_default(
    env: "MjlabEnv",
    env_ids: torch.Tensor | None = None,
) -> None:
    """Reset all entities in the scene to their default states."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    scene = env.scene_manager.scene
    for entity in scene.entities.values():
        from mjlab.entity import Entity
        if not isinstance(entity, Entity):
            continue

        # Reset root/mocap pose
        if entity.is_fixed_base and entity.is_mocap:
            default_root_state = entity.data.default_root_state[env_ids].clone()
            mocap_pose = torch.zeros((len(env_ids), 7), device=env.device)
            mocap_pose[:, 0:3] = default_root_state[:, 0:3] + scene.env_origins[env_ids]
            mocap_pose[:, 3:7] = default_root_state[:, 3:7]
            entity.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)
        elif not entity.is_fixed_base:
            default_root_state = entity.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += scene.env_origins[env_ids]
            entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        # Reset joint state
        if entity.is_articulated:
            default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
            default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
            entity.write_joint_state_to_sim(
                default_joint_pos, default_joint_vel, env_ids=env_ids
            )


def reset_root_state_uniform(
    env: "MjlabEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Reset root state with uniform random perturbations."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    entity_cfg = entity_cfg or EntityCfg()
    entity = env.scene_manager.get_entity(entity_cfg.name)
    scene = env.scene_manager.scene

    # Sample pose perturbations
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = _sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), env.device
    )

    default_root_state = entity.data.default_root_state[env_ids].clone()

    # Position
    positions = (
        default_root_state[:, 0:3]
        + pose_samples[:, 0:3]
        + scene.env_origins[env_ids]
    )

    # Orientation
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(default_root_state[:, 3:7], orientations_delta)

    if entity.is_fixed_base:
        if not entity.is_mocap:
            raise ValueError(f"Cannot reset root state for fixed-base non-mocap entity.")
        entity.write_mocap_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        return

    # Floating-base: also reset velocities
    velocity_range = velocity_range or {}
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=env.device)
    vel_samples = _sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), env.device
    )
    velocities = default_root_state[:, 7:13] + vel_samples

    entity.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    entity.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
    env: "MjlabEnv",
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Reset joint positions and velocities with uniform offsets."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    entity_cfg = entity_cfg or EntityCfg()
    entity_cfg = _resolve_entity_cfg(env, entity_cfg)
    entity = env.scene_manager.get_entity(entity_cfg.name)

    joint_ids = entity_cfg.joint_ids

    # Get defaults
    default_joint_pos = entity.data.default_joint_pos[env_ids]
    default_joint_vel = entity.data.default_joint_vel[env_ids]
    soft_joint_pos_limits = entity.data.soft_joint_pos_limits

    # Select joints
    if isinstance(joint_ids, slice):
        joint_pos = default_joint_pos.clone()
        joint_vel = default_joint_vel.clone()
        limits = soft_joint_pos_limits[env_ids] if soft_joint_pos_limits is not None else None
    else:
        joint_pos = default_joint_pos[:, joint_ids].clone()
        joint_vel = default_joint_vel[:, joint_ids].clone()
        limits = soft_joint_pos_limits[env_ids][:, joint_ids] if soft_joint_pos_limits is not None else None

    # Add noise
    joint_pos += _sample_uniform(
        torch.tensor(position_range[0], device=env.device),
        torch.tensor(position_range[1], device=env.device),
        joint_pos.shape,
        env.device,
    )
    joint_vel += _sample_uniform(
        torch.tensor(velocity_range[0], device=env.device),
        torch.tensor(velocity_range[1], device=env.device),
        joint_vel.shape,
        env.device,
    )

    # Clamp to limits
    if limits is not None:
        joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    # Convert joint_ids for write
    if isinstance(joint_ids, list):
        joint_ids_tensor = torch.tensor(joint_ids, device=env.device)
    else:
        joint_ids_tensor = None

    entity.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids_tensor
    )


# =============================================================================
# Interval events (disturbances)
# =============================================================================

def push_by_setting_velocity(
    env: "MjlabEnv",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Apply push disturbance by setting root velocity."""
    entity_cfg = entity_cfg or EntityCfg()
    entity = env.scene_manager.get_entity(entity_cfg.name)

    vel_w = entity.data.root_link_vel_w[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=env.device)

    vel_w += _sample_uniform(
        ranges[:, 0], ranges[:, 1], vel_w.shape, env.device
    )
    entity.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


# =============================================================================
# Domain randomization events
# =============================================================================

def randomize_field(
    env: "MjlabEnv",
    field: str,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    operation: Literal["add", "scale", "abs"] = "abs",
    entity_cfg: EntityCfg | None = None,
    axes: list[int] | None = None,
    shared_random: bool = False,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Randomize MuJoCo model field.

    Directly calls mjlab's randomize_field function.
    """
    from mjlab.envs.mdp.events import randomize_field as mjlab_randomize_field
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    entity_cfg = entity_cfg or EntityCfg()

    # Convert to mjlab's SceneEntityCfg
    scene_entity_cfg = SceneEntityCfg(
        name=entity_cfg.name,
        joint_ids=entity_cfg.joint_ids,
        body_ids=entity_cfg.body_ids,
        geom_ids=entity_cfg.geom_ids,
        site_ids=entity_cfg.site_ids,
        joint_names=entity_cfg.joint_names,
        body_names=entity_cfg.body_names,
        geom_names=entity_cfg.geom_names,
        site_names=entity_cfg.site_names,
    )

    # Create mjlab-compatible env wrapper
    class MjlabEnvAdapter:
        def __init__(self, rlworld_env: "MjlabEnv"):
            self._env = rlworld_env

        @property
        def num_envs(self):
            return self._env.num_envs

        @property
        def device(self):
            return self._env.device

        @property
        def scene(self):
            return self._env.scene_manager.scene

        @property
        def sim(self):
            return self._env.scene_manager.sim

    adapter = MjlabEnvAdapter(env)
    mjlab_randomize_field(
        env=adapter,
        env_ids=env_ids,
        field=field,
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=scene_entity_cfg,
        axes=axes,
        shared_random=shared_random,
    )


def randomize_encoder_bias(
    env: "MjlabEnv",
    bias_range: tuple[float, float],
    entity_cfg: EntityCfg | None = None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Randomize joint encoder bias."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    entity_cfg = entity_cfg or EntityCfg()
    entity_cfg = _resolve_entity_cfg(env, entity_cfg)
    entity = env.scene_manager.get_entity(entity_cfg.name)

    joint_ids = entity_cfg.joint_ids
    if isinstance(joint_ids, slice):
        num_joints = entity.num_joints
        joint_ids_tensor = torch.arange(num_joints, device=env.device)
    else:
        joint_ids_tensor = torch.tensor(joint_ids, device=env.device)

    num_joints = len(joint_ids_tensor)
    bias_samples = _sample_uniform(
        torch.tensor(bias_range[0], device=env.device),
        torch.tensor(bias_range[1], device=env.device),
        (len(env_ids), num_joints),
        env.device,
    )

    if isinstance(joint_ids, slice):
        entity.data.encoder_bias[env_ids] = bias_samples
    else:
        entity.data.encoder_bias[env_ids[:, None], joint_ids_tensor] = bias_samples


def randomize_pd_gains(
    env: "MjlabEnv",
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    distribution: Literal["uniform", "log_uniform"] = "uniform",
    operation: Literal["scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Randomize PD gains."""
    from mjlab.envs.mdp.events import randomize_pd_gains as mjlab_randomize_pd_gains
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    entity_cfg = entity_cfg or EntityCfg()

    scene_entity_cfg = SceneEntityCfg(
        name=entity_cfg.name,
        actuator_ids=entity_cfg.actuator_ids,
    )

    class MjlabEnvAdapter:
        def __init__(self, rlworld_env: "MjlabEnv"):
            self._env = rlworld_env

        @property
        def num_envs(self):
            return self._env.num_envs

        @property
        def device(self):
            return self._env.device

        @property
        def scene(self):
            return self._env.scene_manager.scene

        @property
        def sim(self):
            return self._env.scene_manager.sim

    adapter = MjlabEnvAdapter(env)
    mjlab_randomize_pd_gains(
        env=adapter,
        env_ids=env_ids,
        kp_range=kp_range,
        kd_range=kd_range,
        asset_cfg=scene_entity_cfg,
        distribution=distribution,
        operation=operation,
    )