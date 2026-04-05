from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv


# =============================================================================
# Entity Configuration (simplified from mjlab's SceneEntityCfg)
# =============================================================================

def _default_slice() -> slice:
    return slice(None)


@dataclass
class EntityCfg:
    """Configuration specifying which entity and components to operate on."""
    name: str = "robot"
    joint_ids: list[int] | slice = field(default_factory=_default_slice)
    body_ids: list[int] | slice = field(default_factory=_default_slice)
    geom_ids: list[int] | slice = field(default_factory=_default_slice)
    site_ids: list[int] | slice = field(default_factory=_default_slice)
    actuator_ids: list[int] | slice = field(default_factory=_default_slice)

    # Named components (resolved to IDs at runtime)
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    geom_names: tuple[str, ...] | None = None
    site_names: tuple[str, ...] | None = None


def _resolve_entity_cfg(env: "MujocoEnv", cfg: EntityCfg) -> EntityCfg:
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
    env: "MujocoEnv",
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
    env: "MujocoEnv",
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
    env: "MujocoEnv",
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
    env: "MujocoEnv",
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
# Mjlab env adapter (bridges rlworld's MujocoEnv to mjlab's ManagerBasedRlEnv)
# =============================================================================

class _MujocoEnvAdapter:
    """Adapter that exposes the interface mjlab DR functions expect."""

    def __init__(self, rlworld_env: "MujocoEnv"):
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


def _to_scene_entity_cfg(entity_cfg: EntityCfg):
    """Convert EntityCfg to mjlab's SceneEntityCfg."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    return SceneEntityCfg(
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


# =============================================================================
# Domain randomization events
# =============================================================================

def randomize_geom_friction(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "abs",
    entity_cfg: EntityCfg | None = None,
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize geom friction via mjlab's dr.geom_friction."""
    from mjlab.envs.mdp.dr import geom_friction

    entity_cfg = entity_cfg or EntityCfg()
    geom_friction(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        operation=operation,
        axes=axes,
        shared_random=shared_random,
    )


def randomize_body_com_offset(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "add",
    entity_cfg: EntityCfg | None = None,
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize body COM offset (body_ipos) via mjlab's dr.body_com_offset."""
    from mjlab.envs.mdp.dr import body_com_offset

    entity_cfg = entity_cfg or EntityCfg()
    body_com_offset(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        operation=operation,
        axes=axes,
        shared_random=shared_random,
    )


def randomize_encoder_bias(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Randomize joint encoder bias via mjlab's dr.encoder_bias."""
    from mjlab.envs.mdp.dr import encoder_bias

    entity_cfg = entity_cfg or EntityCfg()
    encoder_bias(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        bias_range=bias_range,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
    )


def randomize_body_mass(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize body mass via mjlab's dr.body_mass."""
    from mjlab.envs.mdp.dr import body_mass

    entity_cfg = entity_cfg or EntityCfg()
    body_mass(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        operation=operation,
        shared_random=shared_random,
    )


def randomize_pd_gains(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    distribution: Literal["uniform", "log_uniform"] = "uniform",
    operation: Literal["scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Randomize PD gains via mjlab's dr.pd_gains."""
    from mjlab.envs.mdp.dr import pd_gains

    entity_cfg = entity_cfg or EntityCfg()
    pd_gains(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        kp_range=kp_range,
        kd_range=kd_range,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        distribution=distribution,
        operation=operation,
    )


def randomize_joint_armature(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize joint armature via mjlab's dr.joint_armature."""
    from mjlab.envs.mdp.dr import joint_armature

    entity_cfg = entity_cfg or EntityCfg()
    joint_armature(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        operation=operation,
        shared_random=shared_random,
    )


def randomize_joint_friction(
    env: "MujocoEnv",
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "abs",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize joint friction loss via mjlab's dr.joint_friction."""
    from mjlab.envs.mdp.dr import joint_friction

    entity_cfg = entity_cfg or EntityCfg()
    joint_friction(
        env=_MujocoEnvAdapter(env),
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg),
        operation=operation,
        shared_random=shared_random,
    )