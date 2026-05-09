"""MuJoCo-specific event terms.

General-purpose reset / push functions live in ``common.py``;
domain-randomization functions that wrap mjlab's ``dr`` module remain here.
"""

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


# =============================================================================
# Mjlab env adapter (bridges rlworld's MujocoEnv to mjlab's ManagerBasedRlEnv)
# =============================================================================


class _MujocoEnvAdapter:
    """Adapter that exposes the interface mjlab DR functions expect."""

    def __init__(self, rlworld_env: MujocoEnv):
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


def _to_scene_entity_cfg(entity_cfg: EntityCfg, scene):
    """Convert EntityCfg to mjlab's SceneEntityCfg and resolve names → ids.

    The ``resolve(scene)`` call is essential. mjlab's
    ``_get_entity_indices`` only consults the ``*_ids`` slice (which
    defaults to ``slice(None)`` → all entities); ``*_names`` are silently
    ignored unless ``resolve`` has populated the matching ``*_ids``
    field. Without this, ``EntityCfg(body_names=("trunk",))`` would
    randomize every body in the entity instead of just the trunk.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cfg = SceneEntityCfg(
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
    cfg.resolve(scene)
    return cfg


# =============================================================================
# Domain randomization events
# =============================================================================


def randomize_friction(
    env: MujocoEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "abs",
    entity_cfg: EntityCfg | None = None,
    axes: list[int] | None = None,
    shared_random: bool = False,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize geom friction via mjlab's dr.geom_friction.

    Named ``randomize_friction`` for cross-sim naming consistency with
    Newton (``dr.newton.randomize_friction``) and Genesis
    (``dr.genesis.randomize_friction``). Supports per-axis randomization
    (``axes=[0]``/``[1]``/``[2]`` for slide/spin/roll) and alternative
    distributions (``"log_uniform"`` for the spin/roll axes where
    mjlab_playground's getup task samples over a 200x range).
    """
    from mjlab.envs.mdp.dr import geom_friction

    entity_cfg = entity_cfg or EntityCfg()
    adapter = _MujocoEnvAdapter(env)
    geom_friction(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        operation=operation,
        axes=axes,
        shared_random=shared_random,
        distribution=distribution,
    )


# Backward-compatible alias
randomize_geom_friction = randomize_friction


def randomize_body_com_offset(
    env: MujocoEnv,
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
    adapter = _MujocoEnvAdapter(env)
    body_com_offset(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        operation=operation,
        axes=axes,
        shared_random=shared_random,
    )


def randomize_encoder_bias(
    env: MujocoEnv,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    entity_cfg: EntityCfg | None = None,
) -> None:
    """Randomize joint encoder bias via mjlab's dr.encoder_bias."""
    from mjlab.envs.mdp.dr import encoder_bias

    entity_cfg = entity_cfg or EntityCfg()
    adapter = _MujocoEnvAdapter(env)
    encoder_bias(
        env=adapter,
        env_ids=env_ids,
        bias_range=bias_range,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
    )


def randomize_body_mass(
    env: MujocoEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize body mass via mjlab's dr.body_mass."""
    from mjlab.envs.mdp.dr import body_mass

    entity_cfg = entity_cfg or EntityCfg()
    adapter = _MujocoEnvAdapter(env)
    body_mass(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        operation=operation,
        shared_random=shared_random,
    )


def randomize_pd_gains(
    env: MujocoEnv,
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
    adapter = _MujocoEnvAdapter(env)
    pd_gains(
        env=adapter,
        env_ids=env_ids,
        kp_range=kp_range,
        kd_range=kd_range,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        distribution=distribution,
        operation=operation,
    )


def randomize_joint_armature(
    env: MujocoEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "scale",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize joint armature via mjlab's dr.joint_armature."""
    from mjlab.envs.mdp.dr import joint_armature

    entity_cfg = entity_cfg or EntityCfg()
    adapter = _MujocoEnvAdapter(env)
    joint_armature(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        operation=operation,
        shared_random=shared_random,
    )


def randomize_joint_friction(
    env: MujocoEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    operation: Literal["add", "scale", "abs"] = "abs",
    entity_cfg: EntityCfg | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize joint friction loss via mjlab's dr.joint_friction."""
    from mjlab.envs.mdp.dr import joint_friction

    entity_cfg = entity_cfg or EntityCfg()
    adapter = _MujocoEnvAdapter(env)
    joint_friction(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=_to_scene_entity_cfg(entity_cfg, adapter.scene),
        operation=operation,
        shared_random=shared_random,
    )
