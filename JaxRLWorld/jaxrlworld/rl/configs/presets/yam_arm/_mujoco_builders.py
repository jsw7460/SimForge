"""MuJoCo (mjlab) builders for the YAM fixed-base arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import mujoco

from jaxrlworld.rl.actuators import ImplicitActuatorCfg
from jaxrlworld.rl.configs import RewardConfig, TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
    VisualizationConfig,
)
from jaxrlworld.rl.configs.robots.yam import (
    YAM_ACTION_SCALE,
    YAM_ARMATURE,
    YAM_DAMPING,
    YAM_EFFORT_LIMIT,
    YAM_STIFFNESS,
)
from jaxrlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed

from .base import build_rigid_objects

if TYPE_CHECKING:
    from .base import YamArmConfig

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


# ── mjlab-only asset wiring ──────────────────────────────────────────
# mjlab builds an articulation from a spec function rather than a path,
# and its collision policy selects geoms by name. Both are mjlab types,
# so they live here — this module is imported only when sim_type is
# "mujoco", which keeps mjlab out of the Newton and Genesis paths.


@dataclass
class YamSpecFn:
    """Picklable ``spec_fn``: load the vendored MJCF.

    A dataclass rather than a bare function so the path comes from the
    robot config instead of being restated here — the other backends
    read the same ``mjcf_path``, and two copies of it would be free to
    drift.

    Nothing is stripped from the spec: the vendored XML carries no
    ``<actuator>`` block. That matters on this backend specifically —
    mjlab *adds* one ``<position>`` element per declared actuator, so an
    XML that already named the same actuators would fail to compile on
    the duplicate (which is why the K1 spec function deletes them).
    """

    mjcf_path: str

    def __call__(self) -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(Path(self.mjcf_path).resolve()))


# The arm's contact parameters used to be set HERE, as a CollisionCfg over
# the compiled spec, which only mjlab can do -- so Newton and Genesis never
# received them. b803250 levelled the values up until this config only
# restated what the model already compiles to, and the parameter sweep
# confirms it: friction, solref, solimp, condim and priority now read the
# same on all three for every geom
# (jaxrlworld/scripts/diag/contact_param_parity_diag.py).
#
# What was still left is why it is gone. CollisionCfg matches geoms by NAME
# and zeroes contype / conaffinity on everything it does not match, and one
# geom in the arm has no name at all: the wrist camera's collision box
# (assets/i2rt_yam/xmls/yam.xml, the only unnamed one of 27). An expression
# cannot match a nameless geom, so mjlab alone built the arm without its
# camera housing -- 40 collidable geoms against 41 -- and the camera is a
# real object that sticks out of the wrist, which the other two backends
# collide. Do not reintroduce a name-keyed collision config here; state
# what you mean in the asset, where every backend reads it.


def build_visualization(cfg: YamArmConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: YamArmConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        time_out = TerminationTermConfig(max_episode_exceed)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="Yam_Arm",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: YamArmConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    r = cfg.robot

    robot_entity = MujocoEntityCfg(
        init_state=InitialStateCfg(
            pos=cfg.base_pos,
            joint_pos=r.default_joint_angles,
        ),
        # mjlab decides the base type from the spec, not from this flag:
        # the XML's root body carries no free joint, and mjlab wraps such
        # an entity in a mocap body so it stays placeable per env.
        floating=r.floating,
        articulation=ArticulationCfg(
            actuators=(
                ImplicitActuatorCfg(
                    target_names_expr=tuple(r.actuated_dof_patterns),
                    stiffness=YAM_STIFFNESS,
                    damping=YAM_DAMPING,
                    armature=YAM_ARMATURE,
                    effort_limit=YAM_EFFORT_LIMIT,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        # mjlab builds articulations from a spec function, not a path.
        spec_fn=YamSpecFn(mjcf_path=r.mjcf_path),
        mjcf_path=r.mjcf_path,
    )

    return MujocoSceneConfig(
        physics_dt=timing["dt"],
        substeps=timing.get("substeps", 1),
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        entities={"robot": robot_entity},
        rigid_objects=build_rigid_objects(cfg),
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        # The arm's own limit + contact constraints overflow a small
        # buffer; an overflow silently DROPS constraints rather than
        # erroring, so joints escape their limits.
        njmax=600,
        nconmax=400,
        impratio=10.0,
        cone="pyramidal",
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_action(cfg: YamArmConfig) -> MujocoActionConfig:
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=YAM_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: YamArmConfig) -> RewardConfig:
    @dataclass
    class _RewardsCfg(RewardConfig):
        pass

    return _RewardsCfg()


def build_dr_terms(cfg: YamArmConfig) -> Dict[str, EventTermConfig]:
    return {}
