"""MuJoCo (mjlab) builders for the workcell. See ``_newton_builders``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from jaxrlworld.assets.unitree_go2.go2_constants import get_spec as go2_get_spec
from jaxrlworld.rl.actuators import ImplicitActuatorCfg
from jaxrlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoEnvConfig,
    MujocoSceneConfig,
)
from jaxrlworld.rl.configs.presets.yam_arm import _mujoco_builders as arm
from jaxrlworld.rl.configs.robots.go2 import (
    ARMATURE_HIP,
    ARMATURE_KNEE,
    DAMPING_HIP,
    DAMPING_KNEE,
    EFFORT_HIP,
    EFFORT_KNEE,
    STIFFNESS_HIP,
    STIFFNESS_KNEE,
)
from jaxrlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)

from .base import QUADRUPED, build_action_terms

if TYPE_CHECKING:
    from .base import LabCellConfig

CONFIGS_FOR_RUN_CLS = arm.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = arm.OBSERVATION_CFG_CLS

build_visualization = arm.build_visualization
build_reward = arm.build_reward
build_dr_terms = arm.build_dr_terms


def build_env(cfg: LabCellConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    env = arm.build_env(cfg, timing)
    env.task_name = "Lab_Cell"
    return env


def build_scene(cfg: LabCellConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    scene = arm.build_scene(cfg, timing)
    q = cfg.quadruped

    scene.entities[QUADRUPED] = MujocoEntityCfg(
        urdf_path=q.urdf_path,
        init_state=InitialStateCfg(
            pos=cfg.quadruped_pos,
            joint_pos=q.default_joint_angles,
        ),
        floating=True,
        articulation=ArticulationCfg(
            actuators=(
                ImplicitActuatorCfg(
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                    stiffness=STIFFNESS_HIP,
                    damping=DAMPING_HIP,
                    armature=ARMATURE_HIP,
                    effort_limit=EFFORT_HIP,
                ),
                ImplicitActuatorCfg(
                    target_names_expr=(".*_calf_joint",),
                    stiffness=STIFFNESS_KNEE,
                    damping=DAMPING_KNEE,
                    armature=ARMATURE_KNEE,
                    effort_limit=EFFORT_KNEE,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        spec_fn=go2_get_spec,
    )
    # Two robots' limits and contacts against one buffer; an overflow
    # prints nefc overflow and silently DROPS constraints.
    scene.njmax = 1500
    scene.nconmax = 800
    return scene


def build_action(cfg: LabCellConfig) -> MujocoActionConfig:
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=cfg.robot.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms=build_action_terms(cfg),
    )
