"""Genesis builders for the workcell. See ``_newton_builders``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    SceneConfig,
)
from rlworld.rl.configs.presets.yam_arm import _genesis_builders as arm
from rlworld.rl.configs.robots.go2 import (
    ARMATURE_HIP,
    ARMATURE_KNEE,
    DAMPING_HIP,
    DAMPING_KNEE,
    EFFORT_HIP,
    EFFORT_KNEE,
    STIFFNESS_HIP,
    STIFFNESS_KNEE,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    InitialStateCfg,
)

from .base import QUADRUPED, build_action_terms

if TYPE_CHECKING:
    from .base import LabCellConfig

CONFIGS_FOR_RUN_CLS = arm.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = arm.OBSERVATION_CFG_CLS

build_visualization = arm.build_visualization
build_reward = arm.build_reward
build_dr_terms = arm.build_dr_terms


def build_env(cfg: LabCellConfig, timing: Dict[str, Any]) -> EnvConfig:
    env = arm.build_env(cfg, timing)
    env.task_name = "Lab_Cell"
    return env


def build_scene(cfg: LabCellConfig, timing: Dict[str, Any]) -> SceneConfig:
    scene = arm.build_scene(cfg, timing)
    q = cfg.quadruped

    scene.entities[QUADRUPED] = GenesisEntityCfg(
        # MJCF, matching the quadruped's own preset: its Genesis entity is
        # built from the same file on that path.
        mjcf_path=q.mjcf_path,
        init_state=InitialStateCfg(
            pos=cfg.quadruped_pos,
            joint_pos=q.default_joint_angles,
        ),
        floating=True,
        links_to_keep=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
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
        convexify=True,
        visualize_contact=False,
    )
    # Four feet on the ground on top of the arm's own pairs.
    scene.rigid_options.max_collision_pairs = 200
    # The go2 feet are authored condim=6, which mjwarp honours and Genesis
    # ignores unless these are on. Set here rather than in the arm builder
    # it inherits: the arm's own presets top out at condim 3, and turning
    # these on there would change their physics for nothing.
    scene.rigid_options.enable_torsional_friction = True
    scene.rigid_options.enable_rolling_friction = True
    return scene


def build_action(cfg: LabCellConfig) -> ActionConfig:
    return ActionConfig(
        actuated_dof_names=cfg.robot.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms=build_action_terms(cfg),
    )
