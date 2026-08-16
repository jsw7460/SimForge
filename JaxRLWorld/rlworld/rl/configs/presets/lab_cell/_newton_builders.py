"""Newton builders for the workcell.

The single-arm builders already carry the solver recipe this gripper and
bench need, so only the quadruped entity and the term-based action config
are added here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonEnvConfig,
    NewtonSceneConfig,
)
from rlworld.rl.configs.presets.yam_arm import _newton_builders as arm
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
    InitialStateCfg,
    NewtonEntityCfg,
)

from .base import QUADRUPED, build_action_terms

if TYPE_CHECKING:
    from .base import LabCellConfig

CONFIGS_FOR_RUN_CLS = arm.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = arm.OBSERVATION_CFG_CLS

build_visualization = arm.build_visualization
build_reward = arm.build_reward
build_dr_terms = arm.build_dr_terms


def build_env(cfg: LabCellConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    env = arm.build_env(cfg, timing)
    env.task_name = "Lab_Cell"
    return env


def build_scene(cfg: LabCellConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    scene = arm.build_scene(cfg, timing)
    q = cfg.quadruped

    # The arm's prefix is left unset: with more than one articulation the
    # scene assigns each entity its own from the entity name, and two
    # entities sharing a prefix would make one's bodies indistinguishable
    # from the other's.
    scene.entities["robot"].body_label_prefix = None
    scene.entities[QUADRUPED] = NewtonEntityCfg(
        mjcf_path=q.mjcf_path,
        init_state=InitialStateCfg(
            pos=cfg.quadruped_pos,
            joint_pos=q.default_joint_angles,
        ),
        floating=True,
        # Collapsing the welded foot frames is what the quadruped's own
        # preset does; the feet are kept so contact reporting can still
        # name them.
        collapse_fixed_joints=True,
        links_to_keep=[
            "FL_foot_joint",
            "FR_foot_joint",
            "RL_foot_joint",
            "RR_foot_joint",
        ],
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
        enable_self_collisions=False,
    )
    # A quadruped's four feet against the ground plus the arm's gripper
    # against the bench and workpiece; an overflow silently DROPS contacts
    # rather than erroring.
    scene.solver_cfg.nconmax = 800
    return scene


def build_action(cfg: LabCellConfig) -> NewtonActionConfig:
    return NewtonActionConfig(
        actuated_dof_names=cfg.robot.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms=build_action_terms(cfg),
    )
