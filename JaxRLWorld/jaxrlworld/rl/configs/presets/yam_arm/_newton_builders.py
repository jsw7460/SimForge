"""Newton builders for the YAM fixed-base arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from jaxrlworld.rl.actuators import ImplicitActuatorCfg
from jaxrlworld.rl.configs import RewardConfig, TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
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
    NewtonEntityCfg,
)
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed

from .base import build_rigid_objects

if TYPE_CHECKING:
    from .base import YamArmConfig

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def build_visualization(cfg: YamArmConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: YamArmConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        # Time-out only. Orientation and height terminations would never
        # fire on a base that cannot move.
        time_out = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        # Plain env, not NewtonLocomotionEnv: that one requires a gait
        # config, and this preset has none.
        env_name="NewtonEnv",
        task_name="Yam_Arm",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: YamArmConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        robot_cfg=r,
        # Detect contact where the other two backends do. Newton's builder
        # defaults every shape to a 0.1 m gap, and the broad phase expands
        # an AABB by margin + gap on BOTH sides, so anything within 0.2 m
        # arrives as a contact row that carries no force and holds an
        # nconmax slot. The MJCF declares no gap and mjlab reads 0 from the
        # same file. Measured on this bench at rest: 16 rows against
        # Newton's 260, every one of the extra between 79 and 199.99 mm.
        #
        # Set here rather than globally: every other preset's nconmax was
        # measured against the 0.1, and re-sizing them is not this task's
        # to do.
        rigid_gap=0.0,
        solver_cfg=SolverMuJoCoCfg(
            impratio=10.0,
            cone="pyramidal",
            iterations=50,
            ls_iterations=50,
            ccd_iterations=50,
            # The gripper's many small pad geoms against a workpiece
            # overflow a tight budget, and an overflow silently DROPS
            # contacts rather than erroring, so this is sized off a
            # measurement with room on top, never guessed downward.
            #
            # Measured with rigid_gap=0.0 on the bench scene
            # (settling, a scripted reach, and random actions): worst 65
            # rows per world on mjlab, 51 on Newton. 300 is 4.6x that.
            # It was 400 against a peak of 315, which was mostly the
            # phantom rows the 0.1 m gap produced.
            nconmax=300,
        ),
        entities={
            "robot": NewtonEntityCfg(
                # MJCF, never URDF: the gripper's two fingers are coupled
                # by a MuJoCo <equality> constraint that only the MJCF
                # path carries.
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=cfg.base_pos,
                    joint_pos=r.default_joint_angles,
                ),
                floating=r.floating,
                # The XML ships no <actuator> block, so mjwarp would build
                # nu=0 on its own; declaring the actuators here is what
                # sets each joint's target mode to POSITION.
                articulation=ArticulationCfg(
                    actuators=(
                        ImplicitActuatorCfg(
                            # right_finger is absent on purpose: it is
                            # driven by the equality constraint, and
                            # actuating it too would fight that.
                            target_names_expr=tuple(r.actuated_dof_patterns),
                            stiffness=YAM_STIFFNESS,
                            damping=YAM_DAMPING,
                            armature=YAM_ARMATURE,
                            effort_limit=YAM_EFFORT_LIMIT,
                        ),
                    ),
                    soft_joint_pos_limit_factor=0.9,
                ),
                body_label_prefix=r.name,
                enable_self_collisions=False,
            ),
        },
        rigid_objects=build_rigid_objects(cfg),
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_action(cfg: YamArmConfig) -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
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
