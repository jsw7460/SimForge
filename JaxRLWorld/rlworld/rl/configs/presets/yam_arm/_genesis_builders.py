"""Genesis builders for the YAM fixed-base arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs import RewardConfig, TerminationTermConfig
from rlworld.rl.configs.common_config_classes import (
    TerminationsConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    GenesisConfigsForRun,
    ObservationConfig,
    SceneConfig,
)
from rlworld.rl.configs.robots.yam import (
    YAM_ACTION_SCALE,
    YAM_ARMATURE,
    YAM_DAMPING,
    YAM_EFFORT_LIMIT,
    YAM_STIFFNESS,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    InitialStateCfg,
)
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed

from .base import build_rigid_objects

if TYPE_CHECKING:
    from .base import YamArmConfig

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


def build_visualization(cfg: YamArmConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False)


def build_env(cfg: YamArmConfig, timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        time_out = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        env_name="GenesisEnv",
        task_name="Yam_Arm",
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        decimation=timing["decimation"],
        episode_length_s=cfg.episode_length_s,
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: YamArmConfig, timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]

    return SceneConfig(
        entities={
            "robot": GenesisEntityCfg(
                # MJCF only. Genesis carries MuJoCo <equality> constraints
                # through the MJCF path alone, and the gripper's finger
                # coupling is one; the URDF path would decouple them.
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=cfg.base_pos,
                    joint_pos=r.default_joint_angles,
                ),
                # Genesis never forwards this on the MJCF path — it reads
                # the base joint from the file, and the XML's root body has
                # no free joint. Declared anyway so the entity states its
                # own base type rather than leaving it implicit.
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
                convexify=True,
                visualize_contact=False,
            ),
        },
        sim_options=gs.options.SimOptions(
            dt=sim_dt,
            substeps=timing["substeps"],
            gravity=(0.0, 0.0, -9.81),
        ),
        env_spacing=(2.0, 2.0),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            constraint_solver=gs.constraint_solver.Newton,
            constraint_timeconst=0.02,
            iterations=30,
            ls_iterations=40,
            enable_collision=True,
            # The arm's own links cannot reach each other in the poses this
            # preset visits, and self-collision is the dominant cost.
            enable_self_collision=False,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
            contact_pruning_tolerance=None,
            friction_cone=gs.friction_cone.pyramidal,
            impratio=1.0,
            integrator=gs.integrator.implicitfast,
            tolerance=1e-5,
        ),
        rigid_objects=build_rigid_objects(cfg),
        robot_cfg=r,
    )


def build_action(cfg: YamArmConfig) -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
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
