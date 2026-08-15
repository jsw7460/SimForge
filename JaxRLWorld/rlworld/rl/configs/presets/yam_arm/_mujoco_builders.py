"""MuJoCo (mjlab) builders for the YAM fixed-base arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import mujoco
from mjlab.utils.spec_config import CollisionCfg

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs import RewardConfig, TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
    VisualizationConfig,
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
    InitialStateCfg,
    MujocoEntityCfg,
)
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed

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


# The six spheres per finger are the surfaces that actually hold a
# workpiece, so they get grippy, torsion-aware contact; every other geom
# only needs to stop a link passing through something.
_FINGER_PADS = "[lr]f_down(6|7|8|9|10|11)_collision"
_ALL_COLLISION = ".*_collision"

GRIPPER_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_ALL_COLLISION,),
    # Only the wrist and the fingers collide. The upper links have
    # nothing to reach in this preset, and every enabled pair costs.
    contype={"(link6|[lr]f)_.*_collision": 1, _ALL_COLLISION: 0},
    conaffinity={"(link6|[lr]f)_.*_collision": 1, _ALL_COLLISION: 0},
    condim={_FINGER_PADS: 6, _ALL_COLLISION: 3},
    friction={_FINGER_PADS: (1.0, 5e-3, 5e-4), _ALL_COLLISION: (0.6,)},
    solref={_FINGER_PADS: (0.01, 1.0)},
    priority={_FINGER_PADS: 1, ".*": 0},
)


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
        collisions=(GRIPPER_ONLY_COLLISION,),
    )

    return MujocoSceneConfig(
        physics_dt=timing["dt"],
        substeps=timing.get("substeps", 1),
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        entities={"robot": robot_entity},
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        # The arm's own limit + contact constraints overflow a small
        # buffer; an overflow silently DROPS constraints rather than
        # erroring, so joints escape their limits.
        njmax=400,
        impratio=10.0,
        cone="elliptic",
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
