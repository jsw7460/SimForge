"""Unified workcell config: bench-mounted arm + quadruped + workpiece.

Built on top of the single-arm preset, which already carries the solver
settings this gripper and bench need on each backend; this module adds
the quadruped, its action term, its share of the observation and its
reset.

The arm keeps the name ``robot`` and stays the driven entity — the arm's
scene is what is being extended, and the single-robot shortcuts
(``env.robot_data`` and friends) resolve to whatever is called ``robot``.
The quadruped is a second entity driven by its own action term, exactly
as the second arm is in ``yam_dual``.

Usage::

    from jaxrlworld.rl.configs.presets.lab_cell.base import LabCellConfig
    cfgs = LabCellConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from jaxrlworld.rl.configs.common_config_classes import (
    EventConfig,
    ObservationGroupConfig,
)
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.observations import ObservationTermConfig
from jaxrlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from jaxrlworld.rl.configs.presets.yam_arm.base import YamArmConfig
from jaxrlworld.rl.configs.robots.go2 import Go2Config
from jaxrlworld.rl.configs.robots.yam import YAM_ACTION_SCALE
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.envs.mdp.actions.joint_actions import (
    JointPositionAction,
    JointPositionActionCfg,
)
from jaxrlworld.rl.envs.mdp.events import common as common_ef
from jaxrlworld.rl.envs.mdp.observations.common.proprioception import (
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)

QUADRUPED = "quadruped"
"""Scene name of the legged robot. Not ``robot``: that name belongs to
the arm, whose scene this preset extends."""

# Action scale for the legs. The arm's per-joint scale comes from its own
# effort/stiffness ratio; the quadruped's action config uses a flat 0.25
# across all twelve joints, and a term needs the same number.
GO2_ACTION_SCALE = 0.25

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "LabCell_Newton",
    "genesis": "LabCell_Genesis",
    "mujoco": "LabCell_Mujoco",
}


def _get_sim_builders(sim_type: str):
    """Lazy-import the simulator-specific builders module."""
    if sim_type == "newton":
        from . import _newton_builders as mod
    elif sim_type == "genesis":
        from . import _genesis_builders as mod
    elif sim_type == "mujoco":
        from . import _mujoco_builders as mod
    else:
        raise ValueError(f"Unknown sim_type: {sim_type!r}.")
    return mod


def build_action_terms(cfg: LabCellConfig) -> Dict[str, JointPositionActionCfg]:
    """One term per robot, each naming the entity it drives.

    ``use_default_offset`` makes a zero action hold each robot's home
    pose, so an untouched scene shows both machines standing where the
    config put them rather than folding to zero.
    """
    return {
        "arm": JointPositionActionCfg(
            class_type=JointPositionAction,
            asset_name="robot",
            joint_names=list(cfg.robot.actuated_dof_patterns),
            scale=dict(YAM_ACTION_SCALE),
            use_default_offset=True,
            clip=(-100.0, 100.0),
        ),
        "legs": JointPositionActionCfg(
            class_type=JointPositionAction,
            asset_name=QUADRUPED,
            joint_names=list(cfg.quadruped.actuated_dof_patterns),
            scale=GO2_ACTION_SCALE,
            use_default_offset=True,
            clip=(-100.0, 100.0),
        ),
    }


@dataclass
class LabCellConfig(YamArmConfig):
    """An arm on a bench with a quadruped standing beside it."""

    quadruped: Go2Config = field(default_factory=Go2Config)

    quadruped_pos: tuple[float, float, float] = (-0.9, 0.0, 0.30)
    """Behind the bench's near edge, facing it. The bench body is 1.2 m
    long centred at x = 0.35, so it occupies x in [-0.25, 0.95]; a
    quadruped roughly 0.7 m long placed at -0.9 stands clear of it
    instead of spawning inside it."""

    def _sim_builders(self):
        return _get_sim_builders(self.sim_type)

    def _build_observation_config(self):
        """Both robots' joint state, plus the quadruped's orientation.

        One policy would drive both from one action vector, so both
        belong in one group. The quadruped additionally contributes
        ``projected_gravity``: it has a free base that can tip over,
        which is state no joint reading carries. The arm is welded, so
        the same term on the arm would be a constant.
        """
        builders = self._sim_builders()
        ObsCfgClass = builders.OBSERVATION_CFG_CLS
        legs = SceneEntitySelector(name=QUADRUPED)

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            arm_dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            arm_dof_vel = ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5))
            legs_dof_pos = ObservationTermConfig(
                func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01), params={"asset_cfg": legs}
            )
            legs_dof_vel = ObservationTermConfig(
                func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5), params={"asset_cfg": legs}
            )
            legs_gravity = ObservationTermConfig(
                func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05), params={"asset_cfg": legs}
            )
            prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            enable_corruption: bool = False

        @dataclass
        class _ObsCfg(ObsCfgClass):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def _build_event_config(self) -> EventConfig:
        """The arm's events plus the quadruped's placement and joint reset.

        The quadruped's root reset is a real placement, not the pinned
        restatement the welded arm gets: it has a free base, so where it
        starts is state rather than structure.
        """
        cfg = super()._build_event_config()
        cfg.reset_root_quadruped = EventTermConfig(
            func=common_ef.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "default_pos": self.quadruped_pos,
                "asset_cfg": SceneEntitySelector(name=QUADRUPED),
            },
        )
        cfg.reset_dof_pos_quadruped = EventTermConfig(
            func=common_ef.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": self.reset_joint_position_noise,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntitySelector(name=QUADRUPED),
            },
        )
        return cfg

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if not self.run_name:
            runner.run_name = _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        return runner
