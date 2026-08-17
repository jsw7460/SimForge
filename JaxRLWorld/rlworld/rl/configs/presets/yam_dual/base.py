"""Unified two-arm YAM config.

Everything sim-agnostic is inherited from :class:`YamArmConfig`; this
module adds the second arm, the placement of both, and the term-based
action space that splits the policy output between them.

Usage::

    from rlworld.rl.configs.presets.yam_dual.base import YamDualArmConfig
    cfgs = YamDualArmConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from rlworld.rl.configs.common_config_classes import (
    EventConfig,
    ObservationGroupConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.presets.yam_arm.base import (
    BASE_CLEARANCE,
    CUBE_HALF,
    TABLE_TOP_Z,
    YamArmConfig,
)
from rlworld.rl.configs.robots.yam import (
    YAM_ACTION_SCALE,
    YAM_ARM_JOINTS,
    YAM_GRIPPER_JOINT,
)
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.envs.mdp.actions.joint_actions import (
    JointPositionAction,
    JointPositionActionCfg,
)
from rlworld.rl.envs.mdp.events import common as common_ef
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    dof_pos,
    dof_vel,
    raw_actions,
)

RIGHT_ROBOT = "robot_right"
"""Scene name of the second arm. The first keeps the name ``robot``,
which is what ``robot_entity_name`` points at and therefore which arm
the single-robot shortcuts (``env.robot_data`` and friends) resolve to."""

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "YamDual_Newton",
    "genesis": "YamDual_Genesis",
    "mujoco": "YamDual_Mujoco",
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


def build_action_terms(cfg: YamDualArmConfig) -> Dict[str, JointPositionActionCfg]:
    """The three terms, identical on every backend.

    Split this way on purpose rather than one term per robot: an arm
    term and a gripper term on the *same* robot is the configuration
    that catches a term overwriting joints it does not own, and it is
    also how a real task wants to be written (the gripper usually wants
    its own scale, and often its own action space entirely).

    ``use_default_offset`` makes a zero action hold the home pose, so
    the two arms start the episode where the single-arm preset does.
    """
    return {
        "left_arm": JointPositionActionCfg(
            class_type=JointPositionAction,
            asset_name="robot",
            joint_names=list(YAM_ARM_JOINTS),
            scale={n: YAM_ACTION_SCALE[n] for n in YAM_ARM_JOINTS},
            use_default_offset=True,
            clip=(-100.0, 100.0),
        ),
        "left_gripper": JointPositionActionCfg(
            class_type=JointPositionAction,
            asset_name="robot",
            joint_names=[YAM_GRIPPER_JOINT],
            scale={YAM_GRIPPER_JOINT: YAM_ACTION_SCALE[YAM_GRIPPER_JOINT]},
            use_default_offset=True,
            clip=(-100.0, 100.0),
        ),
        "right_arm": JointPositionActionCfg(
            class_type=JointPositionAction,
            asset_name=RIGHT_ROBOT,
            joint_names=[*YAM_ARM_JOINTS, YAM_GRIPPER_JOINT],
            scale=dict(YAM_ACTION_SCALE),
            use_default_offset=True,
            clip=(-100.0, 100.0),
        ),
    }


@dataclass
class YamDualArmConfig(YamArmConfig):
    """Two bench-mounted YAM arms facing the same workpiece."""

    base_pos: tuple[float, float, float] = (0.0, -0.30, TABLE_TOP_Z + BASE_CLEARANCE)
    """Left arm, offset in -y so both arms fit on the bench."""

    right_base_pos: tuple[float, float, float] = (0.0, 0.30, TABLE_TOP_Z + BASE_CLEARANCE)
    """Right arm, mirrored in +y. The 0.6 m separation is wider than
    either arm's reach at bench height, so neither can drive into the
    other — which would otherwise make "the arm nobody commanded stayed
    still" fail for a reason that has nothing to do with the action
    layer."""

    cube_pos: tuple[float, float, float] = (0.35, 0.0, TABLE_TOP_Z + CUBE_HALF)
    """Between the two, where both can reach it."""

    def _sim_builders(self):
        return _get_sim_builders(self.sim_type)

    def _build_observation_config(self):
        """The single-arm observation, once per arm.

        One policy drives both arms from one action vector, so both arms'
        joint state belongs in the same group — the arms have to be
        coordinated, and a policy that cannot see one of them cannot
        coordinate it. Each term names the arm it reads through
        ``asset_cfg``, which is how IsaacLab scopes an observation to an
        asset, so splitting the group later is a matter of moving terms
        rather than rewriting them.
        """
        builders = self._sim_builders()
        ObsCfgClass = builders.OBSERVATION_CFG_CLS
        right = SceneEntitySelector(name=RIGHT_ROBOT)

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5))
            dof_pos_obs_right = ObservationTermConfig(
                func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01), params={"asset_cfg": right}
            )
            dof_vel_obs_right = ObservationTermConfig(
                func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5), params={"asset_cfg": right}
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
        """The single-arm events plus the second arm's placement and reset.

        Both arms need their own ``reset_joints_by_offset``: the term
        resets one entity, and the second arm is not reachable from the
        first one's selector.
        """
        cfg = super()._build_event_config()
        cfg.reset_root_right = EventTermConfig(
            func=common_ef.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "default_pos": self.right_base_pos,
                "asset_cfg": SceneEntitySelector(name=RIGHT_ROBOT),
            },
        )
        cfg.reset_dof_pos_right = EventTermConfig(
            func=common_ef.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": self.reset_joint_position_noise,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntitySelector(name=RIGHT_ROBOT),
            },
        )
        return cfg

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if not self.run_name:
            runner.run_name = _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        return runner
