"""Unified lift-the-cube config, built on the single-arm preset.

The scene is the arm preset's: an arm bolted to a table, with a cube on
it. What this adds is the task — a goal to bring the cube to, rewards
that lead there, and observations that describe the gap in a frame that
moves with the robot.

Usage::

    from rlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
    cfgs = YamLiftConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from rlworld.rl.configs.common_config_classes import (
    CommandConfig,
    ObservationGroupConfig,
    RewardConfig,
    TerminationsConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z, YamArmConfig
from rlworld.rl.configs.rewards.reward_term_config import RewardTermConfig
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.configs.terminations.termination_term_config import TerminationTermConfig
from rlworld.rl.envs.mdp.commands.lifting import LiftingCommandCfg
from rlworld.rl.envs.mdp.observations.common import manipulation as manip_obs
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    dof_pos,
    dof_vel,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import manipulation as manip_rew
from rlworld.rl.envs.mdp.rewards.common.reward_terms import raw_action_rate_l2
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed

CUBE = "cube"
GRASP_SITE = "grasp_site"

CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF
"""Where a cube sits when resting on the table: 0.42."""

DROPPED_Z = TABLE_TOP_Z - 0.05
"""Below the table top by a margin — the cube has left the table and is
on its way to the floor. Not the floor itself: waiting for it to land
spends the rest of the episode rewarding an arm that hovers over
nothing."""

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "YamLift_Newton",
    "genesis": "YamLift_Genesis",
    "mujoco": "YamLift_Mujoco",
}


def _get_sim_builders(sim_type: str):
    if sim_type == "newton":
        from . import _newton_builders as mod
    elif sim_type == "genesis":
        from . import _genesis_builders as mod
    elif sim_type == "mujoco":
        from . import _mujoco_builders as mod
    else:
        raise ValueError(f"Unknown sim_type: {sim_type!r}.")
    return mod


@dataclass
class YamLiftConfig(YamArmConfig):
    """Bring the cube from the table to a commanded point above it."""

    episode_length_s: float = 5.0

    difficulty: str = "dynamic"
    """``"fixed"`` aims at one point every episode. Worth starting there:
    a policy that cannot solve a single goal will not solve a
    distribution of them, and the failure is far easier to read."""

    reaching_std: float = 0.10
    """m — the width of the reaching kernel. About the size of the
    gripper, so the reward starts to rise when the hand is in the
    neighbourhood rather than only once it has arrived."""

    bringing_std: float = 0.15
    success_threshold: float = 0.05
    max_joint_vel: float = 8.0

    w_staged: float = 4.0
    w_bring: float = 2.0
    w_action_rate: float = -0.01
    w_joint_vel: float = -0.001
    w_dropped: float = -5.0

    def _sim_builders(self):
        return _get_sim_builders(self.sim_type)

    # ── Task ─────────────────────────────────────────────────────────

    def _build_command_config(self) -> CommandConfig:
        """The goal, and where the cube starts.

        Both ranges were measured on this scene rather than carried over:
        the goal box is reachable at every corner, and the cube's start
        band sits on the table.
        """
        return CommandConfig(
            terms={
                "lift": LiftingCommandCfg(
                    entity_name=CUBE,
                    difficulty=self.difficulty,
                    success_threshold=self.success_threshold,
                    fixed_target=(0.40, 0.0, 0.70),
                    target_x=(0.30, 0.50),
                    target_y=(-0.20, 0.20),
                    target_z=(0.60, 0.80),
                    object_x=(0.30, 0.35),
                    object_y=(-0.10, 0.10),
                    object_z=(CUBE_REST_Z, CUBE_REST_Z + 0.03),
                )
            }
        )

    def _build_observation_config(self):
        """Joint state, and the task expressed as relative vectors.

        No world positions: a policy handed them learns where this table
        is, not what the task is. Every task term below is a gap — hand
        to object, object to goal — in a frame that travels with the
        robot.
        """
        builders = self._sim_builders()
        ObsCfgClass = builders.OBSERVATION_CFG_CLS
        grasp = SceneEntitySelector(name="robot", site_names=(GRASP_SITE,))

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5))
            ee_to_cube = ObservationTermConfig(
                func=manip_obs.ee_to_object_distance,
                scale=1.0,
                params={"object_name": CUBE, "asset_cfg": grasp},
            )
            cube_to_goal = ObservationTermConfig(
                func=manip_obs.object_to_goal_distance,
                scale=1.0,
                params={"object_name": CUBE, "command_name": "lift"},
            )
            goal_from_ee = ObservationTermConfig(
                func=manip_obs.target_position,
                scale=1.0,
                params={"command_name": "lift", "asset_cfg": grasp},
            )
            ee_vel = ObservationTermConfig(func=manip_obs.ee_velocity, scale=0.1, params={"asset_cfg": grasp})
            cube_height = ObservationTermConfig(
                func=manip_obs.object_height,
                scale=1.0,
                params={"object_name": CUBE, "reference_height": TABLE_TOP_Z},
            )
            prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            enable_corruption = False

        @dataclass
        class _ObsCfg(ObsCfgClass):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def build_rewards(self) -> RewardConfig:
        """The staged reward, plus the costs that keep it from being gamed."""
        grasp = SceneEntitySelector(name="robot", site_names=(GRASP_SITE,))
        cfg = self

        @dataclass
        class _RewardsCfg(RewardConfig):
            # Reaching x (1 + bringing). The product is the whole design:
            # a sum would let the policy collect the reaching half forever
            # without ever picking the cube up.
            staged = RewardTermConfig(
                func=manip_rew.staged_position_reward,
                weight=cfg.w_staged,
                params={
                    "command_name": "lift",
                    "object_name": CUBE,
                    "reaching_std": cfg.reaching_std,
                    "bringing_std": cfg.bringing_std,
                    "asset_cfg": grasp,
                },
            )
            bring = RewardTermConfig(
                func=manip_rew.bring_object_reward,
                weight=cfg.w_bring,
                params={"command_name": "lift", "object_name": CUBE, "std": cfg.bringing_std},
            )
            action_rate = RewardTermConfig(func=raw_action_rate_l2, weight=cfg.w_action_rate)
            joint_vel = RewardTermConfig(
                func=manip_rew.joint_velocity_hinge_penalty,
                weight=cfg.w_joint_vel,
                params={"max_vel": cfg.max_joint_vel},
            )
            # Knocking the cube off the table ends the episode, but the
            # penalty is what stops a policy from learning to sweep it away
            # and then hover for the reaching reward.
            dropped = RewardTermConfig(
                func=manip_rew.object_dropped,
                weight=cfg.w_dropped,
                params={"object_name": CUBE, "min_height": DROPPED_Z},
            )

        return _RewardsCfg()

    def build_terminations(self) -> TerminationsConfig:
        """Time out, or lose the cube off the table."""
        cfg = self

        @dataclass
        class _TerminationsCfg(TerminationsConfig):
            time_out = TerminationTermConfig(max_episode_exceed)
            cube_dropped = TerminationTermConfig(
                func=_cube_below,
                params={"object_name": CUBE, "min_height": DROPPED_Z},
            )

        del cfg
        return _TerminationsCfg()

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if not self.run_name:
            runner.run_name = _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        return runner


def _cube_below(env, object_name: str, min_height: float) -> TerminationResult:
    """Terminate where the object has fallen below ``min_height``.

    A termination rather than a reward shape: once the cube is off the
    table nothing the arm does can bring it back, and every further step
    samples a state the task never meant to visit.

    Not a timeout — the episode ended badly, so its terminal value is
    zero rather than bootstrapped.
    """
    fallen = env.get_entity_data(object_name).root_link_pos_w[:, 2] < min_height
    return TerminationResult(fallen, is_timeout=False)
