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
from rlworld.rl.configs.curriculums import CurriculumManagerConfig, CurriculumTermConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z, YamArmConfig
from rlworld.rl.configs.rewards.reward_term_config import RewardTermConfig
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.configs.terminations.termination_term_config import TerminationTermConfig
from rlworld.rl.envs.mdp.commands.lifting import LiftingCommandCfg
from rlworld.rl.envs.mdp.curriculums import reward_curriculum
from rlworld.rl.envs.mdp.observations.common import manipulation as manip_obs
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    dof_pos,
    dof_vel,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import manipulation as manip_rew
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    penalize_joint_pos_limits_l1,
    raw_action_rate_l2,
)
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common.terminations import illegal_contact

CUBE = "cube"
GRASP_SITE = "grasp_site"

CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF
"""Where a cube sits when resting on the table: 0.42."""

DROPPED_Z = TABLE_TOP_Z - 0.05
"""Below the table top by a margin — the cube has left the table and is
on its way to the floor. Not the floor itself: waiting for it to land
spends the rest of the episode rewarding an arm that hovers over
nothing."""

ARM_TABLE_CONTACT = "arm_table_contact"
"""Contact group name for the arm against its own work surface."""

TABLE_SLAM_N = 20.0
"""N — above this the arm is driving into the table rather than working
on it. Chosen against two measured numbers: the arm at rest registers
0 N, and pressing it down registers 158 N on Newton and far more on
mjlab, while the cube it manipulates weighs 0.5 N. This is the threshold
most likely to need adjusting once a policy is actually working near the
surface."""

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

    episode_length_s: float = 20.0
    """mjlab's own length for this task. Five seconds was mine, and it is
    barely enough to approach, close and lift once — a policy that has
    not yet learned any of those three needs room to stumble into them."""

    difficulty: str = "dynamic"
    """``"fixed"`` aims at one point every episode. Worth starting there:
    a policy that cannot solve a single goal will not solve a
    distribution of them, and the failure is far easier to read."""

    reaching_std: float = 0.20
    """m — the width of the reaching kernel, from mjlab. Wide on purpose:
    the reward has to be felt from where the hand STARTS, not only once
    it has arrived. At half this width the same starting gap pays 0.02
    instead of 0.38, and a policy facing that gradient does the thing
    that pays immediately instead — it stops moving, which is what the
    action-rate penalty rewards."""

    bringing_std: float = 0.30
    """m — the width of the bringing kernel inside the staged term.
    Wider than the precise one below: this half only has to point the
    way, while ``w_bring`` pays for actually arriving."""

    precise_std: float = 0.05
    """m — the standalone bringing term is narrow, which is what makes
    it a precision bonus rather than a second copy of the coarse one."""

    success_threshold: float = 0.05
    max_joint_vel: float = 0.5

    # Every weight here is POSITIVE. The penalty terms this repo uses
    # already return a negative value — ``raw_action_rate_l2`` and
    # ``penalize_joint_pos_limits_l1`` both end in a unary minus — so a
    # negative weight multiplies two of them together and pays the policy
    # for the very thing the term exists to discourage. mjlab's terms
    # return the positive quantity and take a negative weight; carrying
    # its numbers across unchanged is what flipped these.
    w_staged: float = 1.0
    w_bring: float = 1.0
    w_action_rate: float = 0.01
    w_joint_pos_limits: float = 10.0
    w_dropped: float = 5.0

    joint_vel_curriculum: tuple[tuple[int, float], ...] = (
        (0, 0.01),
        (500 * 24, 0.1),
        (1000 * 24, 1.0),
    )
    """Steps and weights for the joint-speed penalty, from mjlab's own
    lift config. It grows a hundredfold across training on purpose: at
    the final weight from the start the arm barely moves and learns
    nothing, and at the starting weight forever it stays violent."""

    def _sim_builders(self):
        return _get_sim_builders(self.sim_type)

    def _build_curriculum_config(self) -> CurriculumManagerConfig:
        """Ramp the joint-speed penalty, as mjlab does.

        Ported rather than dropped: mjlab put a hundredfold ramp on this
        one term, which is a statement that a fixed weight does not work
        for this task at either end.
        """
        stages = [{"step": step, "weight": weight} for step, weight in self.joint_vel_curriculum]

        @dataclass
        class _CurriculumCfg(CurriculumManagerConfig):
            joint_vel_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={"reward_name": "joint_vel", "stages": stages},
                )
            )

        return _CurriculumCfg()

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
                    # mjlab draws the cube from x(0.2, 0.4) y(+-0.2). Kept
                    # narrower here until the reach envelope has been
                    # measured over that wider patch — it was measured at
                    # the cube's nominal xy only, and a start pose the arm
                    # cannot reach is an episode lost before it begins.
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
            enable_corruption: bool = False

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
                params={"command_name": "lift", "object_name": CUBE, "std": cfg.precise_std},
            )
            action_rate = RewardTermConfig(func=raw_action_rate_l2, weight=cfg.w_action_rate)
            # Large, and in mjlab's set from the start. An arm learning to
            # reach finds the joint stops long before it finds the cube.
            joint_pos_limits = RewardTermConfig(
                func=penalize_joint_pos_limits_l1,
                weight=cfg.w_joint_pos_limits,
            )
            joint_vel = RewardTermConfig(
                func=manip_rew.joint_velocity_hinge_penalty,
                weight=cfg.joint_vel_curriculum[0][1],
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
            # One attempt per episode. Without this the arm solves the
            # task in about three seconds and then stands holding the
            # cube for the remaining seventeen, which teaches it almost
            # nothing; ending here spends that time on a fresh attempt
            # instead.
            lifted_to_goal = TerminationTermConfig(
                func=_object_at_goal,
                params={"command_name": "lift"},
                bootstrap_value=True,
            )
            cube_dropped = TerminationTermConfig(
                func=_cube_below,
                params={"object_name": CUBE, "min_height": DROPPED_Z},
            )
            # mjlab ends an episode when the end effector drives into the
            # ground. The table is this scene's ground: without this, a
            # policy can learn to reach the cube THROUGH the surface it
            # is standing on, which no real arm survives.
            table_slam = TerminationTermConfig(
                func=illegal_contact,
                params={"contact_group": ARM_TABLE_CONTACT, "force_threshold": TABLE_SLAM_N},
            )

        del cfg
        return _TerminationsCfg()

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if not self.run_name:
            runner.run_name = _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        return runner


def _object_at_goal(env, command_name: str) -> TerminationResult:
    """End the episode once the object has been brought to its goal.

    Bootstrapped, not absorbing. The reward here is dense and keeps
    paying while the object is held at the goal, so cutting the episode
    with a terminal value of zero would tell the policy that succeeding
    costs it everything it would have earned by standing still — and the
    surest way to avoid that is never to succeed. The value at the cut
    is the value of carrying on, which is what bootstrapping supplies.

    Ending here rather than running out the clock is what makes the
    episode one attempt: the next state the policy meets is a fresh
    reset, arm home and object placed, instead of seventeen seconds of
    holding followed by a teleport it cannot perceive.
    """
    return TerminationResult(env.command_manager.get_term(command_name).at_goal, is_timeout=False)


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
