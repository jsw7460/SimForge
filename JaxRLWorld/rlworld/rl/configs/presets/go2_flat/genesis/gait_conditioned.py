"""Go2 Genesis config with gait-conditioned commands.

Extends the base Go2 flat config with GaitCommandTerm and observation
structure matching Walk-These-Ways (Margolis & Agrawal, CoRL 2022).

Observation (69 dim):
    projected_gravity       (3)
    all_commands * scale    (14)  velocity(3) + gait(11)
    dof_pos - default       (12)
    dof_vel                 (12)
    actions                 (12)  current step
    last_actions            (12)  previous step
    clock_inputs            (4)   sin gait phase per foot
"""
from dataclasses import dataclass, field

from rlworld.rl.configs.common_config_classes import (
    CommandConfig, GaitConfig, ObservationGroupConfig, RewardConfig,
)
from rlworld.rl.configs.genesis_config_classes import ObservationConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.managers.common.command_term import (
    VelocityCommandTermCfg,
    GaitCommandTermCfg,
)
from rlworld.rl.envs.managers.common.gait import QuadrupedOffsets
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    projected_gravity,
    dof_pos_nominal_difference,
    dof_vel,
    prev_raw_actions,
    raw_actions,
    clock_inputs,
    all_commands,
    base_lin_vel,
    base_height,
)
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf_genesis
from .base import Go2FlatGenesisConfig


@dataclass
class Go2GaitConditionedGenesisConfig(Go2FlatGenesisConfig):

    run_name: str = "Go2_GaitConditioned_Genesis"

    def _build_command_config(self) -> CommandConfig:
        return CommandConfig(
            terms={
                "velocity": VelocityCommandTermCfg(
                    resampling_time_range=(10.0, 10.0),
                    lin_vel_x_range=self.lin_vel_x_range,
                    lin_vel_y_range=(-0.6, 0.6),
                    ang_vel_range=self.ang_vel_range,
                    rel_standing_envs=0.0,
                    heading_command=False,
                ),
                "gait": GaitCommandTermCfg(
                    resampling_time_range=(10.0, 10.0),
                    freq_range=(2.0, 4.0),
                    phase_range=(0.0, 1.0),
                    offset_range=(0.0, 1.0),
                    bound_range=(0.0, 1.0),
                    duration_range=(0.5, 0.5),
                    footswing_height_range=(0.03, 0.35),
                    body_height_range=(-0.25, 0.15),
                    body_pitch_range=(-0.4, 0.4),
                    body_roll_range=(0.0, 0.0),
                    stance_width_range=(0.10, 0.45),
                    stance_length_range=(0.35, 0.45),
                    gait_category_mode="gaitwise",
                    binary_phases=True,
                ),
            }
        )

    def _build_gait_config(self) -> GaitConfig:
        return GaitConfig(
            foot_names=self.robot.foot_names,
            offset_mode="command",
            freq_command="gait_freq",
            duration_command="gait_duration",
            foot_offset_provider=QuadrupedOffsets(foot_names=self.robot.foot_names),
        )

    def _build_reward_config(self) -> RewardConfig:
        @dataclass
        class _WTWRewardsCfg(RewardConfig):
            exponential_shaping: bool = True
            shaping_sigma: float = 0.02

            # ── Tracking (common) ──
            track_lin_vel = RewardTermConfig(
                func=rf_common.track_lin_vel, weight=1.0,
                params={"std": 0.5},
            )
            track_ang_vel = RewardTermConfig(
                func=rf_common.track_ang_vel, weight=0.5,
                params={"std": 0.5},
            )

            # ── Gait tracking (Genesis-specific) ──
            tracking_contacts_shaped_force = RewardTermConfig(
                func=rf_genesis.wtw_tracking_contacts_shaped_force, weight=4.0,
                params={"gait_force_sigma": 100.0},
            )
            tracking_contacts_shaped_vel = RewardTermConfig(
                func=rf_genesis.wtw_tracking_contacts_shaped_vel, weight=4.0,
                params={"gait_vel_sigma": 10.0},
            )

            # ── Body commands (common) ──
            body_height_cmd = RewardTermConfig(
                func=rf_common.reward_body_height_cmd, weight=10.0,
                params={"base_height_target": 0.30},
            )
            orientation_control = RewardTermConfig(
                func=rf_common.penalize_orientation_control, weight=5.0,
            )

            # ── Footstep placement (Genesis-specific) ──
            raibert_heuristic = RewardTermConfig(
                func=rf_genesis.wtw_raibert_heuristic, weight=10.0,
            )
            feet_clearance_cmd_linear = RewardTermConfig(
                func=rf_genesis.wtw_feet_clearance_cmd_linear, weight=30.0,
            )

            # ── Regularization (common) ──
            feet_slip = RewardTermConfig(
                func=rf_genesis.wtw_feet_slip, weight=0.04,
            )
            action_smoothness_1 = RewardTermConfig(
                func=rf_common.penalize_action_smoothness_1, weight=0.1,
            )
            action_smoothness_2 = RewardTermConfig(
                func=rf_common.penalize_action_smoothness_2, weight=0.1,
            )
            dof_vel = RewardTermConfig(
                func=rf_common.penalize_dof_vel, weight=1e-4,
            )
            lin_vel_z = RewardTermConfig(
                func=rf_common.penalize_lin_vel_z, weight=0.02,
            )
            ang_vel_xy = RewardTermConfig(
                func=rf_common.penalize_ang_vel_xy, weight=0.001,
            )

            # ── Collision (Genesis-specific, reuse existing) ──
            collision = RewardTermConfig(
                func=rf_genesis.penalize_invalid_contact, weight=5.0,
                params={"contact_allowed_links": self.robot.foot_names, "exclude_self_contact": False},
            )

        return _WTWRewardsCfg()

    def _build_observation_config(self) -> ObservationConfig:
        # WTW observation order:
        #   gravity(3), commands(14)*scale, dof_pos(12), dof_vel(12),
        #   actions(12), last_actions(12), clock_inputs(4) = 69
        #
        # Command scales from WTW obs_scales:
        #   velocity: [lin_vel=2.0, lin_vel=2.0, ang_vel=0.25]
        #   gait: [freq=1.0, phase=1.0, offset=1.0, bound=1.0, duration=1.0,
        #          footswing=0.15, body_height=2.0, pitch=0.3, roll=0.3,
        #          stance_width=1.0, stance_length=1.0]
        #
        # Since commands have per-column scales, we apply scale=1.0 here
        # and handle per-column scaling inside the command observation or
        # via the commands_scale vector in the WTW training pipeline.
        # For now, scale=1.0 (raw commands) — matching WTW's approach
        # where commands_scale is applied during obs construction.

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            projected_gravity = ObservationTermConfig(
                func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05),
            )
            commands = ObservationTermConfig(func=all_commands, scale=1.0)
            dof_pos = ObservationTermConfig(
                func=dof_pos_nominal_difference, scale=1.0, noise=Unoise(-0.01, 0.01),
            )
            dof_vel = ObservationTermConfig(
                func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5),
            )
            actions = ObservationTermConfig(func=raw_actions, scale=1.0)
            last_actions = ObservationTermConfig(func=prev_raw_actions, scale=1.0)
            clock = ObservationTermConfig(func=clock_inputs, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=2.0)
            base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)

        @dataclass
        class _ObsCfg(ObservationConfig):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()


def get_config():
    return Go2GaitConditionedGenesisConfig().build()
