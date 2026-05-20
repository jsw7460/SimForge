"""Go2 MuJoCo config with gait-conditioned commands.

Observation structure matching Walk-These-Ways (69 dim).
See genesis/gait_conditioned.py for detailed documentation.
"""

from dataclasses import dataclass, field

from rlworld.rl.configs.common_config_classes import CommandConfig, GaitConfig, ObservationGroupConfig, RewardConfig
from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun, MujocoObservationConfig as ObservationConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.managers.common.command_term import (
    GaitCommandTermCfg,
    VelocityCommandTermCfg,
)
from rlworld.rl.envs.managers.common.gait import QuadrupedOffsets
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    all_commands,
    base_height,
    base_lin_vel,
    clock_inputs,
    dof_pos_nominal_difference,
    dof_vel,
    prev_raw_actions,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common, wtw as rf_wtw
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf_mujoco


@dataclass
class Go2GaitConditionedMujocoConfig(Go2FlatConfig):
    sim_type: str = "mujoco"
    run_name: str = "Go2_GaitConditioned_Mujoco"

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
        # site_names is resolved by mjlab via find_sites(...,
        # preserve_order=...). With preserve_order=False the result
        # follows go2.xml's site declaration order (FL, FR, RL, RR),
        # not the tuple below. Reward functions that touch site_ids
        # alongside gait_manager state would then silently pair the
        # wrong legs. Use preserve_order=True so the resolved
        # ``site_names`` / ``site_ids`` honour this tuple, and the
        # per-reward _gait_aligned_site_indices helper still reorders
        # to gait_manager order regardless.
        site_names = ("FR", "FL", "RR", "RL")
        foot_asset_cfg = SceneEntitySelector(
            name="robot",
            site_names=site_names,
            preserve_order=False,
        )

        @dataclass
        class _WTWRewardsCfg(RewardConfig):
            reward_mode: str = "exponential_auto"
            shaping_sigma: float = 0.02

            track_lin_vel = RewardTermConfig(
                func=rf_common.track_lin_vel,
                weight=1.0,
                params={"std": 0.5},
            )
            track_ang_vel = RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=0.5,
                params={"std": 0.5},
            )
            tracking_contacts_shaped_force = RewardTermConfig(
                func=rf_mujoco.wtw_tracking_contacts_shaped_force,
                weight=4.0,
                params={"contact_group": "feet_ground_contact", "gait_force_sigma": 100.0},
            )
            tracking_contacts_shaped_vel = RewardTermConfig(
                func=rf_mujoco.wtw_tracking_contacts_shaped_vel,
                weight=4.0,
                params={"gait_vel_sigma": 10.0, "asset_cfg": foot_asset_cfg},
            )
            body_height_cmd = RewardTermConfig(
                func=rf_wtw.reward_body_height_cmd,
                weight=10.0,
                params={"base_height_target": 0.30},
            )
            orientation_control = RewardTermConfig(
                func=rf_wtw.penalize_orientation_control,
                weight=5.0,
            )
            raibert_heuristic = RewardTermConfig(
                func=rf_mujoco.wtw_raibert_heuristic,
                weight=10.0,
                params={"asset_cfg": foot_asset_cfg},
            )
            feet_clearance_cmd_linear = RewardTermConfig(
                func=rf_mujoco.wtw_feet_clearance_cmd_linear,
                weight=30.0,
                params={"asset_cfg": foot_asset_cfg},
            )
            feet_slip = RewardTermConfig(
                func=rf_mujoco.wtw_feet_slip,
                weight=0.04,
                params={"contact_group": "feet_ground_contact", "asset_cfg": foot_asset_cfg},
            )
            action_smoothness_1 = RewardTermConfig(
                func=rf_wtw.penalize_action_smoothness_1,
                weight=0.1,
            )
            action_smoothness_2 = RewardTermConfig(
                func=rf_wtw.penalize_action_smoothness_2,
                weight=0.1,
            )
            dof_vel = RewardTermConfig(
                func=rf_common.penalize_dof_vel,
                weight=1e-4,
            )
            lin_vel_z = RewardTermConfig(
                func=rf_common.penalize_lin_vel_z,
                weight=0.02,
            )
            ang_vel_xy = RewardTermConfig(
                func=rf_common.penalize_ang_vel_xy,
                weight=0.001,
            )
            collision = RewardTermConfig(
                func=rf_mujoco.wtw_collision,
                weight=5.0,
                params={"contact_group": "body_ground_contact", "force_threshold": 10.0},
            )

        return _WTWRewardsCfg()

    def _build_observation_config(self) -> ObservationConfig:
        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            projected_gravity = ObservationTermConfig(
                func=projected_gravity,
                scale=1.0,
                noise=Unoise(-0.05, 0.05),
            )
            commands = ObservationTermConfig(func=all_commands, scale=1.0)
            dof_pos = ObservationTermConfig(
                func=dof_pos_nominal_difference,
                scale=1.0,
                noise=Unoise(-0.01, 0.01),
            )
            dof_vel = ObservationTermConfig(
                func=dof_vel,
                scale=0.05,
                noise=Unoise(-1.5, 1.5),
            )
            actions = ObservationTermConfig(func=raw_actions, scale=1.0)
            last_actions = ObservationTermConfig(func=prev_raw_actions, scale=1.0)
            clock = ObservationTermConfig(func=clock_inputs, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            enable_corruption = False
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=2.0)
            base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)

        @dataclass
        class _ObsCfg(ObservationConfig):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def build(self) -> MujocoConfigsForRun:
        return super().build()


def get_config() -> MujocoConfigsForRun:
    return Go2GaitConditionedMujocoConfig().build()
