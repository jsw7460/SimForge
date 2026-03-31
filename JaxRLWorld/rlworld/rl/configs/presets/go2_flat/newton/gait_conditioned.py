"""Go2 Newton config with gait-conditioned commands."""
from dataclasses import dataclass

from rlworld.rl.configs.common_config_classes import CommandConfig, GaitConfig
from rlworld.rl.envs.managers.common.command_term import (
    VelocityCommandTermCfg,
    GaitCommandTermCfg,
)
from rlworld.rl.envs.managers.common.gait import QuadrupedOffsets
from .base import Go2FlatNewtonConfig


@dataclass
class Go2GaitConditionedNewtonConfig(Go2FlatNewtonConfig):

    run_name: str = "Go2_GaitConditioned_Newton"

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
            foot_names=self.robot.prefixed_foot_names,
            offset_mode="command",
            freq_command="gait_freq",
            duration_command="gait_duration",
            foot_offset_provider=QuadrupedOffsets(),
        )
