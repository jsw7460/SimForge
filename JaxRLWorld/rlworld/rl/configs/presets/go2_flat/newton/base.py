from dataclasses import dataclass, field
from typing import Dict, Any, List

import math
import warp as wp

import newton
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig
from rlworld.rl.configs.components.observations.newton import LocomotionObservations
from rlworld.rl.configs.components.rewards.newton import TrackingRewards, RegularizationRewards
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.go2 import Go2Config
from rlworld.rl.configs.scene import NewtonEntityConfig
from rlworld.rl.configs.sensors import NewtonIMUSensorConfig, NewtonContactSensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.newton import terminations as tf


@dataclass
class Go2FlatNewtonConfig:
    """Base configuration for Go2 flat terrain locomotion on Newton simulator."""

    robot: Go2Config = field(default_factory=Go2Config)

    observations: LocomotionObservations = field(default_factory=lambda: LocomotionObservations(
                # Base linear velocity
                include_base_lin_vel=False,
                # base_lin_vel_scale=2.0,
                # base_lin_vel_noise=Unoise(-0.2, 0.2),
                # IMU angular velocity
                ang_vel_scale=0.25,
                ang_vel_noise=Unoise(-0.2, 0.2),
                # Projected gravity
                gravity_scale=1.0,
                gravity_noise=Unoise(-0.05, 0.05),
                # Command
                command_scale=1.0,
                # DOF position
                dof_pos_scale=1.0,
                dof_pos_noise=Unoise(-0.01, 0.01),
                include_dof_pos=True,
                include_nominal_difference=False,
                # DOF velocity
                dof_vel_scale=0.05,
                dof_vel_noise=Unoise(-1.5, 1.5),
                # Previous actions
                prev_actions_scale=1.0,
    ))

    tracking_rewards: TrackingRewards = field(default_factory=TrackingRewards)
    regularization_rewards: RegularizationRewards = field(default_factory=RegularizationRewards)

    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    dt: float = 0.02
    substeps: int = 4

    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)

    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Newton"

    def build(self) -> "NewtonConfigsForRun":
        """Build the complete configuration as a typed NewtonConfigsForRun."""
        from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)

        return NewtonConfigsForRun(
            env=self._build_env_config(quat),
            scene=self._build_scene_config(quat),
            visualization=VisualizationConfig(show_viewer=False, record_video=False),
            observation=self._build_observation_config(),
            action=self._build_action_config(),
            reward=self._build_reward_config(),
            command=self._build_command_config(),
            event=self._build_event_config(quat),
            algorithm=self._build_algorithm_config(),
            nn=self._build_nn_config(),
            runner=self._build_runner_config(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    def _build_env_config(self, quat) -> NewtonEnvConfig:
        return NewtonEnvConfig(
            num_envs=self.num_envs,
            task_name="Go2 Velocity Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            termination_criteria=[
                TerminationTermConfig(
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0}
                ),
                TerminationTermConfig(max_episode_exceed),
            ],
        )

    def _build_scene_config(self, quat) -> NewtonSceneConfig:
        r = self.robot
        return NewtonSceneConfig(
            dt=self.dt,
            substeps=self.substeps,
            gravity=(0.0, 0.0, -9.81),
            solver_type="mujoco",
            entities=[
                NewtonEntityConfig(
                    entity_name="ground",
                    entity_type="ground_plane",
                    floating=False
                ),
                NewtonEntityConfig(
                    entity_name="robot",
                    body_label_prefix=r.name,
                    urdf_path=r.urdf_path,
                    transform=wp.transform(
                        wp.vec3(0.0, 0.0, r.base_init_height),
                        quat
                    ),
                    floating=True,
                    joint_cfg=newton.ModelBuilder.JointDofConfig(
                        armature=0.1,
                        target_ke=20.0,
                        target_kd=0.5
                    ),
                    sites={"imu_site_base": r.base_link_name},
                    joint_target_ke_map=r.prefixed_p_gains,
                    joint_target_kd_map=r.prefixed_d_gains,
                    collapse_fixed_joints=True,
                    joints_to_keep=[
                        "go2_description/FL_foot_joint",
                        "go2_description/FR_foot_joint",
                        "go2_description/RL_foot_joint",
                        "go2_description/RR_foot_joint",
                    ]
                ),
            ],
            sensors=[
                NewtonIMUSensorConfig(
                    entity_name="robot",
                    sensor_name="imu_base",
                    site_names=["imu_site_base"]
                ),
                NewtonContactSensorConfig(
                    entity_name="robot",
                    sensor_name="contact",
                    sensing_obj_bodies=list(r.prefixed_foot_names),
                    use_regex=True
                )
            ],
            add_ground=True,
            env_spacing=(2.0, 2.0, 0.0),
            robot_cfg=self.robot
        )

    def _build_observation_config(self) -> NewtonObservationConfig:
        return NewtonObservationConfig(
            obs_group={
                "actor": self.observations.to_terms(),
                "critic": self.observations.to_critic_terms(),
            },
        )

    def _build_action_config(self) -> NewtonActionConfig:
        r = self.robot
        return NewtonActionConfig(
            actuated_dof_names=r.prefixed_actuated_dof_patterns,
            action_scale=0.4,
            clip_actions=(-100.0, 100.0),
            offset=r.get_prefixed_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        feet = list(r.prefixed_foot_names)
        base = r.prefixed("base")

        reward_terms = [
            RewardTermConfig(
                func=rf_mjlab.track_lin_vel_mjlab,
                weight=2.0,
                params={"std": 0.5},
            ),
            RewardTermConfig(
                func=rf_mjlab.track_ang_vel_mjlab,
                weight=2.0,
                params={"std": 0.707},
            ),
            RewardTermConfig(
                func=rf_mjlab.flat_orientation_mjlab,
                weight=1.0,
                params={"std": 0.447, "body_name": base},
            ),
            RewardTermConfig(
                func=rf_mjlab.variable_posture,
                weight=1.0,
                params={
                    "std_standing": {
                        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                        r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                    },
                    "std_walking": {
                        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                    },
                    "std_running": {
                        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                    },
                    "walking_threshold": 0.05,
                    "running_threshold": 1.5,
                },
            ),
            RewardTermConfig(
                func=rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_bodies": feet,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                func=rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_bodies": feet,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                func=rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_bodies": feet,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                func=rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_bodies": feet,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                func=rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
                params={"soft_limit_factor": 1.0},
            ),
            RewardTermConfig(
                func=rf_mjlab.processed_action_rate_l2_mjlab,
                weight=0.1,
            ),
        ]

        return RewardConfig(reward_terms)

    def _build_command_config(self) -> CommandConfig:
        return CommandConfig(
            resampling_time_s=(3.0, 8.0),
            sampler=[
                CommandTermConfig(cf.lin_vel_x, params={"range": self.lin_vel_x_range}),
                CommandTermConfig(cf.lin_vel_y, params={"range": self.lin_vel_y_range}),
                CommandTermConfig(cf.ang_vel, params={"range": self.ang_vel_range}),
            ],
            rel_standing_envs=0.1,
            heading_command=True,
            heading_control_stiffness=0.5,
            heading_range=(-3.14, 3.14),
            rel_heading_envs=0.3,
        )

    def _build_event_config(self, quat) -> EventConfig:
        r = self.robot
        from rlworld.rl.envs.mdp.events.newton_event_terms import push_robot
        return EventConfig([
            EventTermConfig(
                func=initf.initialize_base_pose,
                params={
                    "base_init_pos": [0.0, 0.0, r.base_init_height],
                    "base_init_quat": [quat[0], quat[1], quat[2], quat[3]],
                },
                mode="reset"
            ),
            EventTermConfig(
                func=initf.initialize_dof_pos_with_noise,
                params={"position_noise_range": (math.pi / 360, math.pi / 120)},
                mode="reset"
            ),
            EventTermConfig(
                func=initf.randomize_body_mass,
                params={"mass_ratio_range": (0.8, 1.2), "body_patterns": r.prefixed("base")},
                mode="reset",
            ),
            EventTermConfig(
                func=push_robot,
                mode="interval",
                interval_range_s=(2.0, 20.0),
                params={
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.4, 0.4),
                        "roll": (-0.52, 0.52),
                        "pitch": (-0.52, 0.52),
                        "yaw": (-0.78, 0.78),
                    },
                },
            )
        ])

    def _build_algorithm_config(self) -> PPOConfig:
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            obs_normalization=True,
            use_early_stop=False,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            actor_lr=1e-3,
            critic_lr=1e-3,
            estimator_learning_rate=5e-4,
            use_reward_scaling=False,
            max_grad_norm=0.5,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
            use_truth_value_for_actor=False,
            use_truth_value_for_critic=True,
            use_barrier_style=False,
            use_sde=True,
            sde_sample_freq=100,
            learning_starts=10_000,
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor_class_name=self.actor_class_name,
                actor_kwargs={
                    "activation": "elu",
                    "ortho_init": True,
                    "hidden_dims": self.actor_hidden_dims,
                },
                critic_kwargs={
                    "activation": "elu",
                    "ortho_init": True,
                    "hidden_dims": self.actor_hidden_dims,
                },
                init_noise_std=1.0,
                distribution_type="gaussian",
                std_type="state_independent",
            ),
        )

    def _build_runner_config(self) -> RunnerConfig:
        return RunnerConfig(
            checkpoint=-1,
            log_interval=1,
            max_iterations=self.max_iterations,
            init_at_random_ep_len=False,
            resume=False,
            resume_path=None,
            run_name=self.run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            save_interval=250,
            output_dir="auto",
        )
