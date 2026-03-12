from dataclasses import dataclass, field
from typing import Dict, Any, List

import warp as wp

import newton
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig
from rlworld.rl.configs.components.observations.newton import LocomotionObservations
from rlworld.rl.configs.components.rewards.newton import (
    TrackingRewards,
    RegularizationRewards
)
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig, G1_ACTION_SCALE
from rlworld.rl.configs.scene import NewtonEntityConfig
from rlworld.rl.configs.sensors import NewtonIMUSensorConfig, NewtonContactSensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations.newton import proprioception, state
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.newton import terminations as tf
from rlworld.rl.envs.mdp.events.newton_event_terms import push_robot
from rlworld.rl.configs.events.event_term_config import EventTermConfig


@dataclass
class G1FlatNewtonConfig:
    # Robot configuration
    robot: G1MjlabConfig = field(default_factory=G1MjlabConfig)

    # Observation component
    observations: LocomotionObservations | None = None
    extra_actor_observations: List[ObservationTermConfig] = field(default_factory=list)
    extra_critic_observations: List[ObservationTermConfig] = field(default_factory=list)

    # Action component
    action_scale: Dict[str, float] = field(default_factory=lambda: G1_ACTION_SCALE)

    # Reward components
    tracking_rewards: TrackingRewards = field(default_factory=lambda: TrackingRewards(
        tracking_lin_vel_weight=1.0,
        tracking_ang_vel_weight=1.0,
    ))
    regularization_rewards: RegularizationRewards = field(default_factory=lambda: RegularizationRewards(
        lin_vel_z_weight=0.2,
        base_height_weight=None,
        action_rate_weight=0.01,
        similar_to_default_weight=None,
    ))

    # Environment settings
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Simulation settings
    dt: float = 0.02
    substeps: int = 4

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-0.5, 0.5)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 30000

    actor_class_name: str = "MLPActor"
    run_name: str = "G1_29dof_Newton"

    robot_foot_names = None

    def __post_init__(self):
        if self.observations is None:
            self.observations = LocomotionObservations(
                # Base linear velocity (matching mjlab noise)
                base_lin_vel_scale=1.0,
                base_lin_vel_noise=Unoise(-0.5, 0.5),
                # IMU angular velocity
                ang_vel_scale=1.0,
                ang_vel_noise=Unoise(-0.2, 0.2),
                # Projected gravity
                gravity_scale=1.0,
                gravity_noise=Unoise(-0.05, 0.05),
                # Command
                command_scale=1.0,
                # DOF position (relative to default)
                dof_pos_scale=1.0,
                dof_pos_noise=Unoise(-0.01, 0.01),
                include_dof_pos=True,
                include_nominal_difference=False,
                # DOF velocity
                dof_vel_scale=1.0,
                dof_vel_noise=Unoise(-1.5, 1.5),
                # Previous actions
                prev_actions_scale=1.0,
            )

        # Extra observations for G1
        if not self.extra_actor_observations:
            self.extra_actor_observations = self._default_extra_observations()
        if not self.extra_critic_observations:
            self.extra_critic_observations = self._default_extra_critic_observations()

    def _default_extra_observations(self) -> List[ObservationTermConfig]:
        """G1-specific extra observations."""
        return []

    def _default_extra_critic_observations(self) -> List[ObservationTermConfig]:
        """Extra critic observations."""
        return [
            ObservationTermConfig(state.base_height, scale=1.0),
            ObservationTermConfig(state.base_lin_vel, scale=1.0),
            ObservationTermConfig(state.base_euler, scale=1.0),
            ObservationTermConfig(state.feet_air_time, scale=1.0, params={"feet_bodies": tuple(self.robot.prefixed_foot_names)}),
            ObservationTermConfig(state.feet_contact_force, scale=0.01, params={"feet_bodies": tuple(self.robot.prefixed_foot_names)}),
            ObservationTermConfig(state.feet_contact_indicator, scale=1.0, params={"feet_bodies": tuple(self.robot.prefixed_foot_names)}),
            ObservationTermConfig(state.feet_height, scale=1.0, params={"feet_bodies": tuple(self.robot.prefixed_foot_names)}),
        ]

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
            event=self._build_event_config(),
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
            env_name="NewtonLocomotionEnv",
            task_name="G1_12Dof_Velocity_Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            termination_criteria=[
                TerminationTermConfig(
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 70.0, "pitch_threshold_degree": 70.0}
                ),
                TerminationTermConfig(max_episode_exceed),
            ],
        )

    def _build_scene_config(self, quat) -> NewtonSceneConfig:
        return NewtonSceneConfig(
            dt=self.dt,
            substeps=self.substeps,
            gravity=(0.0, 0.0, -9.81),
            solver_type="mujoco",
            robot_cfg=self.robot,
            entities=[
                NewtonEntityConfig(
                    entity_name="ground",
                    body_label_prefix="ground",
                    entity_type="ground_plane",
                    shape_cfg=newton.ModelBuilder.ShapeConfig(
                        ke=2.0e3,
                        kd=1.0e2,
                        kf=1.0e3,
                        mu=1.0,
                        mu_rolling=0.0005,
                        mu_torsional=0.25,
                        # margin=0.00001,
                        # gap=0.0
                    ),
                    floating=False
                ),
                NewtonEntityConfig(
                    entity_name="robot",
                    body_label_prefix=self.robot.name,
                    entity_type="urdf",
                    urdf_path=self.robot.urdf_path,
                    transform=wp.transform(
                        wp.vec3(0.0, 0.0, self.robot.base_init_height),
                        quat
                    ),
                    floating=True,
                    joint_cfg=newton.ModelBuilder.JointDofConfig(
                        armature=0.1,
                        target_ke=400.0,
                        target_kd=5.0
                    ),
                    shape_cfg=newton.ModelBuilder.ShapeConfig(
                        ke=2.0e3,
                        kd=1.0e2,
                        kf=1.0e3,
                        mu=1.0,
                        # gap=0.0
                    ),
                    joint_target_ke_map=self.robot.prefixed_p_gains,
                    joint_target_kd_map=self.robot.prefixed_d_gains,
                    joint_armature_map=self.robot.prefixed_armature,
                    sites={"imu_site_base": self.robot.base_link_name},
                    enable_self_collisions=False,
                    collapse_fixed_joints=True
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
                    sensor_name="foot_contact",
                    sensing_obj_bodies=self.robot.foot_names,
                    counterpart_shapes="ground_plane",
                    use_regex=True,
                    include_total=False
                )
            ],
            add_ground=True,
            env_spacing=(2.0, 2.0, 0.0),
        )

    def _build_observation_config(self) -> NewtonObservationConfig:
        return NewtonObservationConfig(
            obs_group={
                "actor": self.observations.to_terms() + self.extra_actor_observations,
                "critic": self.observations.to_critic_terms() + self.extra_critic_observations
            },
        )

    def _build_action_config(self) -> NewtonActionConfig:
        return NewtonActionConfig(
            actuated_dof_names=self.robot.prefixed_actuated_dof_patterns,
            action_scale=self.robot.prefixed_action_scale,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_prefixed_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        reward_terms = self._mjlab_default_extra_rewards()
        return RewardConfig(reward_terms=reward_terms)

    def _mjlab_default_extra_rewards(self) -> dict[str, RewardTermConfig]:
        """G1-specific reward terms."""
        return {
            # Tracking rewards
            "track_lin_vel_mjlab": RewardTermConfig(
                rf_mjlab.track_lin_vel_mjlab,
                weight=2.0,
                params={"std": 0.5},  # sqrt(0.25)
            ),
            "track_ang_vel_mjlab": RewardTermConfig(
                rf_mjlab.track_ang_vel_mjlab,
                weight=2.0,
                params={"std": 0.707},  # sqrt(0.5)
            ),

            # Orientation
            "flat_orientation_mjlab": RewardTermConfig(
                rf_mjlab.flat_orientation_mjlab,
                weight=1.0,
                params={"std": 0.447, "body_name": self.robot.prefixed("torso_link")},  # sqrt(0.2)
            ),

            # Posture (stateful class)
            "variable_posture": RewardTermConfig(
                rf_mjlab.variable_posture,
                weight=1.0,
                params={
                    "std_standing": {".*": 0.05},
                    "std_walking": {
                        r".*hip_pitch.*": 0.3,
                        r".*hip_roll.*": 0.15,
                        r".*hip_yaw.*": 0.15,
                        r".*knee.*": 0.35,
                        r".*ankle_pitch.*": 0.25,
                        r".*ankle_roll.*": 0.1,
                        # Waist.
                        r".*waist_yaw.*": 0.2,
                        r".*waist_roll.*": 0.08,
                        r".*waist_pitch.*": 0.1,
                        # Arms.
                        r".*shoulder_pitch.*": 0.15,
                        r".*shoulder_roll.*": 0.15,
                        r".*shoulder_yaw.*": 0.1,
                        r".*elbow.*": 0.15,
                        r".*wrist.*": 0.3,
                    },
                    "std_running": {
                        # Lower body.
                        r".*hip_pitch.*": 0.5,
                        r".*hip_roll.*": 0.2,
                        r".*hip_yaw.*": 0.2,
                        r".*knee.*": 0.6,
                        r".*ankle_pitch.*": 0.35,
                        r".*ankle_roll.*": 0.15,
                        # Waist.
                        r".*waist_yaw.*": 0.3,
                        r".*waist_roll.*": 0.08,
                        r".*waist_pitch.*": 0.2,
                        # Arms.
                        r".*shoulder_pitch.*": 0.5,
                        r".*shoulder_roll.*": 0.2,
                        r".*shoulder_yaw.*": 0.15,
                        r".*elbow.*": 0.35,
                        r".*wrist.*": 0.3,
                    },
                    "walking_threshold": 0.05,
                    "running_threshold": 1.5,
                },
            ),

            # Penalties
            "body_ang_vel_penalty_mjlab": RewardTermConfig(
                rf_mjlab.body_ang_vel_penalty_mjlab,
                weight=0.05,
                params={"body_name": self.robot.prefixed("torso_link")},
            ),
            "angular_momentum_penalty": RewardTermConfig(
                rf_mjlab.angular_momentum_penalty,
                weight=0.02,
            ),
            "joint_pos_limits_mjlab": RewardTermConfig(
                rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
            ),
            "raw_action_rate_l2_mjlab": RewardTermConfig(
                rf_mjlab.raw_action_rate_l2_mjlab,
                weight=0.1,
            ),

            # Feet rewards
            "feet_clearance_mjlab": RewardTermConfig(
                rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            "feet_swing_height_mjlab": RewardTermConfig(
                rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            "feet_slip_mjlab": RewardTermConfig(
                rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "command_threshold": 0.05,
                },
            ),
            "soft_landing_mjlab": RewardTermConfig(
                rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "command_threshold": 0.05,
                },
            ),

            # Air time (weight=0 in mjlab config, included for completeness)
            # "feet_air_time_mjlab": RewardTermConfig(
            #     rf_mjlab.feet_air_time_mjlab,
            #     weight=0.0,
            #     params={
            #         "feet_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
            #         "threshold_min": 0.05,
            #         "threshold_max": 0.5,
            #         "command_threshold": 0.5,
            #     },
            # ),
        }

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

    def _build_event_config(self) -> EventConfig:
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
        return EventConfig(event_terms=[
            EventTermConfig(
                func=initf.initialize_base_pose,
                mode="reset",
                params={
                    "base_init_pos": [0.0, 0.0, self.robot.base_init_height],
                    "base_init_quat": [quat[0], quat[1], quat[2], quat[3]],
                }
            ),
            EventTermConfig(
                func=initf.initialize_dof_pos,
                mode="reset"
            ),
            EventTermConfig(
                func=push_robot,
                mode="interval",
                interval_range_s=(1.0, 3.0),
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
            ),
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
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
            use_reward_scaling=False,
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor_class_name=self.actor_class_name,
                actor_kwargs={
                    "activation": "tanh",
                    "ortho_init": True,
                    "hidden_dims": [512, 256, 128],
                },
                critic_kwargs={
                    "activation": "tanh",
                    "ortho_init": True,
                    "hidden_dims": [1024, 512, 256],
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
            init_at_random_ep_len=True,
            resume=False,
            resume_path=None,
            run_name=self.run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            save_interval=250,
            output_dir="auto",
        )
