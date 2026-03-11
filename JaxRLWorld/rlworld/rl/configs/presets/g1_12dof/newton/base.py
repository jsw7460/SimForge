from dataclasses import dataclass, field
from typing import Dict, Any, List

import math
import warp as wp

import newton
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig
from rlworld.rl.configs.components.observations.newton import LocomotionObservations
from rlworld.rl.configs.components.rewards.newton import (
    TrackingRewards,
    RegularizationRewards,
    ContactRewards
)
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_12dof import G1Config
from rlworld.rl.configs.scene import NewtonEntityConfig
from rlworld.rl.configs.sensors import NewtonIMUSensorConfig, NewtonContactSensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    StateInitializationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations.newton import proprioception, state
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.newton import terminations as tf


@dataclass
class G1FlatNewtonConfig:
    # Robot configuration
    robot: G1Config = field(default_factory=G1Config)

    # Observation component
    observations: LocomotionObservations | None = None
    extra_actor_observations: List[ObservationTermConfig] = field(default_factory=list)
    extra_critic_observations: List[ObservationTermConfig] = field(default_factory=list)

    # Reward components
    tracking_rewards: TrackingRewards = field(default_factory=lambda: TrackingRewards(
        tracking_lin_vel_weight=1.0,
        tracking_ang_vel_weight=0.5,
    ))
    regularization_rewards: RegularizationRewards = field(default_factory=lambda: RegularizationRewards(
        lin_vel_z_weight=2.0,
        base_height_weight=None,
        action_rate_weight=0.01,
        similar_to_default_weight=0.1,
    ))

    # Environment settings
    num_envs: int = 8192
    episode_length_s: float = 30.0
    seed: int = 42

    # Simulation settings
    dt: float = 0.02        # Control dt
    substeps: int = 4       # = Decimation

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)
    base_height_range: tuple = (0.72, 0.72)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])

    actor_class_name: str = "MLPActor"
    run_name: str = "G1_12dof_Newton"

    feet_height_target: float = 0.1

    def __post_init__(self):
        if self.observations is None:
            self.observations = LocomotionObservations(
                # IMU angular velocity
                ang_vel_scale=2.0,
                ang_vel_history=4,
                # Projected gravity
                gravity_scale=1.0,
                gravity_history=3,
                # Command
                command_scale=1.0,
                # DOF position nominal difference
                include_dof_pos=True,
                include_nominal_difference=True,
                nominal_difference_scale=1.0,
                # DOF velocity
                dof_vel_scale=0.05,
                dof_vel_history=4,
                # Previous actions
                prev_actions_scale=1.0,

                include_base_lin_vel=False,
            )

        # Extra observations for G1
        if not self.extra_actor_observations:
            self.extra_actor_observations = self._default_extra_observations()
        if not self.extra_critic_observations:
            self.extra_critic_observations = self._default_extra_critic_observations()

    def _default_extra_observations(self) -> List[ObservationTermConfig]:
        """G1-specific extra observations."""
        return [
            ObservationTermConfig(
                proprioception.relative_bodies_pos,
                scale=1.0,
                params={
                    "bodies": ("left_ankle_roll_link", "right_ankle_roll_link"),
                },
            ),
            ObservationTermConfig(proprioception.gait_phase_encoding, scale=1.0),
        ]

    def _default_extra_critic_observations(self) -> List[ObservationTermConfig]:
        """Extra critic observations."""
        return [
            ObservationTermConfig(
                proprioception.relative_bodies_pos,
                scale=1.0,
                params={
                    "bodies": ("left_ankle_roll_link", "right_ankle_roll_link"),
                },
            ),
            ObservationTermConfig(proprioception.gait_phase_encoding, scale=1.0),
            ObservationTermConfig(state.base_height, scale=1.0),
            ObservationTermConfig(state.base_lin_vel, scale=1.0),
            ObservationTermConfig(state.base_euler, scale=1.0),
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
            event=EventConfig(event_terms=[]),
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
            base_init_pos=[0.0, 0.0, self.robot.base_init_height],
            base_init_quat=[quat[0], quat[1], quat[2], quat[3]],
            state_init_terms=[
                StateInitializationTermConfig(func=initf.initialize_base_pose, ),
                StateInitializationTermConfig(func=initf.initialize_dof_pos),
            ],
            termination_criteria=[
                TerminationTermConfig(
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0}
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
                    entity_type="ground_plane",
                    shape_cfg=newton.ModelBuilder.ShapeConfig(
                        ke=1.0e3,
                        kd=1.0e2,
                        kf=1.0e3,
                        mu=1.0,
                    ),
                    floating=False
                ),
                NewtonEntityConfig(
                    entity_name="robot",
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
                    ),
                    joint_target_ke_map=self.robot.p_gains,
                    joint_target_kd_map=self.robot.d_gains,
                    joint_armature_map=self.robot.armature,
                    sites={"imu_site_base": self.robot.base_link_name},
                    enable_self_collisions=True
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
            actuated_dof_names=self.robot.actuated_dof_patterns,
            action_scale=0.25,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        reward_terms: dict[str, RewardTermConfig] = {}
        reward_terms.update(self.tracking_rewards.to_terms())
        reward_terms.update(self.regularization_rewards.to_terms())
        reward_terms.update(ContactRewards(
            feet_links=self.robot.foot_names,
            contact_allowed_links=self.robot.foot_names,
            feet_air_time_weight=0.1,
            feet_air_time_threshold=0.35,
            feet_height_weight=None,
        ).to_terms())
        reward_terms.update(self._default_extra_rewards())
        return RewardConfig(reward_terms=reward_terms)

    def _default_extra_rewards(self) -> dict[str, RewardTermConfig]:
        """G1-specific reward terms."""
        feet_links = ("left_ankle_roll_link", "right_ankle_roll_link")
        hip_joints = (".*hip_roll.*", ".*hip_yaw.*")
        return {
            "penalize_ang_vel_xy": RewardTermConfig(
                rf.penalize_ang_vel_xy,
                weight=0.03,
            ),
            "penalize_nonflat_by_gravity": RewardTermConfig(rf.penalize_nonflat_by_gravity, weight=0.1),
            "penalize_dof_vel": RewardTermConfig(rf.penalize_dof_vel, weight=1e-3),
            "penalize_feet_swing_height_gait": RewardTermConfig(
                rf.penalize_feet_swing_height_gait,
                weight=50.0,
                params={"max_height": self.feet_height_target, "foot_offset": 0.035},
            ),
            "penalize_dof_pos_limits": RewardTermConfig(rf.penalize_dof_pos_limits, weight=5.0),
            "reward_gait_pattern": RewardTermConfig(rf.reward_gait_pattern, weight=2.0),
            "reward_alive": RewardTermConfig(rf.reward_alive, weight=0.15),
            "penalize_hip_deviation": RewardTermConfig(rf.penalize_hip_deviation, weight=0.1, params={"hip_joints": hip_joints}),
            "penalize_torques": RewardTermConfig(rf.penalize_torques, weight=5e-6),
            "penalize_base_acc": RewardTermConfig(
                rf.penalize_base_acc,
                weight=1e-4,
                params={"base_body": self.robot.base_link_name},
            ),
            "penalize_feet_slip": RewardTermConfig(
                rf.penalize_feet_slip,
                weight=0.2,
                params={"feet_bodies": ("right_ankle_roll_link", "left_ankle_roll_link")},
            ),
            "penalize_feet_yaw_mean_deviation": RewardTermConfig(rf.penalize_feet_yaw_mean_deviation, params={"feet_bodies": feet_links}, weight=1.0),
            "penalize_feet_yaw_difference": RewardTermConfig(rf.penalize_feet_yaw_difference, params={"feet_bodies": feet_links}, weight=1.0),
        }

    def _build_command_config(self) -> CommandConfig:
        return CommandConfig(
            resampling_time_s=(8.0, 12.0),
            sampler=[
                CommandTermConfig(cf.lin_vel_x, params={"range": self.lin_vel_x_range}),
                CommandTermConfig(cf.lin_vel_y, params={"range": self.lin_vel_y_range}),
                CommandTermConfig(cf.ang_vel, params={"range": self.ang_vel_range}),
                CommandTermConfig(cf.base_height, params={"range": self.base_height_range}),
            ],
        )

    def _build_algorithm_config(self) -> PPOConfig:
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            use_early_stop=False,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.97,
            lam=0.9,
            actor_lr=3e-4,
            critic_lr=3e-4,
            estimator_learning_rate=5e-4,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="fixed",
            use_clipped_value_loss=False,
            value_loss_coef=1.0,
            use_reward_scaling=False,
            obs_normalization=True,
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor_class_name=self.actor_class_name,
                actor_kwargs={
                    "activation": "tanh",
                    "hidden_dims": self.actor_hidden_dims,
                },
                critic_kwargs={
                    "activation": "tanh",
                    "hidden_dims": self.actor_hidden_dims,
                },
                init_noise_std=0.8,
                distribution_type="gaussian",
                std_type="state_independent",
            ),
            state_estimator={
                "activation": "relu",
                "hidden_dims": [256, 128, 64],
            },
        )

    def _build_runner_config(self) -> RunnerConfig:
        return RunnerConfig(
            checkpoint=-1,
            experiment_name="GoAnywhere",
            load_run=None,
            log_interval=1,
            max_iterations=self.max_iterations,
            init_at_random_ep_len=False,
            state_estimator_class_name="StateEstimator",
            low_level_path=None,
            high_level_update_freq=1,
            record_interval=-1,
            resume=False,
            resume_path=None,
            run_name=self.run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            runner_class_name="runner_class_name",
            save_interval=250,
            output_dir="auto",
        )
