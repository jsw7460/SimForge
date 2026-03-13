from dataclasses import dataclass, field
from typing import Dict, Any, List

import genesis as gs
from rlworld.rl.configs import EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    RewardConfig, CommandConfig, NNConfig, PPOPolicyConfig, RunnerConfig, VisualizationConfig,
)
from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
from rlworld.rl.configs.components.rewards.genesis import TrackingRewards, RegularizationRewards
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.genesis_config_classes import (
    EnvConfig, SceneConfig, ObservationConfig, ActionConfig, CurriculumConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig, G1_ACTION_SCALE
from rlworld.rl.configs.scene import EntityConfig
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations.genesis import state, proprioception
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf
from rlworld.rl.envs.mdp.rewards.genesis.tasks import g1 as g1rf
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf
from rlworld.rl.envs.mdp.terminations.genesis import terminations as tf


@dataclass
class G1FlatGenesisConfig:
    """Configuration for G1 humanoid flat terrain locomotion."""

    # Robot
    robot: G1MjlabConfig = field(default_factory=G1MjlabConfig)

    # Observations
    observations: LocomotionObservations | None = None
    extra_actor_observations: List[ObservationTermConfig] = field(default_factory=list)
    extra_critic_observations: List[ObservationTermConfig] = field(default_factory=list)

    # Rewards
    tracking_rewards: TrackingRewards | None = None
    regularization_rewards: RegularizationRewards | None = None
    extra_reward_terms: dict[str, RewardTermConfig] = field(default_factory=dict)

    # Environment
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42
    decimation: int = 4

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-0.5, 0.5)

    # Algorithm
    algorithm_name: str = "PPO"
    max_iterations: int = 30000
    actor_class_name: str = "MLPActor"
    run_name: str = "mlp_ppo_g1_29dof"

    def __post_init__(self):
        # Observations - match original config
        if self.observations is None:
            self.observations = LocomotionObservations(
                base_name="pelvis",
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

        # Rewards
        if self.tracking_rewards is None:
            self.tracking_rewards = TrackingRewards(
                base_name=self.robot.base_link_name,
                tracking_lin_vel_weight=1.0,
                tracking_ang_vel_weight=1.0,
            )
        if self.regularization_rewards is None:
            self.regularization_rewards = RegularizationRewards(
                lin_vel_z_weight=0.2,
                base_height_weight=None,
                action_rate_weight=0.1,
                similar_to_default_weight=None,
            )

    def _default_extra_observations(self) -> List[ObservationTermConfig]:
        """G1-specific extra observations."""
        return []

    def _default_extra_critic_observations(self) -> List[ObservationTermConfig]:
        """Extra critic observations."""
        feet_links = ("left_ankle_roll_link", "right_ankle_roll_link")
        return [
            ObservationTermConfig(state.base_height, scale=1.0),
            ObservationTermConfig(state.base_euler, scale=1.0),
            ObservationTermConfig(state.contact_indicator, scale=1.0, params={"links": feet_links}),
            ObservationTermConfig(state.contact_force, scale=0.01, params={"links": feet_links}),
            ObservationTermConfig(state.feet_height, scale=1.0, params={"links": feet_links}),
            ObservationTermConfig(state.foot_air_time, scale=1.0, params={"links": feet_links}),

            # ObservationTermConfig(, scale=1.0),
        ]

    def _mjlab_rewards(self) -> dict[str, RewardTermConfig]:
        """G1-specific reward terms (mjlab-compatible)."""
        feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]

        return {
            # Tracking rewards (common — uses RobotData interface)
            "track_lin_vel": RewardTermConfig(
                rf_common.track_lin_vel,
                weight=2.0,
                params={"std": 0.5, "penalize_z": True},
            ),
            "track_ang_vel": RewardTermConfig(
                rf_common.track_ang_vel,
                weight=2.0,
                params={"std": 0.707, "penalize_xy": True},
            ),

            # Orientation (common — uses RobotData interface)
            "flat_orientation": RewardTermConfig(
                rf_common.flat_orientation,
                weight=1.0,
                params={"std": 0.447},
            ),

            # Posture
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
                        r".*waist_yaw.*": 0.2,
                        r".*waist_roll.*": 0.08,
                        r".*waist_pitch.*": 0.1,
                        r".*shoulder_pitch.*": 0.15,
                        r".*shoulder_roll.*": 0.15,
                        r".*shoulder_yaw.*": 0.1,
                        r".*elbow.*": 0.15,
                        r".*wrist.*": 0.3,
                    },
                    "std_running": {
                        r".*hip_pitch.*": 0.5,
                        r".*hip_roll.*": 0.2,
                        r".*hip_yaw.*": 0.2,
                        r".*knee.*": 0.6,
                        r".*ankle_pitch.*": 0.35,
                        r".*ankle_roll.*": 0.15,
                        r".*waist_yaw.*": 0.3,
                        r".*waist_roll.*": 0.08,
                        r".*waist_pitch.*": 0.2,
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
                params={"body_name": self.robot.base_link_name},
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
                    "feet_links": feet_links,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            "feet_swing_height_mjlab": RewardTermConfig(
                rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_links": feet_links,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            "feet_slip_mjlab": RewardTermConfig(
                rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_links": feet_links,
                    "command_threshold": 0.05,
                },
            ),
            "soft_landing_mjlab": RewardTermConfig(
                rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_links": feet_links,
                    "command_threshold": 0.05,
                },
            ),
        }

    def build(self) -> "GenesisConfigsForRun":
        """Build the complete configuration as a typed GenesisConfigsForRun."""
        from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun

        return GenesisConfigsForRun(
            env=self._build_env_config(),
            scene=self._build_scene_config(),
            visualization=VisualizationConfig(show_viewer=False),
            observation=self._build_observation_config(),
            action=self._build_action_config(),
            reward=self._build_reward_config(),
            command=self._build_command_config(),
            event=self._build_event_config(),
            curriculum=self._build_curriculum_config(),
            algorithm=self._build_algorithm_config(),
            nn=self._build_nn_config(),
            runner=self._build_runner_config(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    def _build_env_config(self) -> EnvConfig:
        return EnvConfig(
            env_name="LocomotionEnv",
            task_name="G1_Velocity_Tracking",
            num_envs=self.num_envs,
            seed=self.seed,
            decimation=self.decimation,
            episode_length_s=self.episode_length_s,
            termination_criteria=[
                TerminationTermConfig(
                    common_tf.roll_pitch_violation,
                    {"roll_threshold_degree": 20.0, "pitch_threshold_degree": 20.0},
                ),
                TerminationTermConfig(max_episode_exceed),
            ],
        )

    def _build_action_config(self) -> ActionConfig:
        return ActionConfig(
            actuated_dof_names=self.robot.actuated_dof_patterns,
            action_scale=G1_ACTION_SCALE,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.default_joint_angles,
        )

    def _build_event_config(self) -> EventConfig:
        return EventConfig(event_terms=[
            EventTermConfig(func=initf.initialize_dof_pos, mode="reset"),
            EventTermConfig(
                func=initf.initialize_pos_quat,
                mode="reset",
                params={
                    "base_init_pos": [0.0, 0.0, self.robot.base_init_height],
                    "base_init_quat": [1.0, 0.0, 0.0, 0.0]
                }
            ),
        ])

    def _build_scene_config(self) -> SceneConfig:
        return SceneConfig(
            entities=[
                EntityConfig(
                    entity_name="base_entity",
                    morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
                ),
                EntityConfig(
                    entity_name="robot",
                    morph=gs.morphs.URDF(
                        file=self.robot.urdf_path,
                        # links_to_keep=[self.robot.base_link_name],
                        convexify=True,
                    ),
                    visualize_contact=False,
                    p_gain=self.robot.p_gains,
                    d_gain=self.robot.d_gains,
                    armature=self.robot.armature,
                ),
            ],
            sensors=[
                SensorConfig(
                    entity_name="robot",
                    link_name="pelvis",
                    sensor_class=gs.sensors.IMU,
                ),
                SensorConfig(
                    entity_name="robot",
                    link_name="left_ankle_roll_link",
                    sensor_class=gs.sensors.Contact,
                ),
                SensorConfig(
                    entity_name="robot",
                    link_name="right_ankle_roll_link",
                    sensor_class=gs.sensors.Contact,
                ),
                SensorConfig(
                    entity_name="robot",
                    link_name="left_ankle_roll_link",
                    sensor_class=gs.sensors.ContactForce,
                ),
                SensorConfig(
                    entity_name="robot",
                    link_name="right_ankle_roll_link",
                    sensor_class=gs.sensors.ContactForce,
                )
            ],
            sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
            rigid_options=gs.options.RigidOptions(
                dt=0.005,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
            robot_cfg=self.robot,
        )

    def _build_observation_config(self) -> ObservationConfig:
        actor_obs = self.observations.to_terms()
        critic_obs = self.observations.to_critic_terms() + self.extra_critic_observations
        return ObservationConfig(
            obs_group={
                "actor": actor_obs,
                "critic": critic_obs,
            },
        )

    def _build_reward_config(self) -> RewardConfig:
        reward_terms = self._mjlab_rewards()
        return RewardConfig(reward_terms=reward_terms)

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

    def _build_curriculum_config(self) -> CurriculumConfig:
        return CurriculumConfig(
            enable=False,
            initial_level=0,
            max_level=3,
            success_threshold=0.8,
            min_steps_per_level=50000,
            eval_window_size=2,
            curriculum_components={},
            criterion={"tracking_lin_vel_xy": -100, "mean_return": -100},
        )

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
            init_at_random_ep_len=False,
            resume=False,
            resume_path=None,
            run_name=self.run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            save_interval=1000,
        )
