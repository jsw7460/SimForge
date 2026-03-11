from dataclasses import dataclass, field
from typing import Dict, Any, List

import genesis as gs
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    RewardConfig, CommandConfig, EventConfig, NNConfig, PPOPolicyConfig, RunnerConfig, VisualizationConfig,
)
from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
from rlworld.rl.configs.components.rewards.genesis import TrackingRewards, RegularizationRewards
from rlworld.rl.configs.components.rewards.genesis import ContactRewards
from rlworld.rl.configs.genesis_config_classes import (
    EnvConfig, SceneConfig, ObservationConfig, ActionConfig, CurriculumConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_12dof import G1Config
from rlworld.rl.configs.scene import EntityConfig
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    StateInitializationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations import proprioception, state
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf
from rlworld.rl.envs.mdp.rewards.genesis.tasks import g1 as g1rf
from rlworld.rl.envs.mdp.terminations import terminations as tf


@dataclass
class G1FlatGenesisConfig:
    """Configuration for G1 humanoid flat terrain locomotion."""

    # Robot
    robot: G1Config = field(default_factory=G1Config)

    # Observations
    observations: LocomotionObservations | None = None
    extra_actor_observations: List[ObservationTermConfig] = field(default_factory=list)
    extra_critic_observations: List[ObservationTermConfig] = field(default_factory=list)

    # Rewards
    tracking_rewards: TrackingRewards | None = None
    regularization_rewards: RegularizationRewards | None = None
    extra_reward_terms: dict[str, RewardTermConfig] = field(default_factory=dict)

    # Environment
    num_envs: int = 8192
    episode_length_s: float = 30.0
    seed: int = 42
    decimation: int = 4

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)
    base_height_range: tuple = (0.72, 0.72)

    # Algorithm
    algorithm_name: str = "PPO"
    max_iterations: int = 5000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_12dof_Genesis"

    def __post_init__(self):
        # Observations - match original config
        if self.observations is None:
            self.observations = LocomotionObservations(
                base_name=self.robot.base_link_name,
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

        # Rewards
        if self.tracking_rewards is None:
            self.tracking_rewards = TrackingRewards(
                base_name=self.robot.base_link_name,
                tracking_lin_vel_weight=1.0,
                tracking_ang_vel_weight=0.5,
            )
        if self.regularization_rewards is None:
            self.regularization_rewards = RegularizationRewards(
                lin_vel_z_weight=2.0,
                base_height_weight=None,
                action_rate_weight=0.01,
                similar_to_default_weight=0.1,
            )
        if not self.extra_reward_terms:
            self.extra_reward_terms = self._default_extra_rewards()

    def _default_extra_observations(self) -> List[ObservationTermConfig]:
        """G1-specific extra observations."""
        return [
            ObservationTermConfig(
                proprioception.relative_links_pos,
                scale=1.0,
                params={
                    "base_name": self.robot.base_link_name,
                    "links": ("left_ankle_roll_link", "right_ankle_roll_link"),
                },
            ),
            ObservationTermConfig(proprioception.gait_phase_encoding, scale=1.0),
        ]

    def _default_extra_critic_observations(self) -> List[ObservationTermConfig]:
        """Extra critic observations."""
        return [
            ObservationTermConfig(state.base_height, scale=1.0),
            ObservationTermConfig(state.base_lin_vel, scale=1.0),
            ObservationTermConfig(state.base_euler, scale=1.0),
        ]

    def _default_extra_rewards(self) -> dict[str, RewardTermConfig]:
        """G1-specific reward terms."""
        feet_links = ("left_ankle_roll_link", "right_ankle_roll_link")
        hip_joints = (".*hip_roll.*", ".*hip_yaw.*")
        return {
            "penalize_ang_vel_xy": RewardTermConfig(
                rf.penalize_ang_vel_xy,
                weight=0.03,
                params={"base_name": self.robot.base_link_name},
            ),
            "penalize_nonflat_by_gravity": RewardTermConfig(rf.penalize_nonflat_by_gravity, weight=0.1),
            "penalize_dof_vel": RewardTermConfig(rf.penalize_dof_vel, weight=1e-3),
            "penalize_dof_pos_limits": RewardTermConfig(g1rf.penalize_dof_pos_limits, weight=5.0),
            "reward_gait_pattern": RewardTermConfig(rf.reward_gait_pattern, weight=2.0),
            "reward_alive": RewardTermConfig(rf.reward_alive, weight=0.15),
            "penalize_hip_pos": RewardTermConfig(g1rf.penalize_hip_pos, weight=0.1, params={"hip_joints": hip_joints}),
            "penalize_torques": RewardTermConfig(rf.penalize_torques, weight=1e-5),
            "penalize_feet_slip": RewardTermConfig(
                rf.penalize_feet_slip,
                weight=0.2,
                params={"feet_links": feet_links},
            ),
            "penalize_feet_yaw_mean_deviation": RewardTermConfig(rf.penalize_feet_yaw_mean_deviation, params={"feet_links": feet_links}, weight=1.0),
            "penalize_feet_yaw_difference": RewardTermConfig(rf.penalize_feet_yaw_difference, params={"feet_links": feet_links}, weight=1.0),
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
            event=EventConfig(),
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
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0},
                ),
                TerminationTermConfig(tf.max_episode_exceed),
            ],
        )

    def _build_action_config(self) -> ActionConfig:
        return ActionConfig(
            actuated_dof_names=self.robot.actuated_dof_patterns,
            action_scale=0.25,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_action_offset(),
        )

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
                        links_to_keep=[self.robot.base_link_name],
                        convexify=True,
                    ),
                    visualize_contact=True,
                    p_gain=self.robot.p_gains,
                    d_gain=self.robot.d_gains,
                ),
            ],
            sensors=[
                SensorConfig(
                    entity_name="robot",
                    link_name=self.robot.base_link_name,
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
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            robot_cfg=self.robot,
        )

    def _build_observation_config(self) -> ObservationConfig:
        actor_obs = self.observations.to_terms() + self.extra_actor_observations
        critic_obs = self.observations.to_critic_terms() + self.extra_critic_observations
        return ObservationConfig(
            obs_group={
                "actor": actor_obs,
                "critic": critic_obs,
            },
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
        reward_terms.update(self.extra_reward_terms)
        return RewardConfig(
            tracking_sigma=0.25,
            reward_terms=reward_terms,
        )

    def _build_command_config(self) -> CommandConfig:
        return CommandConfig(
            sampler=[
                CommandTermConfig(cf.lin_vel_x, params={"range": self.lin_vel_x_range}),
                CommandTermConfig(cf.lin_vel_y, params={"range": self.lin_vel_y_range}),
                CommandTermConfig(cf.ang_vel, params={"range": self.ang_vel_range}),
                CommandTermConfig(cf.base_height, params={"range": self.base_height_range}),
            ],
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
            num_steps_per_env=24,
            policy_class_name="PPOActorCritic",
            state_estimator_class_name="StateEstimator",
            record_interval=-1,
            resume=False,
            resume_path=None,
            run_name=self.run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            runner_class_name="runner_class_name",
            save_interval=250,
            save_path="auto",
        )
