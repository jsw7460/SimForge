from dataclasses import dataclass, field
from typing import Dict, Any, List

import genesis as gs
from rlworld.rl.configs import EventConfig
from rlworld.rl.configs.common_config_classes import CommandConfig
from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
from rlworld.rl.configs.components.rewards.genesis import TrackingRewards, RegularizationRewards
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig, G1_ACTION_SCALE
from rlworld.rl.configs.scene import EntityConfig
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations.genesis import proprioception, state
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf
from rlworld.rl.envs.mdp.rewards.genesis.tasks import g1 as g1rf
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
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
    extra_reward_terms: List[RewardTermConfig] = field(default_factory=list)

    # Environment
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42
    decimation: int = 4

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-0.5, 0.5)
    base_height_range: tuple = (0.68, 0.69)

    # Algorithm
    algorithm_name: str = "PPO"
    max_iterations: int = 15000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    actor_class_name: str = "MLPActor"
    run_name: str = "mlp_ppo_g1_29dof"

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
                tracking_ang_vel_weight=1.0,
            )
        if self.regularization_rewards is None:
            self.regularization_rewards = RegularizationRewards(
                lin_vel_z_weight=0.2,
                base_height_weight=None,
                action_rate_weight=0.01,
                similar_to_default_weight=None,
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

    def _default_extra_rewards(self) -> List[RewardTermConfig]:
        """G1-specific reward terms."""
        feet_links = ("left_ankle_roll_link", "right_ankle_roll_link")
        hip_joints = (".*hip_roll.*", ".*hip_yaw.*")
        return [

            RewardTermConfig(rf.reward_feet_air_time, weight=0.75, params={"links": feet_links}),

            RewardTermConfig(
                rf.penalize_joint_deviation_l1, weight=0.1, params={"joints": hip_joints}
            ),  # hip
            RewardTermConfig(
                rf.penalize_joint_deviation_l1,
                weight=0.1,
                params={"joints": [
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ]}
            ),  # arms
            RewardTermConfig(
                rf.penalize_joint_deviation_l1,
                weight=0.05,
                params={"joints": [
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint"
                ]}
            ),  # hands
            RewardTermConfig(
                rf.penalize_joint_deviation_l1,
                weight=0.1,
                params={"joints": ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]}
            ),  # torso

            RewardTermConfig(
                rf.penalize_ang_vel_xy,
                weight=0.05,
                params={"base_name": self.robot.base_link_name},
            ),
            RewardTermConfig(rf.penalize_nonflat_by_gravity, weight=0.1),
            RewardTermConfig(rf.penalize_dof_vel, weight=1e-3),
            RewardTermConfig(
                g1rf.penalize_feet_swing_height_gait,
                weight=20.0,
                params={"max_height": 0.1},
            ),
            RewardTermConfig(g1rf.penalize_dof_pos_limits, weight=1.0),
            RewardTermConfig(rf.reward_gait_pattern, weight=0.18),
            RewardTermConfig(rf.reward_alive, weight=0.15),
            RewardTermConfig(rf.penalize_torques, weight=1e-5),
            RewardTermConfig(
                rf.penalize_base_acc,
                weight=1e-4,
                params={"base_name": self.robot.base_link_name},
            ),
            RewardTermConfig(
                rf.penalize_feet_slip,
                weight=0.1,
                params={"feet_links": feet_links},
            ),
            RewardTermConfig(rf.penalize_feet_yaw_mean_deviation, params={"feet_links": feet_links}, weight=1.0),
            RewardTermConfig(rf.penalize_feet_yaw_difference, params={"feet_links": feet_links}, weight=1.0),
            RewardTermConfig(
                rf.penalize_feet_distance,
                params={"feet_links": feet_links, "feet_distance_ref": 0.21},
                weight=1.0
            ),
        ]

    def _mjlab_rewards(self) -> List[RewardTermConfig]:
        """G1-specific reward terms (mjlab-compatible)."""
        feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]

        return [
            # Tracking rewards
            RewardTermConfig(
                rf_mjlab.track_lin_vel_mjlab,
                weight=2.0,
                params={"std": 0.5},
            ),
            RewardTermConfig(
                rf_mjlab.track_ang_vel_mjlab,
                weight=2.0,
                params={"std": 0.707, "base_name": self.robot.base_link_name},
            ),

            # Orientation
            RewardTermConfig(
                rf_mjlab.flat_orientation_mjlab,
                weight=1.0,
                params={"std": 0.447, "body_name": self.robot.base_link_name},
            ),

            # Posture
            RewardTermConfig(
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
            RewardTermConfig(
                rf_mjlab.body_ang_vel_penalty_mjlab,
                weight=0.05,
                params={"body_name": self.robot.base_link_name},
            ),
            RewardTermConfig(
                rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
            ),
            RewardTermConfig(
                rf_mjlab.processed_action_rate_l2_mjlab,
                weight=0.1,
            ),

            # Feet rewards
            RewardTermConfig(
                rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_links": feet_links,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_links": feet_links,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_links": feet_links,
                    "command_threshold": 0.05,
                },
            ),
            RewardTermConfig(
                rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_links": feet_links,
                    "command_threshold": 0.05,
                },
            ),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Generate complete configuration dictionary."""
        return {
            "env": self._build_env_config(),
            "visualization": {"show_viewer": False},
            "event": self._build_event_config(),
            "action": self._build_action_config(),
            "scene": self._build_scene_config(),
            "observation": self._build_observation_config(),
            "reward": self._build_reward_config(),
            "command": self._build_command_config(),
            "curriculum": self._build_curriculum_config(),
            "algorithm": self._build_algorithm_config(),
            "nn": self._build_nn_config(),
            "runner": self._build_runner_config(),
        }

    def _build_env_config(self) -> Dict[str, Any]:
        return {
            "env_name": "LocomotionEnv",
            "task_name": "G1_Velocity_Tracking",
            "num_envs": self.num_envs,
            "seed": self.seed,
            "decimation": self.decimation,
            "episode_length_s": self.episode_length_s,
            "termination_criteria": [
                TerminationTermConfig(
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 20.0, "pitch_threshold_degree": 20.0},
                ),
                TerminationTermConfig(max_episode_exceed),
            ],
        }

    def _build_action_config(self) -> Dict[str, Any]:
        return {
            "actuated_dof_names": self.robot.actuated_dof_patterns,
            "action_scale": G1_ACTION_SCALE,
            "clip_actions": (-100.0, 100.0),
            "offset": self.robot.default_joint_angles,
        }

    def _build_event_config(self) -> EventConfig:
        return EventConfig([
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

    def _build_scene_config(self) -> Dict[str, Any]:
        return {
            "entities": [
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
                    armature=self.robot.armature,
                ),
            ],
            "sensors": [
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
            "sim_options": gs.options.SimOptions(dt=0.005, substeps=1),
            "rigid_options": gs.options.RigidOptions(
                dt=0.005,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=True,
            ),
            "robot_cfg": self.robot
        }

    def _build_observation_config(self) -> Dict[str, Any]:
        actor_obs = self.observations.to_terms() + self.extra_actor_observations
        critic_obs = self.observations.to_critic_terms() + self.extra_critic_observations
        return {
            "obs_group": {
                "actor": actor_obs,
                "critic": critic_obs,
            },
        }

    def _build_reward_config(self) -> Dict[str, Any]:
        reward_terms = self._mjlab_rewards()
        return {
            "reward_terms": reward_terms,
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

    def _build_curriculum_config(self) -> Dict[str, Any]:
        return {
            "enable": False,
            "initial_level": 0,
            "max_level": 3,
            "success_threshold": 0.8,
            "min_steps_per_level": 50000,
            "eval_window_size": 2,
            "curriculum_components": {},
            "criterion": {"tracking_lin_vel_xy": -100, "mean_return": -100},
        }

    def _build_algorithm_config(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "clip_param": 0.2,
            "obs_normalization": True,
            "use_early_stop": False,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "estimator_learning_rate": 5e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "use_reward_scaling": False,
        }

    def _build_nn_config(self) -> Dict[str, Any]:
        return {
            "policy": {
                "actor_class_name": self.actor_class_name,
                "actor_kwargs": {
                    "activation": "tanh",
                    "ortho_init": True,
                    "hidden_dims": self.actor_hidden_dims,
                },
                "critic_kwargs": {
                    "activation": "tanh",
                    "ortho_init": True,
                    "hidden_dims": self.actor_hidden_dims,
                },
                "init_noise_std": 1.0,
                "distribution_type": "gaussian",
                "std_type": "state_independent",
            },
            "state_estimator": {
                "activation": "relu",
                "hidden_dims": [256, 128, 64],
            },
        }

    def _build_runner_config(self) -> Dict[str, Any]:
        return {
            "checkpoint": -1,
            "experiment_name": "GoAnywhere",
            "load_run": None,
            "log_interval": 1,
            "max_iterations": self.max_iterations,
            "init_at_random_ep_len": False,
            "state_estimator_class_name": "StateEstimator",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": self.run_name,
            "logger": "wandb",
            "wandb_project": "RLArchitecture",
            "runner_class_name": "runner_class_name",
            "save_interval": 1000,
            "save_path": "auto",
        }
