from dataclasses import dataclass, field
from typing import Dict, Any, List

import math

import genesis as gs
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    RewardConfig, CommandConfig, EventConfig, NNConfig, PPOPolicyConfig, RunnerConfig, VisualizationConfig,
)
from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.genesis_config_classes import (
    EnvConfig, SceneConfig, ObservationConfig, ActionConfig, CurriculumConfig,
)
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.go2 import Go2Config, GO2_ACTION_SCALE, STIFFNESS_HIP, STIFFNESS_KNEE, DAMPING_HIP, DAMPING_KNEE, ARMATURE_HIP, ARMATURE_KNEE, EFFORT_HIP, EFFORT_KNEE
from rlworld.rl.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from rlworld.rl.configs.scene.unified_entity_config import GenesisEntityCfg, ArticulationCfg, InitialStateCfg, GroundPlaneCfg
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.events import event_terms as ef
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf
from rlworld.rl.envs.mdp.terminations.genesis import terminations as tf


@dataclass
class Go2FlatGenesisConfig:
    """
    Base configuration for Go2 flat terrain locomotion.

    Composes robot config and observation/reward components to generate
    the full configuration dictionary.

    Usage:
        config = Go2FlatConfig()
        config_dict = config.to_dict(actor_class_name="MLPActor", run_name="Go2_MLP")
    """

    # Robot configuration
    robot: Go2Config = field(default_factory=Go2Config)

    # Observation component
    observations: LocomotionObservations | None = None

    # Environment settings
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42
    sim_dt = 0.005
    decimation: int = 4

    # Command ranges
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Genesis"

    def __post_init__(self):
        if self.observations is None:
            self.observations = LocomotionObservations(
                base_name=self.robot.base_link_name,
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
            )

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
            env_name="GenesisEnv",
            task_name="Go2_Locomotion",
            num_envs=self.num_envs,
            seed=self.seed,
            decimation=self.decimation,
            episode_length_s=self.episode_length_s,
            termination_criteria=[
                TerminationTermConfig(
                    common_tf.roll_pitch_violation,
                    {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0}
                ),
                TerminationTermConfig(max_episode_exceed),
            ],
        )

    def _build_action_config(self) -> ActionConfig:
        return ActionConfig(
            actuated_dof_names=self.robot.actuated_dof_patterns,
            action_scale=GO2_ACTION_SCALE,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_action_offset(),
        )

    def _build_scene_config(self) -> SceneConfig:
        return SceneConfig(
            env_spacing=(20.0, 20.0),
            entities={
                "base_entity": GroundPlaneCfg(),
                "robot": GenesisEntityCfg(
                    urdf_path=self.robot.urdf_path,
                    init_state=InitialStateCfg(
                        pos=(1.5, 1.5, self.robot.base_init_height),
                        joint_pos=self.robot.default_joint_angles,
                    ),
                    floating=True,
                    links_to_keep=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    articulation=ArticulationCfg(
                        actuators=(
                            DelayedPDActuatorCfg(
                                target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                                stiffness=STIFFNESS_HIP,
                                damping=DAMPING_HIP,
                                effort_limit=EFFORT_HIP,
                                armature=ARMATURE_HIP,
                                min_delay=1,
                                max_delay=3,
                            ),
                            DelayedPDActuatorCfg(
                                target_names_expr=(".*_calf_joint",),
                                stiffness=STIFFNESS_KNEE,
                                damping=DAMPING_KNEE,
                                effort_limit=EFFORT_KNEE,
                                armature=ARMATURE_KNEE,
                                min_delay=1,
                                max_delay=3,
                            ),
                        ),
                    ),
                    convexify=False,
                    visualize_contact=True,
                ),
            },
            sensors=[
                SensorConfig(entity_name="robot", link_name="base", sensor_class=gs.sensors.IMU),
                SensorConfig(entity_name="robot", link_name="FR_foot", sensor_class=gs.sensors.Contact),
                SensorConfig(entity_name="robot", link_name="FL_foot", sensor_class=gs.sensors.Contact),
                SensorConfig(entity_name="robot", link_name="RR_foot", sensor_class=gs.sensors.Contact),
                SensorConfig(entity_name="robot", link_name="RL_foot", sensor_class=gs.sensors.Contact),
                SensorConfig(entity_name="robot", link_name="FR_foot", sensor_class=gs.sensors.ContactForce),
                SensorConfig(entity_name="robot", link_name="FL_foot", sensor_class=gs.sensors.ContactForce),
                SensorConfig(entity_name="robot", link_name="RR_foot", sensor_class=gs.sensors.ContactForce),
                SensorConfig(entity_name="robot", link_name="RL_foot", sensor_class=gs.sensors.ContactForce),
            ],
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            robot_cfg=self.robot,
        )

    def _build_event_config(self) -> EventConfig:
        return EventConfig(event_terms=[
            EventTermConfig(
                func=initf.initialize_dof_pos_with_noise,
                mode="reset",
                params={"position_noise_range": (math.pi / 360, math.pi / 120)},
            ),
            EventTermConfig(
                func=initf.initialize_pos_quat,
                mode="reset",
                params={
                    "base_init_pos": [1.5, 1.5, self.robot.base_init_height + 0.025],
                    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
                },
            ),
            EventTermConfig(
                func=initf.randomize_base_mass,
                mode="reset",
                params={"mass_ratio_range": (0.85, 1.15)},
            ),
            EventTermConfig(
                func=initf.randomize_friction,
                mode="reset",
            ),
            # Interval terms
            EventTermConfig(
                func=ef.apply_external_force_torque,
                mode="interval",
                interval_range_s=(2.0, 20.0),
                params={"force_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-20.0, 20.0)}},
            ),
        ])

    def _build_observation_config(self) -> ObservationConfig:
        return ObservationConfig(
            obs_group={
                "actor": self.observations.to_terms(),
                "critic": self.observations.to_critic_terms(),
            },
        )

    def _build_reward_config(self) -> RewardConfig:
        reward_terms = {
            # Tracking rewards (common — uses RobotData interface)
            "track_lin_vel": RewardTermConfig(
                func=rf_common.track_lin_vel,
                weight=2.0,
                params={"std": 0.5, "penalize_z": True},
            ),
            "track_ang_vel": RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=2.0,
                params={"std": 0.707, "penalize_xy": True},
            ),

            # Orientation (common — uses RobotData interface)
            "flat_orientation": RewardTermConfig(
                func=rf_common.flat_orientation,
                weight=1.0,
                params={"std": 0.447},
            ),

            # Posture reward (stateful class)
            "variable_posture": RewardTermConfig(
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

            # Feet swing height (stateful class)
            "feet_swing_height_mjlab": RewardTermConfig(
                func=rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),

            # Feet clearance
            "feet_clearance_mjlab": RewardTermConfig(
                func=rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),

            # Feet slip
            "feet_slip_mjlab": RewardTermConfig(
                func=rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    "command_threshold": 0.05,
                },
            ),

            # Soft landing
            "soft_landing_mjlab": RewardTermConfig(
                func=rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    "command_threshold": 0.05,
                },
            ),

            # Joint position limits
            "joint_pos_limits_mjlab": RewardTermConfig(
                func=rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
                params={"soft_limit_factor": 1.0},
            ),

            # Action rate
            "processed_action_rate_l2_mjlab": RewardTermConfig(
                func=rf_mjlab.processed_action_rate_l2_mjlab,
                weight=0.1,
            ),
        }

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
            criterion={
                "tracking_lin_vel_xy": -100,
                "mean_return": -100,
            },
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
            upload_checkpoint=False,
            output_dir="auto",
        )
