"""Go1 MuJoCo base configuration.

This configuration follows the mjlab velocity task setup,
adapted for rlworld's MujocoEnv interface.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

import math

from mjlab.asset_zoo.robots import get_go1_robot_cfg, GO1_ACTION_SCALE
from mjlab.managers.scene_entity_config import SceneEntityCfg
# mjlab imports for scene configuration
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainImporterCfg
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig
from rlworld.rl.configs.components.observations.mujoco import LocomotionObservations
from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoEnvConfig,
    MujocoSceneConfig,
    MujocoObservationConfig,
    MujocoActionConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.go1 import Go1MujocoConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf


@dataclass
class Go1FlatMujocoConfig:
    """Go1 flat terrain MuJoCo configuration.

    This configuration mirrors the mjlab velocity task setup for Go1,
    including the same reward functions, observations, and termination
    conditions.
    """

    # Robot configuration
    robot: Go1MujocoConfig = field(default_factory=Go1MujocoConfig)

    # Observation component
    observations: LocomotionObservations | None = None

    # Environment settings
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Simulation settings (matching mjlab)
    physics_dt: float = 0.005  # 5ms physics timestep (200Hz)
    decimation: int = 4  # Control at 50Hz

    # Command ranges (matching mjlab velocity task)
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-0.5, 0.5)
    base_height_range: tuple = (0.278, 0.278)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 4000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])

    actor_class_name: str = "MLPActor"
    run_name: str = "Go1_Mujoco"

    def __post_init__(self):
        if self.observations is None:
            self.observations = LocomotionObservations(
                # Base linear velocity (matching mjlab noise)
                base_lin_vel_scale=1.0,  # mjlab uses raw values
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
                include_dof_pos=False,
                include_nominal_difference=True,
                # DOF velocity
                dof_vel_scale=1.0,
                dof_vel_noise=Unoise(-1.5, 1.5),
                # Previous actions
                prev_actions_scale=1.0,
            )

    def build(self) -> "MujocoConfigsForRun":
        """Build the complete configuration as a typed MujocoConfigsForRun."""
        cfgs = MujocoConfigsForRun(
            env=self._build_env_config(),
            scene=self._build_scene_config(),
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
        cfgs.preset_module = type(self).__module__
        return cfgs

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    def _build_env_config(self) -> MujocoEnvConfig:
        return MujocoEnvConfig(
            num_envs=self.num_envs,
            env_name="MujocoEnv",
            task_name="Go1 Velocity Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            decimation=self.decimation,
            termination_criteria=[
                TerminationTermConfig(
                    tf.bad_orientation,
                    {"limit_angle": math.radians(70.0)}
                ),
                TerminationTermConfig(tf.time_out),
            ],
        )

    def _build_scene_config(self) -> MujocoSceneConfig:
        """Build scene config with mjlab SceneCfg and SimulationCfg."""
        # Foot geom names for contact sensing
        foot_names = ("FR", "FL", "RR", "RL")
        geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

        # Contact sensor for feet-ground contact
        feet_ground_cfg = ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        )

        # Contact sensor for non-foot ground contact (for illegal contact termination)
        nonfoot_ground_cfg = ContactSensorCfg(
            name="nonfoot_ground_touch",
            primary=ContactMatch(
                mode="geom",
                entity="robot",
                pattern=r".*_collision\d*$",
                exclude=tuple(geom_names),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=self.decimation
        )

        # mjlab SceneCfg with robot entity and sensors
        mjlab_scene_cfg = SceneCfg(
            num_envs=self.num_envs,
            env_spacing=2.0,
            terrain=TerrainImporterCfg(terrain_type="plane"),  # Flat terrain
            entities={"robot": get_go1_robot_cfg()},
            sensors=(feet_ground_cfg, nonfoot_ground_cfg),
        )

        # mjlab SimulationCfg
        mjlab_sim_cfg = SimulationCfg(
            nconmax=300,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=self.physics_dt,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
            contact_sensor_maxmatch=64,
        )

        return MujocoSceneConfig(
            physics_dt=self.physics_dt,
            num_envs=self.num_envs,
            env_spacing=2.0,
            robot_entity_name="robot",
            mjlab_scene_cfg=mjlab_scene_cfg,
            mjlab_sim_cfg=mjlab_sim_cfg,
            preset_class_name=self.__class__.__name__,
            preset_module_path=type(self).__module__,
        )

    def _build_event_config(self) -> EventConfig:
        from rlworld.rl.envs.mdp.events import mujoco_event_terms as ef
        from rlworld.rl.envs.mdp.events.mujoco_event_terms import EntityCfg
        from rlworld.rl.configs.events import EventTermConfig

        return EventConfig([
            # Reset events
            EventTermConfig(
                func=ef.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.01, 0.05),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {},
                },
            ),
            EventTermConfig(
                func=ef.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "entity_cfg": EntityCfg(name="robot", joint_names=(".*",)),
                },
            ),

            # Interval events
            EventTermConfig(
                func=ef.push_by_setting_velocity,
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

            # Startup events (domain randomization)
            EventTermConfig(
                func=ef.randomize_geom_friction,
                mode="startup",
                params={
                    "ranges": (0.3, 1.2),
                    "operation": "abs",
                    "shared_random": True,
                    "entity_cfg": EntityCfg(
                        name="robot",
                        geom_names=("FR_foot_collision", "FL_foot_collision",
                                    "RR_foot_collision", "RL_foot_collision"),
                    ),
                },
            ),
            EventTermConfig(
                func=ef.randomize_encoder_bias,
                mode="startup",
                params={
                    "bias_range": (-0.015, 0.015),
                    "entity_cfg": EntityCfg(name="robot"),
                },
            ),
            EventTermConfig(
                func=ef.randomize_body_com_offset,
                mode="startup",
                params={
                    "ranges": {
                        0: (-0.025, 0.025),
                        1: (-0.025, 0.025),
                        2: (-0.03, 0.03),
                    },
                    "operation": "add",
                    "entity_cfg": EntityCfg(name="robot", body_names=("trunk",)),
                },
            ),
        ])

    def _build_observation_config(self) -> MujocoObservationConfig:
        return MujocoObservationConfig(
            obs_group={
                "actor": self.observations.to_terms(),
                "critic": self.observations.to_critic_terms(),
            },
        )

    def _build_action_config(self) -> MujocoActionConfig:
        # Use mjlab's GO1_ACTION_SCALE for per-joint scaling
        return MujocoActionConfig(
            entity_name="robot",
            actuated_dof_names=[
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
            action_scale=GO1_ACTION_SCALE,  # Per-joint action scale from mjlab
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        """Build reward configuration matching mjlab velocity task."""
        reward_terms = {
            # === Tracking rewards (matching mjlab) ===
            "track_linear_velocity": RewardTermConfig(
                func=rf.track_linear_velocity,
                weight=2.0,
                params={"std": math.sqrt(0.25)},
            ),
            "track_angular_velocity": RewardTermConfig(
                func=rf.track_angular_velocity,
                weight=2.0,
                params={"std": math.sqrt(0.5)},
            ),

            # === Orientation reward ===
            "flat_orientation": RewardTermConfig(
                func=rf.flat_orientation,
                weight=1.0,
                params={"std": math.sqrt(0.2)},
            ),

            # === Posture reward ===
            "posture": RewardTermConfig(
                func=rf.posture,
                weight=1.0,
                params={"std": 0.25},
            ),

            # === Joint limits penalty ===
            "joint_pos_limits": RewardTermConfig(
                func=rf.joint_pos_limits,
                weight=1.0,
            ),

            # === Action rate penalty ===
            "action_rate_l2": RewardTermConfig(
                func=rf.action_rate_l2,
                weight=0.1,
            ),

            # === Feet air time ===
            "feet_air_time": RewardTermConfig(
                func=rf.feet_air_time,
                weight=0.0,
                params={
                    "contact_group": "feet_ground_contact",
                    "threshold_min": 0.05,
                    "threshold_max": 0.5,
                    "command_threshold": 0.5,
                },
            ),

            # === Feet clearance penalty ===
            "feet_clearance": RewardTermConfig(
                func=rf.feet_clearance,
                weight=2.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        site_names=("FR", "FL", "RR", "RL"),
                    ),
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),

            # === Feet slip penalty ===
            "feet_slip": RewardTermConfig(
                func=rf.feet_slip,
                weight=0.1,
                params={
                    "contact_group": "feet_ground_contact",
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        site_names=("FR", "FL", "RR", "RL"),
                    ),
                    "command_threshold": 0.05,
                },
            ),

            # === Soft landing penalty ===
            "soft_landing": RewardTermConfig(
                func=rf.soft_landing,
                weight=1e-5,
                params={
                    "contact_group": "feet_ground_contact",
                    "command_threshold": 0.05,
                },
            ),

            # === Body angular velocity penalty ===
            "body_angular_velocity_penalty": RewardTermConfig(
                func=rf.body_angular_velocity_penalty,
                weight=0.0,
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
                    "activation": "elu",
                    "hidden_dims": self.actor_hidden_dims,
                    "ortho_init": True,
                    "output_gain": 0.01,
                },
                critic_kwargs={
                    "activation": "elu",
                    "hidden_dims": self.actor_hidden_dims,
                    "ortho_init": True,
                    "output_gain": 0.01,
                },
                init_noise_std=1.0,
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
