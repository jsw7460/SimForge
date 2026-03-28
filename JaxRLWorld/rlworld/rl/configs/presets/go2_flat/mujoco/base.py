"""Go2 MuJoCo base configuration.

This configuration mirrors the Genesis/Newton Go2 flat terrain setup,
adapted for rlworld's MjlabEnv interface.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

import math

from mjlab.asset_zoo.robots import get_go2_robot_cfg, GO2_ACTION_SCALE as MJLAB_GO2_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_spec as go2_get_spec, FULL_COLLISION
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
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
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.go2 import (
    Go2Config,
    STIFFNESS_HIP, STIFFNESS_KNEE, DAMPING_HIP, DAMPING_KNEE,
    ARMATURE_HIP, ARMATURE_KNEE, EFFORT_HIP, EFFORT_KNEE,
)
from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs.scene.unified_entity_config import (
    MujocoEntityCfg, ArticulationCfg, InitialStateCfg,
)
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.observations.mujoco import proprioception
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf


@dataclass
class Go2FlatMujocoConfig:
    """Go2 flat terrain MuJoCo configuration.

    This configuration mirrors the Genesis/Newton Go2 flat terrain setup
    for cross-simulator evaluation via MuJoCo (mjlab).
    """

    # Robot configuration
    robot: Go2Config = field(default_factory=Go2Config)

    # Observation component (matching Genesis/Newton Go2 obs)
    observations: LocomotionObservations | None = None
    extra_actor_observations: List[ObservationTermConfig] = field(default_factory=list)
    extra_critic_observations: List[ObservationTermConfig] = field(default_factory=list)

    # Environment settings
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Simulation settings
    physics_dt: float = 0.005  # 5ms physics timestep (200Hz)
    decimation: int = 4  # Control at 50Hz (matching Genesis/Newton)

    # Command ranges (matching Genesis/Newton Go2)
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Mujoco"

    def __post_init__(self):
        if self.observations is None:
            self.observations = LocomotionObservations(
                # No base linear velocity for actor (matching Genesis/Newton Go2)
                include_base_lin_vel=False,
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

        if not self.extra_critic_observations:
            self.extra_critic_observations = self._default_extra_critic_observations()

    def _default_extra_critic_observations(self) -> List[ObservationTermConfig]:
        """Go2-specific extra critic observations."""
        return [
            ObservationTermConfig(proprioception.base_lin_vel, scale=2.0),
            ObservationTermConfig(proprioception.base_height, scale=1.0),
        ]

    def build(self) -> MujocoConfigsForRun:
        """Build the complete configuration as a typed MujocoConfigsForRun."""
        return MujocoConfigsForRun(
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

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    def _build_env_config(self) -> MujocoEnvConfig:
        return MujocoEnvConfig(
            num_envs=self.num_envs,
            env_name="MjlabEnv",
            task_name="Go2 Velocity Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            decimation=self.decimation,
            termination_criteria=[
                TerminationTermConfig(
                    tf.bad_orientation,
                    {"limit_angle": math.radians(30.0)}
                ),
                TerminationTermConfig(tf.time_out),
            ],
        )

    def _build_scene_config(self) -> MujocoSceneConfig:
        """Build scene config with mjlab SceneCfg."""
        # Foot contact sensor
        foot_names = ("FR", "FL", "RR", "RL")
        geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

        feet_ground_cfg = ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=geom_names,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        )

        # Unified entity config — actuator type determines mjlab actuator:
        #   ImplicitActuatorCfg → BuiltinPositionActuator (simulator PD)
        #   IdealPD/Delayed/etc → BuiltinMotorActuator (direct torque)
        robot_entity = MujocoEntityCfg(
            urdf_path=self.robot.urdf_path,
            init_state=InitialStateCfg(
                pos=(0, 0, self.robot.base_init_height),
                joint_pos=self.robot.default_joint_angles,
            ),
            floating=True,
            articulation=ArticulationCfg(
                actuators=(
                    ImplicitActuatorCfg(
                        target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                        stiffness=STIFFNESS_HIP,
                        damping=DAMPING_HIP,
                        effort_limit=EFFORT_HIP,
                        armature=ARMATURE_HIP,
                    ),
                    ImplicitActuatorCfg(
                        target_names_expr=(".*_calf_joint",),
                        stiffness=STIFFNESS_KNEE,
                        damping=DAMPING_KNEE,
                        effort_limit=EFFORT_KNEE,
                        armature=ARMATURE_KNEE,
                    ),
                ),
            ),
            spec_fn=go2_get_spec,
            collisions=(FULL_COLLISION,),
        )

        # mjlab SceneCfg — entities still uses mjlab EntityCfg for now;
        # the unified_entities dict triggers automatic actuator conversion.
        mjlab_scene_cfg = SceneCfg(
            num_envs=self.num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_go2_robot_cfg()},
            sensors=(feet_ground_cfg,),
        )

        # mjlab SimulationCfg
        mjlab_sim_cfg = SimulationCfg(
            nconmax=35,
            njmax=1500,
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
            unified_entities={"robot": robot_entity},
            preset_class_name=self.__class__.__name__,
            preset_module_path=type(self).__module__,
        )

    def _build_event_config(self) -> EventConfig:
        from rlworld.rl.envs.mdp.events import mujoco_event_terms as ef
        from rlworld.rl.envs.mdp.events.mujoco_event_terms import EntityCfg
        from rlworld.rl.configs.events import EventTermConfig

        foot_geom_names = (
            "FR_foot_collision", "FL_foot_collision",
            "RR_foot_collision", "RL_foot_collision",
        )

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
            ),

            # Startup events (domain randomization)
            EventTermConfig(
                func=ef.randomize_geom_friction,
                mode="startup",
                params={
                    "ranges": (0.3, 1.2),
                    "operation": "abs",
                    "shared_random": True,
                    "entity_cfg": EntityCfg(name="robot", geom_names=foot_geom_names),
                },
            ),
        ])

    def _build_observation_config(self) -> MujocoObservationConfig:
        actor_obs = self.observations.to_terms() + self.extra_actor_observations
        critic_obs = self.observations.to_critic_terms() + self.extra_critic_observations
        return MujocoObservationConfig(
            obs_group={
                "actor": actor_obs,
                "critic": critic_obs,
            },
        )

    def _build_action_config(self) -> MujocoActionConfig:
        return MujocoActionConfig(
            entity_name="robot",
            actuated_dof_names=self.robot.actuated_dof_patterns,
            action_scale=MJLAB_GO2_ACTION_SCALE,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        """Build reward configuration matching Genesis/Newton Go2 rewards."""
        # Site names in Go2 MJCF are "FL", "FR", "RL", "RR"
        site_names = ("FR", "FL", "RR", "RL")

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

            # Orientation reward
            "flat_orientation": RewardTermConfig(
                func=rf.flat_orientation,
                weight=1.0,
                params={"std": 0.447},
            ),

            # Variable posture reward (Go2-specific std values)
            "variable_posture": RewardTermConfig(
                func=rf.variable_posture,
                weight=1.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        joint_names=(".*",),
                    ),
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

            # Joint position limits
            "joint_pos_limits": RewardTermConfig(
                func=rf.joint_pos_limits,
                weight=1.0,
            ),

            # Action rate
            "raw_action_rate_l2": RewardTermConfig(
                func=rf.raw_action_rate_l2,
                weight=0.1,
            ),

            # Feet clearance
            "feet_clearance": RewardTermConfig(
                func=rf.feet_clearance,
                weight=2.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        site_names=site_names,
                    ),
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),

            # Feet swing height
            "feet_swing_height": RewardTermConfig(
                func=rf.feet_swing_height,
                weight=0.25,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        site_names=site_names,
                    ),
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            ),

            # Feet slip
            "feet_slip": RewardTermConfig(
                func=rf.feet_slip,
                weight=0.1,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        site_names=site_names,
                    ),
                    "command_threshold": 0.05,
                },
            ),

            # Soft landing
            "soft_landing": RewardTermConfig(
                func=rf.soft_landing,
                weight=1e-5,
                params={
                    "sensor_name": "feet_ground_contact",
                    "command_threshold": 0.05,
                },
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
