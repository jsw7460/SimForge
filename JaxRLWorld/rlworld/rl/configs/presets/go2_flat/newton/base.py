from dataclasses import dataclass, field
from typing import Dict, Any, List

import math
import warp as wp

import newton
from rlworld.rl.configs import RewardConfig, CommandConfig, GaitConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig, TerminationsConfig, ObservationGroupConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.envs.mdp.observations.common.proprioception import base_ang_vel, projected_gravity, dof_pos, dof_vel, raw_actions, base_lin_vel, base_height
from rlworld.rl.envs.mdp.observations.genesis.exteroception import command as command_obs
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
from rlworld.rl.configs.robots.go2 import (
    Go2Config, GO2_ACTION_SCALE,
    STIFFNESS_HIP, STIFFNESS_KNEE, DAMPING_HIP, DAMPING_KNEE,
    ARMATURE_HIP, ARMATURE_KNEE, EFFORT_HIP, EFFORT_KNEE,
)
from rlworld.rl.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from rlworld.rl.configs.scene.unified_entity_config import (
    NewtonEntityCfg as UnifiedNewtonEntityCfg, ArticulationCfg, InitialStateCfg, GroundPlaneCfg,
)
from rlworld.rl.configs.sensors import NewtonIMUSensorConfig, NewtonContactSensorConfig
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
)
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf


@dataclass
class Go2FlatNewtonConfig:
    """Base configuration for Go2 flat terrain locomotion on Newton simulator."""

    robot: Go2Config = field(default_factory=Go2Config)

    tracking_rewards: TrackingRewards = field(default_factory=TrackingRewards)
    regularization_rewards: RegularizationRewards = field(default_factory=RegularizationRewards)

    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    dt: float = 0.005
    substeps: int = 2

    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)

    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Newton"

    def build(self) -> "NewtonConfigsForRun":
        """Build the complete configuration as a typed NewtonConfigsForRun."""
        from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)

        cfgs = NewtonConfigsForRun(
            env=self._build_env_config(quat),
            scene=self._build_scene_config(quat),
            visualization=VisualizationConfig(show_viewer=False, record_video=False),
            observation=self._build_observation_config(),
            action=self._build_action_config(),
            reward=self._build_reward_config(),
            command=self._build_command_config(),
            event=self._build_event_config(quat),
            gait=self._build_gait_config(),
            algorithm=self._build_algorithm_config(),
            nn=self._build_nn_config(),
            runner=self._build_runner_config(),
        )
        cfgs.preset_module = type(self).__module__
        return cfgs

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    def _build_env_config(self, quat) -> NewtonEnvConfig:
        @dataclass
        class _TerminationsCfg(TerminationsConfig):
            roll_pitch = TerminationTermConfig(
                common_tf.roll_pitch_violation,
                {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0},
            )
            max_episode = TerminationTermConfig(max_episode_exceed)

        return NewtonEnvConfig(
            env_name="NewtonLocomotionEnv",
            num_envs=self.num_envs,
            task_name="Go2 Velocity Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            decimation=4,
            terminations=_TerminationsCfg(),
        )

    def _build_scene_config(self, quat) -> NewtonSceneConfig:
        r = self.robot
        return NewtonSceneConfig(
            dt=self.dt,
            substeps=self.substeps,
            gravity=(0.0, 0.0, -9.81),
            solver_type="mujoco",
            entities={
                "ground": GroundPlaneCfg(),
                "robot": UnifiedNewtonEntityCfg(
                    urdf_path=r.urdf_path,
                    init_state=InitialStateCfg(
                        pos=(0.0, 0.0, r.base_init_height),
                        rot=(quat[0], quat[1], quat[2], quat[3]),
                    ),
                    floating=True,
                    collapse_fixed_joints=True,
                    links_to_keep=[
                        "go2_description/FL_foot_joint",
                        "go2_description/FR_foot_joint",
                        "go2_description/RL_foot_joint",
                        "go2_description/RR_foot_joint",
                    ],
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
                    body_label_prefix=r.name,
                    sites={"imu_site_base": r.base_link_name},
                ),
            },
            sensors=[
                NewtonIMUSensorConfig(
                    entity_name="robot",
                    sensor_name="imu_base",
                    site_names=["imu_site_base"]
                ),
                NewtonContactSensorConfig(
                    entity_name="robot",
                    sensor_name="foot_contact",
                    sensing_obj_bodies=list(r.prefixed_foot_names),
                ),
                NewtonContactSensorConfig(
                    entity_name="robot",
                    sensor_name="body_ground_contact",
                    sensing_obj_bodies=["*"],
                    exclude_bodies=("*foot*",),
                )
            ],
            add_ground=True,
            env_spacing=(2.0, 2.0, 0.0),
            robot_cfg=self.robot
        )

    def _build_observation_config(self) -> NewtonObservationConfig:
        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=0.25, noise=Unoise(-0.2, 0.2))
            projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
            command = ObservationTermConfig(func=command_obs, scale=1.0)
            dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            dof_vel = ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5))
            prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=2.0)
            base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)

        @dataclass
        class _ObsCfg(NewtonObservationConfig):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def _build_action_config(self) -> NewtonActionConfig:
        r = self.robot
        return NewtonActionConfig(
            actuated_dof_names=r.prefixed_actuated_dof_patterns,
            action_scale=GO2_ACTION_SCALE,
            clip_actions=(-100.0, 100.0),
            offset=r.get_prefixed_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        feet = list(r.prefixed_foot_names)

        @dataclass
        class _RewardsCfg(RewardConfig):
            # Tracking rewards (common — uses RobotData interface)
            track_lin_vel = RewardTermConfig(
                func=rf_common.track_lin_vel,
                weight=2.0,
                params={"std": 0.5, "penalize_z": True},
            )
            track_ang_vel = RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=2.0,
                params={"std": 0.707, "penalize_xy": True},
            )
            # Orientation (common — uses RobotData interface)
            flat_orientation = RewardTermConfig(
                func=rf_common.flat_orientation,
                weight=1.0,
                params={"std": 0.447},
            )
            variable_posture = RewardTermConfig(
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
            )
            feet_swing_height_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_bodies": feet,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            )
            feet_clearance_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_bodies": feet,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            )
            feet_slip_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_bodies": feet,
                    "command_threshold": 0.05,
                },
            )
            soft_landing_mjlab = RewardTermConfig(
                func=rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_bodies": feet,
                    "command_threshold": 0.05,
                },
            )
            joint_pos_limits_mjlab = RewardTermConfig(
                func=rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
                params={"soft_limit_factor": 1.0},
            )
            processed_action_rate_l2_mjlab = RewardTermConfig(
                func=rf_mjlab.processed_action_rate_l2_mjlab,
                weight=0.1,
            )

        return _RewardsCfg()

    def _build_command_config(self) -> CommandConfig:
        from rlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
        return CommandConfig(
            terms={
                "velocity": VelocityCommandTermCfg(
                    resampling_time_range=(3.0, 8.0),
                    lin_vel_x_range=self.lin_vel_x_range,
                    lin_vel_y_range=self.lin_vel_y_range,
                    ang_vel_range=self.ang_vel_range,
                    rel_standing_envs=0.1,
                    heading_command=True,
                    heading_control_stiffness=0.5,
                    heading_range=(-3.14, 3.14),
                    rel_heading_envs=0.3,
                ),
            }
        )

    def _build_gait_config(self) -> GaitConfig:
        return GaitConfig(
            foot_names=self.robot.prefixed_foot_names,
        )

    def _build_event_config(self, quat) -> EventConfig:
        r = self.robot
        from rlworld.rl.envs.mdp.events.newton_event_terms import (
            push_robot as _push_robot_fn,
            reset_root_state_uniform as _reset_root_fn,
        )
        from rlworld.rl.envs.mdp.events.dr import newton as newton_dr

        @dataclass
        class _EventsCfg(EventConfig):
            reset_root = EventTermConfig(
                func=_reset_root_fn,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.0, 0.0),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {},
                },
            )
            reset_dof_pos = EventTermConfig(
                func=initf.initialize_dof_pos_with_noise,
                params={"position_noise_range": (math.pi / 360, math.pi / 120)},
                mode="reset",
            )
            randomize_body_mass = EventTermConfig(
                func=newton_dr.randomize_body_mass,
                params={
                    "mass_range": (0.8, 1.2),
                    "operation": "scale",
                    "body_patterns": r.prefixed("base"),
                },
                mode="reset",
            )
            randomize_friction = EventTermConfig(
                func=newton_dr.randomize_friction,
                mode="reset",
                params={"friction_range": (0.3, 1.2)},
            )
            randomize_pd_gains = EventTermConfig(
                func=newton_dr.randomize_pd_gains,
                mode="reset",
                params={
                    "kp_range": (0.9, 1.1),
                    "kd_range": (0.9, 1.1),
                    "operation": "scale",
                },
            )
            randomize_joint_armature = EventTermConfig(
                func=newton_dr.randomize_joint_armature,
                mode="reset",
                params={
                    "armature_range": (0.9, 1.1),
                    "operation": "scale",
                },
            )
            randomize_joint_friction = EventTermConfig(
                func=newton_dr.randomize_joint_friction,
                mode="reset",
                params={"friction_range": (0.0, 0.05)},
            )
            push_robot = EventTermConfig(
                func=_push_robot_fn,
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

        return _EventsCfg()

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
