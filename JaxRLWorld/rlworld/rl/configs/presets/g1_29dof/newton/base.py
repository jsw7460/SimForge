from dataclasses import dataclass, field
from typing import Dict, Any

import warp as wp

import newton
from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig, RunnerConfig, TerminationsConfig, \
    ObservationGroupConfig
from rlworld.rl.configs.events import EventTermConfig
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
from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig, G1_ACTION_SCALE
from rlworld.rl.configs.scene.unified_entity_config import NewtonEntityCfg, ArticulationCfg, InitialStateCfg, \
    GroundPlaneCfg
from rlworld.rl.configs.sensors import NewtonIMUSensorConfig, NewtonContactSensorConfig
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
)
from rlworld.rl.envs.mdp.events.newton_event_terms import push_robot as _push_robot_fn
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel, projected_gravity, dof_pos, dof_vel, prev_processed_actions,
)
from rlworld.rl.envs.mdp.observations.genesis.exteroception import command as command_obs
from rlworld.rl.envs.mdp.observations.newton import state
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf


@dataclass
class G1FlatNewtonConfig:
    # Robot configuration
    robot: G1MujocoConfig = field(default_factory=G1MujocoConfig)

    # Action component
    action_scale: Dict[str, float] = field(default_factory=lambda: G1_ACTION_SCALE)

    # Environment settings
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Simulation settings
    dt: float = 0.005
    decimation: int = 4
    substeps: int = 2

    # Command ranges
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-0.5, 0.5)

    # Algorithm settings
    algorithm_name: str = "PPO"
    max_iterations: int = 30000

    actor_class_name: str = "MLPActor"
    run_name: str = "G1_29dof_Newton"

    robot_foot_names = None

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

    def _build_env_config(self, quat) -> NewtonEnvConfig:
        @dataclass
        class _TerminationsCfg(TerminationsConfig):
            roll_pitch = TerminationTermConfig(
                common_tf.roll_pitch_violation,
                {"roll_threshold_degree": 20.0, "pitch_threshold_degree": 20.0},
            )
            max_episode = TerminationTermConfig(max_episode_exceed)

        return NewtonEnvConfig(
            num_envs=self.num_envs,
            env_name="NewtonEnv",
            task_name="G1_12Dof_Velocity_Tracking",
            seed=self.seed,
            episode_length_s=self.episode_length_s,
            decimation=self.decimation,
            terminations=_TerminationsCfg(),
        )

    def _build_scene_config(self, quat) -> NewtonSceneConfig:
        r = self.robot
        return NewtonSceneConfig(
            dt=self.dt,
            substeps=self.substeps,
            gravity=(0.0, 0.0, -9.81),
            solver_type="mujoco",
            robot_cfg=self.robot,
            entities={
                "ground": GroundPlaneCfg(
                    contact_stiffness=2.0e3,
                    contact_damping=1.0e2,
                    friction=1.0,
                    ground_kf=1.0e3,
                    ground_mu_rolling=0.0005,
                    ground_mu_torsional=0.25,
                ),
                "robot": NewtonEntityCfg(
                    urdf_path=r.urdf_path,
                    init_state=InitialStateCfg(
                        pos=(0.0, 0.0, r.base_init_height),
                        rot=(quat[0], quat[1], quat[2], quat[3]),
                        joint_pos=r.default_joint_angles,
                    ),
                    floating=True,
                    collapse_fixed_joints=True,
                    articulation=ArticulationCfg(
                        actuators=(
                            DelayedPDActuatorCfg(
                                target_names_expr=(".*",),
                                stiffness=r.p_gains,
                                damping=r.d_gains,
                                armature=r.armature,
                                min_delay=0,
                                max_delay=2
                            ),
                        ),
                    ),
                    body_label_prefix=r.name,
                    shape_cfg=newton.ModelBuilder.ShapeConfig(
                        ke=2.0e3, kd=1.0e2, kf=1.0e3, mu=1.0,
                    ),
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
        feet_bodies = tuple(self.robot.prefixed_foot_names)

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
            projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
            command = ObservationTermConfig(func=command_obs, scale=1.0)
            dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
            prev_actions = ObservationTermConfig(func=prev_processed_actions, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            base_height = ObservationTermConfig(func=state.base_height, scale=1.0)
            base_lin_vel_obs = ObservationTermConfig(func=state.base_lin_vel, scale=1.0)
            base_euler = ObservationTermConfig(func=state.base_euler, scale=1.0)
            feet_air_time = ObservationTermConfig(func=state.feet_air_time, scale=1.0,
                                                  params={"feet_bodies": feet_bodies})
            feet_contact_force = ObservationTermConfig(func=state.feet_contact_force, scale=0.01,
                                                       params={"feet_bodies": feet_bodies})
            feet_contact_indicator = ObservationTermConfig(func=state.feet_contact_indicator, scale=1.0,
                                                           params={"feet_bodies": feet_bodies})
            feet_height = ObservationTermConfig(func=state.feet_height, scale=1.0, params={"feet_bodies": feet_bodies})

        @dataclass
        class _ObsCfg(NewtonObservationConfig):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def _build_action_config(self) -> NewtonActionConfig:
        return NewtonActionConfig(
            actuated_dof_names=self.robot.prefixed_actuated_dof_patterns,
            action_scale=self.robot.prefixed_action_scale,
            clip_actions=(-100.0, 100.0),
            offset=self.robot.get_prefixed_action_offset(),
        )

    def _build_reward_config(self) -> RewardConfig:
        @dataclass
        class _RewardsCfg(RewardConfig):
            exponential_shaping: bool = True
            shaping_sigma: float = 0.25

            # Tracking rewards (common -- uses RobotData interface)
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

            # Orientation (common -- uses RobotData interface)
            flat_orientation = RewardTermConfig(
                func=rf_common.flat_orientation,
                weight=1.0,
                params={"std": 0.447},
            )

            # Posture (stateful class)
            variable_posture = RewardTermConfig(
                func=rf_mjlab.variable_posture,
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
            )

            # Penalties
            body_ang_vel_penalty_mjlab = RewardTermConfig(
                func=rf_mjlab.body_ang_vel_penalty_mjlab,
                weight=0.05,
                params={"body_name": self.robot.prefixed("torso_link")},
            )
            angular_momentum_penalty = RewardTermConfig(
                func=rf_mjlab.angular_momentum_penalty,
                weight=0.02,
            )
            joint_pos_limits_mjlab = RewardTermConfig(
                func=rf_mjlab.joint_pos_limits_mjlab,
                weight=1.0,
            )
            raw_action_rate_l2_mjlab = RewardTermConfig(
                func=rf_mjlab.raw_action_rate_l2_mjlab,
                weight=0.1,
            )

            # Feet rewards
            feet_clearance_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            )
            feet_swing_height_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_swing_height_mjlab,
                weight=0.25,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "target_height": 0.1,
                    "command_threshold": 0.05,
                },
            )
            feet_slip_mjlab = RewardTermConfig(
                func=rf_mjlab.feet_slip_mjlab,
                weight=0.1,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "command_threshold": 0.05,
                },
            )
            soft_landing_mjlab = RewardTermConfig(
                func=rf_mjlab.soft_landing_mjlab,
                weight=1e-5,
                params={
                    "feet_bodies": self.robot.prefixed_foot_names,
                    "command_threshold": 0.05,
                },
            )

        # # --- Previous reward config (without exponential shaping) ---
        # @dataclass
        # class _RewardsCfg(RewardConfig):
        #     track_lin_vel = RewardTermConfig(
        #         func=rf_common.track_lin_vel, weight=2.0,
        #         params={"std": 0.5, "penalize_z": True},
        #     )
        #     track_ang_vel = RewardTermConfig(
        #         func=rf_common.track_ang_vel, weight=2.0,
        #         params={"std": 0.707, "penalize_xy": True},
        #     )
        #     flat_orientation = RewardTermConfig(
        #         func=rf_common.flat_orientation, weight=1.0,
        #         params={"std": 0.447},
        #     )
        #     variable_posture = RewardTermConfig(
        #         func=rf_mjlab.variable_posture, weight=1.0,
        #         params={...},  # same posture params as above
        #     )
        #     body_ang_vel_penalty_mjlab = RewardTermConfig(
        #         func=rf_mjlab.body_ang_vel_penalty_mjlab, weight=0.05,
        #         params={"body_name": self.robot.prefixed("torso_link")},
        #     )
        #     angular_momentum_penalty = RewardTermConfig(
        #         func=rf_mjlab.angular_momentum_penalty, weight=0.02,
        #     )
        #     joint_pos_limits_mjlab = RewardTermConfig(
        #         func=rf_mjlab.joint_pos_limits_mjlab, weight=1.0,
        #     )
        #     raw_action_rate_l2_mjlab = RewardTermConfig(
        #         func=rf_mjlab.raw_action_rate_l2_mjlab, weight=0.1,
        #     )
        #     feet_clearance_mjlab = RewardTermConfig(
        #         func=rf_mjlab.feet_clearance_mjlab, weight=2.0,
        #         params={"feet_bodies": ..., "target_height": 0.1, "command_threshold": 0.05},
        #     )
        #     feet_swing_height_mjlab = RewardTermConfig(
        #         func=rf_mjlab.feet_swing_height_mjlab, weight=0.25,
        #         params={"feet_bodies": ..., "target_height": 0.1, "command_threshold": 0.05},
        #     )
        #     feet_slip_mjlab = RewardTermConfig(
        #         func=rf_mjlab.feet_slip_mjlab, weight=0.1,
        #         params={"feet_bodies": ..., "command_threshold": 0.05},
        #     )
        #     soft_landing_mjlab = RewardTermConfig(
        #         func=rf_mjlab.soft_landing_mjlab, weight=1e-5,
        #         params={"feet_bodies": ..., "command_threshold": 0.05},
        #     )

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

    def _build_event_config(self) -> EventConfig:
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)

        @dataclass
        class _EventsCfg(EventConfig):
            reset_base_pose = EventTermConfig(
                func=initf.initialize_base_pose,
                mode="reset",
                params={
                    "base_init_pos": [0.0, 0.0, self.robot.base_init_height],
                    "base_init_quat": [quat[0], quat[1], quat[2], quat[3]],
                }
            )
            reset_dof_pos = EventTermConfig(
                func=initf.initialize_dof_pos,
                mode="reset",
            )
            push_robot = EventTermConfig(
                func=_push_robot_fn,
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
