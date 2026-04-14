import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))

import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner

from typing import Dict
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig, StateInitializationTermConfig, CommandTermConfig

from rlworld.rl.envs.mdp.observations import proprioception, exteroception
from rlworld.rl.envs.mdp.rewards import reward_terms as rf
from rlworld.rl.envs.mdp.terminations import terminations as tf
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.commands import command_terms as cf

import gymnasium as gym
from rlworld.rl.envs import GymnasiumEnv
import genesis as gs

DEFAULT_ENV_CONFIG: Dict = {
    "env_name": "LocomotionEnv",
    "robot_type": "go2",
    "num_envs": 1,
    "use_lpf": False,
    "rotation_steps": 30000,
    "env_type": "plane",

    "sim_dt": 0.02,
    "control_dt": 0.02,
    "actuated_kp": 20.0,
    "actuated_kd": 0.5,
    "unactuated_kp": 1.0,
    "unactuated_kd": 0.01,
    "foot_stiffness": 5.,
    "foot_damping": 0.5,
    "friction": 0.5,

    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],

    "magnetic_links_name": [
        "planeLink",
        "Foot_link_FL_roll",
        "Foot_link_FR_roll",
        "Foot_link_RL_roll",
        "Foot_link_RR_roll"
    ],  # "Block_obj_baselink",

    "magnet_on_threshold": 0.0,
    "magnetic_force_magnitude": 400.0,  # N
    "min_contact_points": 3,  # 3
    "magnetic_coupling_factor": 0.1,
    "use_air_gap": False,

    "episode_length_s": 20.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": False,
    "clip_actions": 100.0,

    "concavity_threshold": 0.01,
    "max_convex_hull": 1000,
    "hausdorff_resolution": 2000,
    "preprocess_mode": "on",
    "preprocess_resolution": 50,

    "vis_opacity": 0.4,
    # "viewer_camera_pos": (-5.0, -4.0, 1.5),
    # "viewer_camera_lookat": (0.0, -4.0, 1.5),
    "viewer_camera_lookat": [0.0, 0.0, 2.5],
    "viewer_camera_pos": [3.0, 3.0, 5.0],

    "foot_contact_criteria": [],

    # ------------- Gait -------------

    "gait_period": 1.20,
    "gait_phase_offset": {  # Phase offsets for each foot (as fraction of period)
        "RR": 0.0,  # Rear right foot
        "RL": 1 / 4,  # Rear left foot
        "FR": 2 / 4,  # Front right foot
        "FL": 3 / 4,  # Front left foot
    }
}

DEFAULT_ACT_CONFIG: Dict = {
    # "num_actions": 12,
    "actuated_dof_names": ["FL.*", "RL.*", "FR.*", "RR.*"],
    "num_active_joint_actions": 12,
    "action_scale": 0.4,
    "simulate_action_latency": False,
    "clip_actions": (-100.0, 100.0),
    "offset": {
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }
}

DEFAULT_SCENE_CONFIG: Dict = {
    "entities": {
        "plane": {"morph": gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)},
        "robot": {
            "morph": gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", convexify=False),
            "visualize_contact": True
        }
    },

    "sensors": None
}

DEFAULT_CURRICULUM_CONFIG: Dict = {
    # Basic curriculum settings
    "enable": False,
    "initial_level": 0,
    "max_level": 3,

    # Level progression conditions
    "success_threshold": 0.8,
    "min_steps_per_level": 50000,  # Minimum steps required per level
    "eval_window_size": 2,  # Number of episodes to evaluate

    # Components that change with curriculum
    "curriculum_components": {
        "command_ranges": {
            0: {  # Initial level: forward walking only
                "lin_vel_x": (0.3, 0.3),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel": (0.0, 0.0)
            },
            1: {  # Intermediate level: expanded velocity ranges
                "lin_vel_x": (0.3, 0.5),
                "lin_vel_y": (-0.1, 0.1),
                "ang_vel": (-0.1, 0.1)
            },
            2: {  # Advanced level: wider ranges
                "lin_vel_x": (0.3, 0.8),
                "lin_vel_y": (-0.2, 0.2),
                "ang_vel": (-0.2, 0.2)
            },
            3: {  # Final level: maximum ranges
                "lin_vel_x": (0.3, 1.0),
                "lin_vel_y": (-0.3, 0.3),
                "ang_vel": (-0.3, 0.3)
            }
        },
        "magnetic_forces": {
            0: 500.0,  # Initially high force for stability
            1: 450.0,
            2: 400.0,
            3: 350.0  # Finally lower force for efficiency
        },
        "reward_scales": {
            0: {  # Initial focus on basic movement rewards
                "tracking_lin_vel": 0.5,
                "tracking_ang_vel": 0.1,
                "base_height": 15.0,
                "action_rate": 0.003
            },
            1: {  # Gradual increase in difficulty
                "base_height": 12.0,
                "action_rate": 0.004,
                "gait": 1.0,
                "foot_height": 1.0,
                "foot_slip": 0.3,
                "foot_clearance": 50.0,
                "first_order_action_smoothness": 10.0,
                "orientation": 3.5,
                "joint_torque": 0.003,
                "gait_flexible_constraint": 0.5
            },
            2: {
                "tracking_lin_vel": 0.9,
                "tracking_ang_vel": 0.18,
                "base_height": 1.0,
                "action_rate": 0.0045,
                "joint_jerk": 0.03,
                "first_order_action_smoothness": 2.5,
            },
            3: {  # Final reward structure for target behavior
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "gait": 0.5,
                "foot_height": 0.5,
                "foot_slip": 0.15,
                "foot_clearance": 140.0,
                "orientation": 3.0,
                "joint_torque": 0.003,
                "joint_position": 0.75,
                "joint_speed": 0.15,
                "joint_jerk": 0.03,
                "first_order_action_smoothness": "auto",
                "second_order_action_smoothness": "auto",
                "base_motion": 3.0,
                "lin_vel_z": 0.4,
                "base_height": 10.0,
                "action_rate": 0.005,
                "similar_to_default": 0.01,
            }
        },
        "stability_thresholds": {
            0: {
                "max_base_rotation_deg": 20,  # More lenient thresholds initially
                "max_joint_velocity": 5.0
            },
            1: {
                "max_base_rotation_deg": 25,
                "max_joint_velocity": 7.0
            },
            2: {
                "max_base_rotation_deg": 28,
                "max_joint_velocity": 8.0
            },
            3: {
                "max_base_rotation_deg": 30,  # More challenging thresholds in final level
                "max_joint_velocity": 10.0
            }
        }
    },

    # Criterion for incremental curriculum learning
    "criterion": {
        "tracking_lin_vel_xy": -100,
        "mean_return": -100
    }
}

DEFAULT_OBS_CONFIG: Dict = {
    "obs_type": "current",

    "use_vision": False,
    "map_size": 64,
    "map_resolution": 0.1,
    "camera_fov": 60,
    "base_lookat": [0.0, -1.0, 0.0],
    "cam_base_offset": [0.0, -0.23, 0.0],
    "cam_GUI": True,
    "cam_resolution": [640, 480],

    "obs_scales": {
        "lin_vel_x": 2.0,
        "lin_vel_y": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    },
}

DEFAULT_REWARD_CONFIG: Dict = {
    "tracking_sigma": 0.2,
    "base_height_target": 0.3,
    "feet_height_target": 0.5,
}

DEFAULT_COMMAND_CONFIG: Dict = {
    "num_commands": 4,
}

DEFAULT_ALGORITHM_CONFIG: Dict = {
    "clip_param": 0.2,
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "lam": 0.95,
    "actor_learning_rate": 0.001,
    "critic_learning_rate": 0.001,
    "estimator_learning_rate": 5e-4,
    "max_grad_norm": 1.0,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "schedule": "adaptive",
    "use_clipped_value_loss": True,
    "value_loss_coef": 1.0,
    "use_truth_value_for_actor": True,
    "use_truth_value_for_critic": True,
    "use_barrier_style": False,
}

DEFAULT_NN_CONFIG: Dict = {
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64, 64],
        "init_noise_std": 1.0,
        "std_type": "fixed"
    },
    "state_estimator": {
        "activation": "relu",
        "hidden_dims": [256, 128, 64]
    }
}

DEFAULT_RUNNER_CONFIG: Dict = {
    "algorithm_class_name": "PPO",
    "checkpoint": -1,
    "experiment_name": "GoAnywhere",
    "load_run": None,
    "log_interval": 1,
    "max_iterations": 10000,
    "init_at_random_ep_len": False,
    "num_steps_per_env": 24,
    "policy_class_name": "RodriguesActorCritic",
    "state_estimator_class_name": "StateEstimator",
    "record_interval": -1,
    "resume": False,
    "resume_path": None,
    "run_name": "rod_gym_stand",
    "logger": 'wandb',
    "wandb_project": "legged_gym",
    "runner_class_name": "runner_class_name",
    "save_interval": 1000,
    "save_path": "auto"
}


def main():
    # Generate base configs and apply command line overrides

    obs_terms = [
        ObservationTermConfig(proprioception.imu_ang_vel, scale=2.0, history_length=4),
        ObservationTermConfig(proprioception.projected_gravity, scale=1.0),
        ObservationTermConfig(exteroception.command, scale=1.0),
        ObservationTermConfig(proprioception.dof_pos_nominal_difference, scale=1.0),
        ObservationTermConfig(proprioception.dof_vel, scale=0.05, history_length=4),
        ObservationTermConfig(proprioception.prev_processed_actions, scale=1.0)
    ]

    state_init_terms = [
        StateInitializationTermConfig(initf.initialize_dof_pos),
        StateInitializationTermConfig(initf.initialize_pos_quat_on_terrain)
    ]

    termination_terms = [
        TerminationTermConfig(tf.roll_pitch_violation, {"roll_threshold_degree": 45.0, "pitch_threshold_degree": 45.0}),
        TerminationTermConfig(tf.max_episode_exceed),
        TerminationTermConfig(tf.out_of_terrain_bounds)
    ]

    DEFAULT_OBS_CONFIG["obs_group"] = {
        "actor": obs_terms,
        "critic": obs_terms
    }

    DEFAULT_ENV_CONFIG["termination_criteria"] = termination_terms
    DEFAULT_ENV_CONFIG["state_init_terms"] = state_init_terms

    reward_terms = {
        "tracking_lin_vel": RewardTermConfig(rf.tracking_lin_vel, weight=1.0),
        "tracking_ang_vel": RewardTermConfig(rf.tracking_ang_vel, weight=0.2),
        "lin_vel_z": RewardTermConfig(rf.lin_vel_z, weight=1.0),
        "action_rate": RewardTermConfig(rf.action_rate, weight=0.005),
        "similar_to_default": RewardTermConfig(rf.similar_to_default, weight=0.1),
    }

    DEFAULT_REWARD_CONFIG["reward_terms"] = reward_terms

    from rlworld.rl.configs.scene import GenesisSceneInitConfig, EntityConfig
    sim_dt = 0.02
    control_dt = 0.02
    sim_options = gs.options.SimOptions(
        dt=control_dt,
        substeps=int(control_dt / sim_dt)
    )
    rigid_options = gs.options.RigidOptions(
        dt=control_dt,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_self_collision=True,
        enable_joint_limit=True
    )

    DEFAULT_SCENE_CONFIG["sim_options"] = sim_options
    DEFAULT_SCENE_CONFIG["rigid_options"] = rigid_options

    entities = [
        EntityConfig(
            entity_name="base_entity",
            morph=gs.morphs.Terrain(
                n_subterrains=(4, 4),
                subterrain_size=(8.0, 8.0),
                horizontal_scale=0.2,
                vertical_scale=0.01,
                subterrain_types=[
                    ["flat_terrain", "stairs_terrain", "flat_terrain", "discrete_obstacles_terrain"],
                    ["pyramid_sloped_terrain", "wave_terrain", "random_uniform_terrain", "pyramid_stairs_terrain"],
                    ["flat_terrain", "sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"],
                    ["pyramid_stairs_terrain", "random_uniform_terrain", "pyramid_sloped_terrain", "stairs_terrain"],
                ],
                subterrain_parameters={
                    "stairs_terrain": {
                        "step_width": 0.6,
                        "step_height": -0.12,
                    },
                    "pyramid_stairs_terrain": {
                        "step_width": 0.6,
                        "step_height": -0.1,
                    },
                    "random_uniform_terrain": {
                        "min_height": -0.15,
                        "max_height": 0.15,
                        "step": 0.05,
                        "downsampled_scale": 0.5,
                    },
                    "sloped_terrain": {
                        "slope": -0.4,
                    },
                    "pyramid_sloped_terrain": {
                        "slope": -0.15,
                    },
                    "discrete_obstacles_terrain": {
                        "max_height": 0.1,
                        "min_size": 0.6,
                        "max_size": 2.5,
                        "num_rects": 25,
                    },
                    "wave_terrain": {
                        "num_waves": 3.0,
                        "amplitude": 0.12,
                    },
                },
            )
        ),
        EntityConfig(
            entity_name="robot",
            morph=gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", convexify=False),
            visualize_contact=True,
            p_gain={"FL.*": 20.0, "FR.*": 20.0, "RL.*": 20.0, "RR.*": 20.0},
            d_gain={"FL.*": 0.5, "FR.*": 0.5, "RL.*": 0.5, "RR.*": 0.5},
        )
    ]

    from rlworld.rl.configs.sensors import SensorConfig
    sensors = [
        SensorConfig(entity_name="robot", sensor_class=gs.sensors.IMU)
    ]

    DEFAULT_SCENE_CONFIG["entities"] = entities
    DEFAULT_SCENE_CONFIG["sensors"] = sensors

    command_terms = [
        CommandTermConfig(cf.lin_vel_x, params={"range": (-0.5, 0.5)}),
        CommandTermConfig(cf.lin_vel_y, params={"range": (-0.5, 0.5)}),
        CommandTermConfig(cf.ang_vel, params={"range": (-0.4, 0.4)}),
        CommandTermConfig(cf.base_height, params={"range": (0.3, 0.3)})
    ]

    DEFAULT_COMMAND_CONFIG["sampler"] = command_terms

    configs_dict = {
        "env": DEFAULT_ENV_CONFIG,
        "scene": DEFAULT_SCENE_CONFIG,
        "observation": DEFAULT_OBS_CONFIG,
        "action": DEFAULT_ACT_CONFIG,
        "reward": DEFAULT_REWARD_CONFIG,
        "command": DEFAULT_COMMAND_CONFIG,
        "algorithm": DEFAULT_ALGORITHM_CONFIG,
        "nn": DEFAULT_NN_CONFIG,
        "runner": DEFAULT_RUNNER_CONFIG
    }

    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    # runner = OnPolicyRunner.create_with_env(cfgs_for_run, show_viewer=False)
    env = gym.make_vec("HumanoidStandup-v5", num_envs=cfgs_for_run.env.num_envs, max_episode_steps=1000)
    env = GymnasiumEnv(env, seed=cfgs_for_run.env.seed)

    runner = OnPolicyRunner(
        env=env, cfgs=cfgs_for_run, use_wandb=True
    )

    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
