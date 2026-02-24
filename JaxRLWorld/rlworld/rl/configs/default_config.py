import math
from typing import Dict



DEFAULT_ENV_CONFIG: Dict = {
    "env_name": "World",        # World, LocomotionEnv, Maniskill
    "task_name": "Unknown",      # Only used for Gymnasium style environments ("e.g.: InvertedPendulum-v5")
    "num_envs": 10000,
    "seed": 42,

    "termination_criteria": [],
    "base_init_pos": [1.5, 1.5, 0.15],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],

    "episode_length_s": 20.0,
}


DEFAULT_SCENE_CONFIG: Dict = {
    "env_spacing": (20.0, 20.0),
    "entities": [],
    "sensors": [],
}


DEFAULT_ACT_CONFIG: Dict = {
    "actuated_dof_names": [""],
    "num_joint_actions": 12,
    "action_scale": 0.4,
    "simulate_action_latency": False,
    "clip_actions": (-100.0, 100.0),
    "offset": {

    }
}


DEFAULT_EVENT_CONFIG: Dict = {}


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
    "robot_state_dim": 3,
    "state_init_terms": [],
    "max_history_len": 66,  # used only when obs_type=="dual"
    "short_history_len": 4,  # used only when obs_type=="dual"
    "obs_type": "current",

    "use_vision": False,
    "use_height_map": False,
    "map_size": 64,
    "horizontal_scale": 0.03,
    "map_resolution": 0.1,
    "camera_fov": 60,
    "base_lookat": [0.0, -1.0, 0.0],
    "cam_base_offset": [0.0, -0.23, 0.0],
    "cam_GUI": False,
    "cam_resolution": [640, 480],
}

DEFAULT_REWARD_CONFIG: Dict = {
    "tracking_sigma": 0.25,
    "cf_sigma": 0.5,
    "base_height_target": 0.2,
    "feet_height_target": 0.4,
    "reward_scales": {
        "tracking_lin_vel": 3.0,  # 1.0,
        # "tracking_ang_vel": 3.0, # 0.2,
        "airtime": 1.5,
        # "tracking_lin_vel": 1.0, # 1.0,
        # "tracking_ang_vel": 0.3, # 0.2,
        "gait": 2.0,
        # "foot_height": 0.5,
        # "foot_slip": 6e-2,
        # "num_contacted_feet": 1.0,
        # "foot_clearance": 100.0,
        "orientation": 3.0,
        # "joint_torque": 0.003,
        "joint_position": 0.1,
        "base_height": 5.0,
        # "action_rate": 0.005,
        # "similar_to_default": 0.01,

        # "gait_flexible_constraint": 0.1,
        # "foot_height_constraint": 0.1,
        # "joint_position_constraint": 0.1,
        # "num_contacted_feet_constraint": 0.02
    },
    "reward_groups": {
        "linear_components": [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "base_height",
            "action_rate",
            "similar_to_default",
            "airtime"
            "gait",
            "num_contacted_feet"

            "gait_flexible_constraint",
            "foot_height_constraint"
            "joint_position_constraint",
            "num_contacted_feet_constraint"
        ],
        "exp_components": [
            "foot_height",
            "foot_slip",
            "foot_clearance",
            "orientation",
            "joint_torque",
            "joint_position",
            "joint_speed",
            "joint_jerk",
            "first_order_action_smoothness",
            "second_order_action_smoothness",
            "base_motion",
            "action_magnet",
        ]
    },
    "constraint_params": {
        "gait_flexible_constraint": {
            "d_lower": -0.6,
            "delta": 0.1
        },
        "foot_height_constraint": {
            "d_lower": -0.08,
            "delta": 0.01
        },
        "joint_position_constraint": {
            "yaw_d_lower": -math.pi / 6,
            "yaw_d_upper": math.pi / 6,
            "hip_d_lower": -math.pi / 4,
            "hip_d_upper": math.pi / 4,
            "knee_d_lower": -2 * math.pi / 5,
            "knee_d_upper": math.pi / 4,
            "delta": 0.08
        },
        "num_contacted_feet_constraint": {
            "d_lower": 2.01,
            "d_upper": 3.99,
            "delta": 0.1
        },
        "action_magnet_constraint": {
            "d_lower": -0.05,
            "d_upper": 0.05,
            "delta": 0.01
        }
    },
    "auto_scaling_rewards": {}
}

DEFAULT_COMMAND_CONFIG: Dict = {
    "num_commands": 7,
    "resampling_time_s": (8.0, 12.0),
    "sampler": (),
    "ranges": {
        "lin_vel_x": [-0.5, 0.5],  # m/s
        "lin_vel_y": [-0.5, 0.5],  # m/s
        # "ang_vel_range": [-0.01, 0.01],   # rad/s
        # "base_height_range": [0.25, 0.35],
        # "foot_height_range": [0.3, 0.5],
        # "roll_range": [-5.0, 5.0],    # Degree
        # "pitch_range": [-5.0, 5.0],   # Degree
    }
}

DEFAULT_ALGORITHM_CONFIG: Dict = {
    "clip_param": 0.2,
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "lam": 0.95,
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    "estimator_learning_rate": 5e-4,
    "max_grad_norm": 0.5,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "schedule": "adaptive",
    "use_clipped_value_loss": False,
    "value_loss_coef": 1.0,
    "use_truth_value_for_actor": False,
    "use_truth_value_for_critic": True,
    "use_barrier_style": False,
    "use_sde": True,
    "sde_sample_freq": 100,
    "learning_starts": 10_000,
    "num_steps_per_env": 24,
}

DEFAULT_NN_CONFIG: Dict = {
    "policy": {
        "actor_class": None,
        "activation": "tanh",
        "actor_hidden_dims": [128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 1.0,
        "distribution_type": "gaussian",
        "std_type": "fixed"
    },
    "state_estimator": {
        "activation": "relu",
        "hidden_dims": [256, 128, 64]
    }
}

DEFAULT_RUNNER_CONFIG: Dict = {
    "checkpoint": -1,
    "experiment_name": "GoAnywhere",
    "load_run": None,
    "log_interval": 1,
    "max_iterations": 99999,
    "init_at_random_ep_len": False,
    "num_steps_per_env": 24,
    "policy_class_name": "PPOActorCritic",
    "state_estimator_class_name": "StateEstimator",
    "low_level_path": None,
    "high_level_update_freq": 1,
    "record_interval": -1,
    "resume": False,
    "resume_path": None,
    "run_name": "",
    "logger": 'wandb',
    "wandb_project": "RLArchitecture",
    "runner_class_name": "runner_class_name",
    "save_interval": 1000,
    "save_path": "auto"
}
