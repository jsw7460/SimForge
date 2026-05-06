"""Unified Unitree G1 motion tracking config.

Sim-agnostic port of Mjlab's ``Mjlab-Tracking-Flat-Unitree-G1`` task
(``Mjlab/src/mjlab/tasks/tracking/config/g1/``). Parameters mirror
Mjlab's ``tracking_env_cfg.py`` + ``g1/env_cfgs.py`` + ``g1/rl_cfg.py``
so training should reproduce the Mjlab reference behaviour.

Key differences vs JaxRLWorld's existing ``g1_29dof`` locomotion
preset:
  - ``CommandConfig`` has a single ``MotionCommandCfg`` term
    (motion-tracking reference, not velocity command)
  - terminations: tracking-specific (``bad_anchor_pos_z_only``,
    ``bad_anchor_ori``, ``bad_motion_body_pos_z_only``) + max_episode
  - rewards: 6 exponential tracking terms + 3 smoothness penalties
  - observations: anchor-relative motion reference in actor + body
    poses in critic (privileged info)
  - action: ``JointPositionAction`` with per-joint ``G1_ACTION_SCALE``
    and ``offset=default_joint_angles`` (matches Mjlab exactly)
  - event: DR terms only (no reset event — motion owns initial state)
  - episode: 10 s (Mjlab default)

Usage::

    from rlworld.rl.configs.presets.g1_tracking.base import G1TrackingConfig
    cfgs = G1TrackingConfig(
        sim_type="newton",
        motion_files=("/tmp/g1_walk.npz",),
    ).build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    CommandConfig,
    EventConfig,
    NNConfig,
    PPOPolicyConfig,
    RunnerConfig,
)
from rlworld.rl.configs.curriculums import CurriculumManagerConfig
from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig
from rlworld.rl.envs.mdp.commands import MotionCommandCfg

_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "mujoco": {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "G1_Tracking_Newton",
    "genesis": "G1_Tracking_Genesis",
    "mujoco": "G1_Tracking_Mujoco",
}


def _get_sim_builders(sim_type: str):
    if sim_type == "newton":
        from . import _newton_builders as mod
    elif sim_type == "genesis":
        from . import _genesis_builders as mod
    elif sim_type == "mujoco":
        from . import _mujoco_builders as mod
    else:
        raise ValueError(f"Unknown sim_type: {sim_type!r}. Expected one of {sorted(_SIM_TIMINGS)}.")
    return mod


@dataclass
class G1TrackingConfig:
    """Unified base config for Unitree G1 motion tracking."""

    sim_type: str = "newton"
    robot: G1MujocoConfig = field(default_factory=G1MujocoConfig)

    # Environment / training.
    num_envs: int = 4096
    # Mjlab's tracking_env_cfg.py uses 10s episodes for G1.
    episode_length_s: float = 10.0
    seed: int = 42

    # Motion source: tuple of NPZ paths (length-1 for single-clip, length
    # >= 2 for multi-motion). Each episode reset samples one clip per env
    # (uniform by default) and keeps it for the rest of the episode.
    motion_files: tuple[str, ...] = ("JaxRLWorld/rlworld/assets/motions/gangnam_style/G1_gangnam_style_V01.npz",)
    motion_weights: tuple[float, ...] | None = None

    # ── Body list (Mjlab G1 tracking config/g1/env_cfgs.py) ───────────
    # Anchor body is the shared torso link; the first entry of body_names
    # is the floating-base body (pelvis on G1) whose pose/velocity seeds
    # the root state written on reset.
    anchor_body_name: str = "torso_link"
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    # End-effector subset used by ``bad_motion_body_pos_z_only`` termination.
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    # ── Sampling + RSI (Mjlab tracking_env_cfg.py defaults) ───────────
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
    pose_range: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
    )
    velocity_range: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
    )
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # ── Reward weights / std (Mjlab tracking_env_cfg.py lines 209-251) ─
    anchor_pos_weight: float = 0.5
    anchor_pos_std: float = 0.3
    anchor_ori_weight: float = 0.5
    anchor_ori_std: float = 0.4
    body_pos_weight: float = 1.0
    body_pos_std: float = 0.3
    body_ori_weight: float = 1.0
    body_ori_std: float = 0.4
    body_lin_vel_weight: float = 1.0
    body_lin_vel_std: float = 1.0
    body_ang_vel_weight: float = 1.0
    body_ang_vel_std: float = 3.14

    # Smoothness / safety penalties (Mjlab sign-flipped — positive weight
    # on positive-valued penalty fn would subtract from reward; using
    # positive weight on signed fn to match mjlab's |weight| magnitudes).
    action_rate_l2_weight: float = 0.1
    joint_pos_limits_weight: float = 10.0
    self_collision_weight: float = 10.0

    # ── Termination thresholds (Mjlab tracking_env_cfg.py lines 257-279)
    bad_anchor_pos_z_threshold: float = 0.25
    bad_anchor_ori_threshold: float = 0.8
    bad_motion_body_pos_z_threshold: float = 0.25

    # ── Algorithm (Mjlab rl_cfg.py unitree_g1_tracking_ppo_runner_cfg) ─
    algorithm_name: str = "PPO"
    max_iterations: int = 30_000

    run_name: str | None = None

    def build(self):
        if not self.motion_files:
            raise ValueError(
                "G1TrackingConfig.motion_files is empty. Provide at least "
                "one NPZ path via the preset default, CLI override, or a "
                "subclass before calling build()."
            )
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        kwargs: Dict[str, Any] = dict(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
            observation=builders.build_observation(self),
            action=builders.build_action(self),
            reward=builders.build_reward(self),
            command=self._build_command_config(builders),
            event=self._build_event_config(builders),
            curriculum=CurriculumManagerConfig(),
            algorithm=self._build_algorithm_config(),
            nn=self._build_nn_config(),
            runner=self._build_runner_config(),
        )

        cfgs = builders.CONFIGS_FOR_RUN_CLS(**kwargs)
        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        cfgs.preset_kwargs = self._get_preset_kwargs()
        return cfgs

    def _get_preset_kwargs(self) -> Dict[str, Any]:
        from dataclasses import MISSING, fields

        kwargs: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "robot":
                continue
            value = getattr(self, f.name)
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                kwargs[f.name] = value
                continue
            if value != default:
                kwargs[f.name] = value
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        return self.build().recursive_to_dict()

    def _build_command_config(self, builders) -> CommandConfig:
        return CommandConfig(
            terms={
                "motion": MotionCommandCfg(
                    motion_files=self.motion_files,
                    motion_weights=self.motion_weights,
                    anchor_body_name=self.anchor_body_name,
                    body_names=self.body_names,
                    entity_name="robot",
                    pose_range=self.pose_range,
                    velocity_range=self.velocity_range,
                    joint_position_range=self.joint_position_range,
                    adaptive_kernel_size=self.adaptive_kernel_size,
                    adaptive_lambda=self.adaptive_lambda,
                    adaptive_uniform_ratio=self.adaptive_uniform_ratio,
                    adaptive_alpha=self.adaptive_alpha,
                    sampling_mode=self.sampling_mode,
                ),
            },
        )

    def _build_event_config(self, builders) -> EventConfig:
        """DR-only — motion command owns the initial state."""
        cfg = EventConfig()
        for name, term in builders.build_dr_terms(self).items():
            setattr(cfg, name, term)
        return cfg

    def _build_algorithm_config(self) -> PPOConfig:
        # Hyperparameters from Mjlab rl_cfg.py unitree_g1_tracking_ppo_runner_cfg.
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            obs_normalization=True,
            use_early_stop=False,
            desired_kl=0.01,
            entropy_coef=0.005,
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
                actor_class_name="MLPActor",
                actor_kwargs={
                    "activation": "elu",
                    "ortho_init": True,
                    "output_gain": 1.0,
                    "hidden_dims": [512, 256, 128],
                },
                critic_kwargs={
                    "activation": "elu",
                    "ortho_init": True,
                    "output_gain": 1.0,
                    "hidden_dims": [512, 256, 128],
                },
                init_noise_std=1.0,
                distribution_type="gaussian",
                std_type="state_independent",
            ),
        )

    def _build_runner_config(self) -> RunnerConfig:
        run_name = self.run_name or _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        return RunnerConfig(
            checkpoint=-1,
            log_interval=1,
            max_iterations=self.max_iterations,
            # Mjlab G1 tracking uses num_steps_per_env=24, save_interval=500.
            init_at_random_ep_len=True,
            resume=False,
            resume_path=None,
            run_name=run_name,
            logger="wandb",
            wandb_project="G1_Tracking",
            save_interval=500,
            output_dir="auto",
        )
