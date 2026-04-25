"""Unified T1 humanoid motion-tracking config.

Single source of truth for the T1 motion tracking task across Newton,
Genesis, and MuJoCo. Sim-agnostic port of Mjlab's
``tasks/tracking/tracking_env_cfg.py``.

Key differences vs ``t1_getup``:
  - ``CommandConfig`` has a single ``MotionCommandCfg`` term named
    ``"motion"`` — this is what owns episode initial state (no
    ``reset_fallen_or_standing`` event)
  - terminations are tracking-specific (``bad_anchor_pos_z_only``,
    ``bad_anchor_ori``, ``bad_motion_body_pos_z_only``) plus the
    standard ``max_episode`` timeout
  - rewards are the 6 exponential tracking terms from Mjlab
  - observations expose the anchor-relative motion reference to the
    policy (actor) and the robot's body poses to the critic
  - action uses ``SettleRelativeJointPositionAction`` with
    ``settle_steps = 0`` so residuals are computed against the
    motion-written reference pose

Usage::

    from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig
    cfgs = T1TrackingConfig(
        sim_type="newton",
        motion_files=("/tmp/t1_walk.npz",),
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
from rlworld.rl.configs.robots.t1 import T1Config
from rlworld.rl.envs.mdp.commands import MotionCommandCfg
from rlworld.rl.envs.mdp.configs.curriculums import CurriculumManagerConfig


# ── Per-simulator timing (same as t1_getup) ──────────────────────────
_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton":  {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "mujoco":  {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton":  "T1_Tracking_Newton",
    "genesis": "T1_Tracking_Genesis",
    "mujoco":  "T1_Tracking_Mujoco",
}


def _get_sim_builders(sim_type: str):
    """Lazy-import the simulator-specific builders module."""
    if sim_type == "newton":
        from . import _newton_builders as mod
    elif sim_type == "genesis":
        from . import _genesis_builders as mod
    elif sim_type == "mujoco":
        from . import _mujoco_builders as mod
    else:
        raise ValueError(
            f"Unknown sim_type: {sim_type!r}. "
            f"Expected one of {sorted(_SIM_TIMINGS)}."
        )
    return mod


@dataclass
class T1TrackingConfig:
    """Unified base configuration for Booster T1 motion tracking."""

    # Simulator selection.
    sim_type: str = "newton"

    # Robot.
    robot: T1Config = field(default_factory=T1Config)

    # Environment / training.
    num_envs: int = 4096
    episode_length_s: float = 10.0
    seed: int = 42

    # Motion source: one or more NPZ clips produced by booster_to_npz /
    # csv_to_npz. Each episode reset samples one clip per env (per
    # ``motion_weights``, uniform by default) and keeps it for the rest
    # of the episode. For single-clip experiments use a length-1 tuple.
    #
    # Default targets multi-motion tracking over the nine Booster T1 soccer
    # clips our ``booster_to_npz`` adapter can convert end to end
    # (``kick_ball1`` / ``walking2`` use a different recording schema and
    # need a separate adapter path — intentionally omitted here).
    motion_files: tuple[str, ...] = (
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/goal_kick.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/jogging.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/kick_ball2.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/kick_ball3.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/pass_ball1.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/powerful_kick.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/running.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/soccer_drill_run.npz",
        "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/walking1.npz",
    )
    motion_weights: "tuple[float, ...] | None" = None

    # Body list tracked by rewards / observations / terminations.
    # Must exist in both the NPZ's ``body_names`` (bare names, from the
    # preprocessor's MuJoCo model) and the simulator's body namespace
    # (which Newton prefixes with ``"T1/"``).
    anchor_body_name: str = "Trunk"
    body_names: tuple[str, ...] = (
        "Trunk",
        "Waist",
        "left_foot_link",
        "right_foot_link",
        "left_hand_link",
        "right_hand_link",
    )

    # End-effector bodies for the ``bad_motion_body_pos_z_only``
    # termination. Subset of ``body_names``.
    ee_body_names: tuple[str, ...] = (
        "left_foot_link",
        "right_foot_link",
        "left_hand_link",
        "right_hand_link",
    )

    # Sampling mode. Default is ``"uniform"`` to be compatible with the
    # multi-motion ``motion_files`` default above — ``MotionCommand``
    # deliberately disallows ``"adaptive"`` in multi-motion mode (the
    # failure-weighted bins are defined per single clip, not across clips).
    # Override to ``"adaptive"`` when running a single-clip experiment.
    sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"

    # RSI ranges (Mjlab defaults for humanoid locomotion tracking).
    pose_range: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        },
    )
    velocity_range: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        },
    )
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    # Adaptive-sampling hyperparameters (Mjlab defaults).
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # ── Reward weights and exp-shaping std values ─────────────────────
    # Values come from Mjlab's tracking_env_cfg.py rewards section.
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

    # Smoothness / safety penalties (positive magnitude on negative-valued
    # reward fns → same sign as mjlab).
    action_rate_l2_weight: float = 0.1
    joint_pos_limits_weight: float = 10.0
    self_collision_weight: float = 10.0

    # ── Termination thresholds ────────────────────────────────────────
    bad_anchor_pos_z_threshold: float = 0.25   # meters
    bad_anchor_ori_threshold: float = 0.8       # ~46° projected-gravity mismatch
    bad_motion_body_pos_z_threshold: float = 0.25  # meters

    # Action processing. settle_steps=0 disables the hold period so the
    # policy can act on the motion-written initial pose immediately.
    action_scale: float = 0.25
    settle_steps: int = 0

    # ── Policy architecture (SpaceTimeTransformer) ────────────────────
    # When future_offsets is non-empty the motion command exposes a
    # sparse preview of upcoming reference frames to the policy, and the
    # tracking preset swaps in the SpaceTimeTransformer actor/critic.
    # Set to an empty tuple to fall back to the MLP pipeline.
    future_offsets: tuple[int, ...] = (1, 2, 4, 8, 16)
    ref_feature_dim: int = 9
    transformer_embed_dim: int = 64
    transformer_num_heads: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 128

    # Algorithm.
    algorithm_name: str = "PPO"
    max_iterations: int = 30000

    # Run name (None → auto from sim_type).
    run_name: str | None = None

    # ── Build entry point ─────────────────────────────────────────────
    def build(self):
        if not self.motion_files:
            raise ValueError(
                "T1TrackingConfig.motion_files is empty. Provide at least "
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
        from dataclasses import fields, MISSING
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

    # ── Shared build methods ──────────────────────────────────────────
    def _build_command_config(self, builders) -> CommandConfig:
        """Motion command — tracks the clips in ``motion_files`` (length-1
        tuple for single-clip, length >= 2 for multi-motion)."""
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
                    future_offsets=self.future_offsets,
                ),
            },
        )

    def _build_event_config(self, builders) -> EventConfig:
        """DR terms only — motion command handles initial state."""
        cfg = EventConfig()
        for name, term in builders.build_dr_terms(self).items():
            setattr(cfg, name, term)
        return cfg

    def _build_algorithm_config(self) -> PPOConfig:
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            obs_normalization=False,
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
            # 4 is fine again now that the encoder uses gradient
            # checkpointing — activation memory stays under budget even at
            # num_envs=4096 with the factorized attention layer stack.
            num_mini_batches=8,
            num_steps_per_env=8,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
            use_reward_scaling=False,
        )

    def _build_nn_config(self) -> NNConfig:
        if self.future_offsets:
            transformer_kwargs = {
                "tracked_body_names": self.body_names,
                "future_offsets": self.future_offsets,
                "ref_feature_dim": self.ref_feature_dim,
                "embed_dim": self.transformer_embed_dim,
                "num_heads": self.transformer_num_heads,
                "num_layers": self.transformer_num_layers,
                "dim_feedforward": self.transformer_dim_feedforward,
                "use_kinematic_mask": True,
            }
            return NNConfig(
                policy=PPOPolicyConfig(
                    actor_class_name="SpaceTimeTransformerActor",
                    critic_class_name="SpaceTimeTransformerCritic",
                    actor_kwargs=transformer_kwargs,
                    critic_kwargs=transformer_kwargs,
                    init_noise_std=1.0,
                    distribution_type="gaussian",
                    std_type="state_independent",
                ),
            )
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
            init_at_random_ep_len=True,
            resume=False,
            resume_path=None,
            run_name=run_name,
            logger="wandb",
            wandb_project="T1_Tracking",
            save_interval=500,
            output_dir="auto",
        )
