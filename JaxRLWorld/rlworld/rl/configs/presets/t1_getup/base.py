"""Unified T1 humanoid fall-recovery (getup) config.

Single source of truth for the T1 getup task across Newton, Genesis,
and MuJoCo. Inspired by ``mjlab_playground/getup/getup_env_cfg.py`` but
reimplemented on top of JaxRLWorld's cross-sim RobotData / act_manager
abstractions.

Compared to ``g1_29dof/base.py`` this preset:
  - uses ``reset_fallen_or_standing`` instead of uniform root reset
  - has no velocity command (``CommandConfig(terms={})``)
  - omits the ``roll_pitch_violation`` termination (the robot **must**
    be allowed to lie down without dying)
  - registers ``act_manager.settle_steps`` so the first ~1 s after a
    fallen reset is held at the landing pose
  - uses the getup reward terms from ``rewards/common/getup.py``:
    orientation_upright, height_to_target (trunk + waist), and
    GatedPostureTracker
  - shorter episode length (6 s, matching mjlab_playground)

Curriculum, energy termination, and encoder-bias DR are deferred to
Phase I/J (after the skeleton MVP converges).

Usage:
    from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
    cfgs_for_run = T1GetupConfig(sim_type="newton").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    CommandConfig,
    EventConfig,
    NNConfig,
    PPOPolicyConfig,
    RunnerConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.robots.t1 import T1Config
from rlworld.rl.envs.mdp.configs.curriculums import (
    CurriculumManagerConfig,
    CurriculumTermConfig,
)
from rlworld.rl.envs.mdp.curriculums import (
    reward_curriculum,
    termination_curriculum,
)
from rlworld.rl.envs.mdp.events import common as common_ef


# ── Per-simulator constants ──────────────────────────────────────────
# 50 Hz control (decimation 4 × dt 0.005 s = 0.02 s).

_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton":  {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "mujoco":  {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton":  "T1_Getup_Newton",
    "genesis": "T1_Getup_Genesis",
    "mujoco":  "T1_Getup_Mujoco",
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


# ── Unified config ───────────────────────────────────────────────────


@dataclass
class T1GetupConfig:
    """Unified base configuration for Booster T1 fall-recovery.

    Set ``sim_type`` to choose the simulator backend. Per-simulator
    details (scene layout, contact sensors, DR term set) are delegated
    to ``_{sim}_builders`` modules at build time; everything else — the
    reward config, event config, command config, observation layout,
    PPO hyperparameters, NN architecture — lives here and is shared.
    """

    # Simulator selection.
    sim_type: str = "newton"

    # Robot configuration.
    robot: T1Config = field(default_factory=T1Config)

    # Environment / training settings.
    num_envs: int = 4096
    episode_length_s: float = 6.0
    seed: int = 42

    # Reset event parameters (passed to reset_fallen_or_standing).
    fallen_prob: float = 0.6
    fall_height: float = 0.8
    fall_velocity_range: tuple[float, float] = (-0.5, 0.5)
    # "soft_limit" → uniform over full soft joint limit range
    # (mjlab_playground default). A ``(lo, hi)`` tuple switches back
    # to additive noise around the default pose.
    fall_joint_noise_range: "tuple[float, float] | str" = "soft_limit"
    standing_z_offset: float = 0.02

    # Action settling: hold joint position for the first 50 control
    # steps (= 1 s at 50 Hz) after each reset so the robot can settle
    # after a drop/impact before the policy's commands take effect.
    settle_steps: int = 30

    # Uniform per-step action scale. mjlab_playground T1 getup uses a
    # single scalar 0.6 across every joint (getup_env_cfg.py:89), NOT
    # the locomotion-convention ``0.25 * effort / stiffness`` per-joint
    # dict. Keeping this in sync across Newton / Genesis / MuJoCo.
    action_scale: float = 0.6

    # Getup reward parameters.
    orientation_std: float = 0.707  # ≈ exp(-2 * err)
    # mjlab_playground T1: _TORSO_HEIGHT=0.67, _WAIST_HEIGHT=0.55
    # (see getup/config/t1/env_cfgs.py lines 17-18).
    trunk_desired_height: float = 0.67
    waist_desired_height: float = 0.55
    posture_gate_threshold: float = 0.01

    # Reward weights (exposed so experiments can override per-run).
    orientation_weight: float = 1.0
    trunk_height_weight: float = 1.0
    waist_height_weight: float = 1.0
    gated_posture_weight: float = 5.0
    joint_pos_limits_weight: float = 1.0
    action_rate_l2_weight: float = 0.01
    joint_vel_l2_weight: float = 0.0  # mjlab_playground initial value
    self_collision_weight: float = 0.1
    # Mechanical power penalty weight. Replaces the earlier
    # ``energy_termination`` (now commented out in the builders) — see
    # :func:`rewards.common.getup.power_penalty`. Ramps up via the
    # curriculum below so the policy first learns getup, then gets
    # squeezed into lower-power solutions.
    power_penalty_weight: float = 1e-4

    # Energy termination initial threshold. The termination curriculum
    # (see ``_build_curriculum_config``) overrides this value once the
    # schedule's first stage fires; mjlab_playground starts with ``∞``
    # (disabled) and first sets a finite threshold at env-step 900*24.
    # The check is also suppressed for the first ``settle_steps``
    # control steps after each reset so landing impacts never fire
    # it spuriously.
    energy_threshold: float = float("inf")

    # Per-joint std for gated posture reward. Regex matched against
    # act_manager.actuated_joint_names at tracker construction time.
    # Newton exposes joint names with an entity prefix ("T1/Waist"),
    # Genesis/MuJoCo use bare names ("Waist"); the leading ``.*`` on
    # every pattern absorbs the prefix on Newton and is a no-op on
    # Genesis/MuJoCo, so the same dict works on all three sims.
    # Values are looser than mjlab's T1 because T1 getup converges
    # slower on a novel framework — Phase K tuning will tighten.
    posture_std_dict: Dict[str, float] = field(default_factory=lambda: {
        r".*_Hip_Roll":       0.08,
        r".*_Hip_Yaw":        0.08,
        r".*_Hip_Pitch":      0.12,
        r".*_Knee_Pitch":     0.15,
        r".*_Ankle_Pitch":    0.2,
        r".*_Ankle_Roll":     0.2,
        r".*AAHead_yaw":      0.15,
        r".*Head_pitch":      0.15,
        r".*Waist":           0.5,
        r".*_Shoulder_Pitch": 0.5,
        r".*_Shoulder_Roll":  0.5,
        r".*_Elbow_Pitch":    0.5,
        r".*_Elbow_Yaw":      0.5,
    })

    # Algorithm.
    algorithm_name: str = "PPO"
    max_iterations: int = 10000

    # Run name (None → auto from sim_type).
    run_name: str | None = None

    # ── Build entry point ─────────────────────────────────────────────

    def build(self):
        """Build the complete typed ConfigsForRun for the configured sim."""
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        kwargs: Dict[str, Any] = dict(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
            observation=builders.build_observation(self),
            action=builders.build_action(self),
            reward=builders.build_reward(self),
            command=self._build_command_config(),
            event=self._build_event_config(),
            curriculum=self._build_curriculum_config(),
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

    def _build_command_config(self) -> CommandConfig:
        """No velocity command — getup has no task-level target velocity.

        ``CommandManager`` handles an empty terms dict gracefully
        (``get_commands_tensor`` returns a ``(num_envs, 0)`` tensor, and
        ``compute`` / ``reset`` become no-ops).
        """
        return CommandConfig(terms={})

    def _build_event_config(self) -> EventConfig:
        """Build full event config: fallen-or-standing reset + sim DR."""
        builders = _get_sim_builders(self.sim_type)

        reset_root_params: Dict[str, Any] = {
            "fallen_prob": self.fallen_prob,
            "fall_height": self.fall_height,
            "fall_velocity_range": self.fall_velocity_range,
            "fall_joint_noise_range": self.fall_joint_noise_range,
            "standing_z_offset": self.standing_z_offset,
            "default_pos": (0.0, 0.0, self.robot.base_init_height),
            "default_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
            "default_joint_pos_dict": self.robot.default_joint_angles,
        }
        if hasattr(builders, "customize_reset_root_params"):
            builders.customize_reset_root_params(self, reset_root_params)

        common_terms = {
            "reset_fallen_or_standing": EventTermConfig(
                func=common_ef.reset_fallen_or_standing,
                mode="reset",
                params=reset_root_params,
            ),
        }
        dr_terms = builders.build_dr_terms(self)

        cfg = EventConfig()
        for name, term in {**common_terms, **dr_terms}.items():
            setattr(cfg, name, term)
        return cfg

    def _build_curriculum_config(self) -> CurriculumManagerConfig:
        """Step-based curriculum — mirrors mjlab_playground T1 getup.

        Schedules reference ``env.env_step_counter`` (number of
        ``env.step()`` calls on the vectorised env). mjlab_playground
        schedule steps are iteration * num_steps_per_env where
        num_steps_per_env == 24; the values below are copied verbatim.

        Signs: mjlab_playground uses **negative weights** for penalty
        terms; we use **positive weights on negative-valued reward
        functions**, so the stage values below mirror the *magnitude*
        of mjlab's weights with positive sign. The resulting cost is
        mathematically identical.
        """
        @dataclass
        class _CurriculumCfg(CurriculumManagerConfig):
            action_rate_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={
                        "reward_name": "raw_action_rate_l2",
                        "stages": [
                            {"step": 0,         "weight": 0.01},
                            {"step": 600 * 24,  "weight": 0.05},
                            {"step": 900 * 24,  "weight": 0.08},
                            {"step": 1200 * 24, "weight": 0.10},
                        ],
                    },
                )
            )
            joint_vel_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={
                        "reward_name": "joint_vel_l2",
                        "stages": [
                            {"step": 0,         "weight": 0.0},
                            {"step": 900 * 24,  "weight": 0.005},
                            {"step": 1200 * 24, "weight": 0.008},
                            {"step": 1500 * 24, "weight": 0.010},
                        ],
                    },
                )
            )

        return _CurriculumCfg()

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
            init_at_random_ep_len=False,
            resume=False,
            resume_path=None,
            run_name=run_name,
            logger="wandb",
            wandb_project="T1_Getup",
            save_interval=500,
            output_dir="auto",
        )
