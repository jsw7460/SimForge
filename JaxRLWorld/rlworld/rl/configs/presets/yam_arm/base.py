"""Unified YAM fixed-base arm config.

Single source of truth for the YAM arm across Newton, Genesis and
MuJoCo. Per-simulator differences live in ``_{sim}_builders``.

What is deliberately absent, and why: the arm is welded to the world, so
every locomotion term would be meaningless or actively wrong.

* the root reset is kept, but pinned to ``base_pos + env_origins`` with
  no perturbation. mjlab lays environments out on a grid and expects a
  robot to sit over its own cell; on a plain plane that is only a viewer
  concern (worlds are batched, so arms in different environments never
  meet), but it becomes load-bearing the moment the terrain is a
  generator, where each environment is assigned a different sub-patch of
  one shared terrain mesh. Genesis and Newton report zero origins here,
  so the write restates the pose the model was built with;
* no velocity push — it writes root velocity, which a welded base does
  not have;
* no orientation / height terminations — the base cannot move, so they
  would never fire;
* no base-velocity or projected-gravity observations — constants, which
  only inflate the observation dimension;
* no velocity command and no gait, hence the plain ``*Env`` classes
  rather than ``*LocomotionEnv`` (which require a gait config).

Usage::

    from rlworld.rl.configs.presets.yam_arm.base import YamArmConfig
    cfgs = YamArmConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from rlworld.rl.configs import ConfigsForRun
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    Activation,
    CommandConfig,
    DistributionType,
    EventConfig,
    MLPActorCfg,
    MLPCriticCfg,
    NNConfig,
    ObservationGroupConfig,
    OrthoInit,
    PPOPolicyConfig,
    RunnerConfig,
    StdType,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.robots.yam import YamConfig
from rlworld.rl.envs.mdp.events import common as common_ef
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    dof_pos,
    dof_vel,
    raw_actions,
)

# ── Per-simulator constants ──────────────────────────────────────────
_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "mujoco": {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "YamArm_Newton",
    "genesis": "YamArm_Genesis",
    "mujoco": "YamArm_Mujoco",
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
        raise ValueError(f"Unknown sim_type: {sim_type!r}. Expected one of {sorted(_SIM_TIMINGS)}.")
    return mod


@dataclass
class YamArmConfig:
    """Unified base configuration for the YAM fixed-base arm."""

    sim_type: str = "newton"

    robot: YamConfig = field(default_factory=YamConfig)

    num_envs: int = 4096
    episode_length_s: float = 10.0
    seed: int = 42

    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.05)
    """Where the arm is bolted down, in the environment's own frame.

    Baked into the model at build time on every backend, because a welded
    root is structure rather than state. The default clears the ground
    plane: the base collision capsule reaches ~17 mm below the body
    origin, and a base sunk into the floor would generate a permanent
    contact force that quietly pollutes every contact-derived signal.
    """

    reset_joint_position_noise: tuple[float, float] = (-0.05, 0.05)

    max_iterations: int = 1000
    run_name: str = ""
    algorithm_name: str = "PPO"

    # ── Build entry point ─────────────────────────────────────────────

    def build(self) -> ConfigsForRun:
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        kwargs: Dict[str, Any] = dict(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
            observation=self._build_observation_config(),
            action=builders.build_action(self),
            reward=builders.build_reward(self),
            command=self._build_command_config(),
            event=self._build_event_config(),
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
        """Constructor kwargs needed to reconstruct this config at eval time."""
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

    # ── Shared build methods ──────────────────────────────────────────

    def _build_observation_config(self):
        """Joint state and the previous action — nothing base-derived.

        A welded base makes ``base_lin_vel`` / ``base_ang_vel`` /
        ``projected_gravity`` / ``base_height`` constant, so including
        them would widen the observation without adding information.
        """
        builders = _get_sim_builders(self.sim_type)
        ObsCfgClass = builders.OBSERVATION_CFG_CLS

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
            dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5))
            prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

        @dataclass
        class _CriticObsCfg(_ActorObsCfg):
            enable_corruption = False

        @dataclass
        class _ObsCfg(ObsCfgClass):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    def _build_command_config(self) -> CommandConfig:
        """No command: the arm has no task-level target yet."""
        return CommandConfig(terms={})

    def _build_event_config(self) -> EventConfig:
        """Place the arm, then randomise its joints slightly.

        ``reset_root`` carries no perturbation — its only job is to add
        ``env_origins``, which mjlab makes non-zero even on a plane.
        Worlds are batched on every backend, so arms in different
        environments cannot collide either way; the placement matters
        because generator terrain gives each environment its own patch of
        one shared mesh, and because a viewer full of stacked arms is
        unreadable. Genesis and Newton report zero origins on a plane,
        where the write restates the pose the model was built with.

        ``reset_joints_by_offset`` starts from the action manager's
        offset (the home pose) and clamps to the joint limits.
        """
        builders = _get_sim_builders(self.sim_type)

        common_terms = {
            "reset_root": EventTermConfig(
                func=common_ef.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {},
                    "velocity_range": {},
                    "default_pos": self.base_pos,
                },
            ),
            "reset_dof_pos": EventTermConfig(
                func=common_ef.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": self.reset_joint_position_noise,
                    "velocity_range": (0.0, 0.0),
                },
            ),
        }

        cfg = EventConfig()
        for name, term in {**common_terms, **builders.build_dr_terms(self)}.items():
            setattr(cfg, name, term)
        return cfg

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
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor=MLPActorCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=[512, 256, 128],
                ),
                critic=MLPCriticCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=[512, 256, 128],
                ),
                init_noise_std=1.0,
                distribution_type=DistributionType.GAUSSIAN,
                std_type=StdType.STATE_INDEPENDENT,
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
            wandb_project="YamArm",
            save_interval=2000,
            output_dir="auto",
        )
