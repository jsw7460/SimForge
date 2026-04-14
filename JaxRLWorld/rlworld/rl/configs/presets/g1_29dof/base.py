"""Unified G1 29-DOF flat-terrain locomotion config.

Single source of truth for g1_29dof across Newton, Genesis, and MuJoCo.
The shared dataclass fields and shared build methods are defined here
once. Per-simulator differences (env, scene, action, reward,
observation, event, visualization) are dispatched to ``_{sim}_builders``
modules at build time.

Compared to go2_flat, the g1_29dof preset has more sim-specific drift:
the critic observation and reward configs are *not* shared across sims
because the underlying state helpers and reward functions have
different names and parameter schemes per simulator. Only the actor
observation, command, algorithm, NN, and runner configs are shared
here; everything else is dispatched to a sim builder.

Usage:
    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    cfgs_for_run = G1FlatConfig(sim_type="newton").build()

Variants (e.g. ``newton/abdnet.py``) inherit ``G1FlatConfig`` directly
and pin ``sim_type`` themselves.
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
from rlworld.rl.configs.presets._sim_builder_protocol import G1SimBuilderProtocol
from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig
from rlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from rlworld.rl.envs.mdp.events import common_event_terms as common_ef


# ── Per-simulator constants ──────────────────────────────────────────
#
# Same effective control rate (50 Hz) across all three sims, but MuJoCo
# uses a 2x faster physics step than Newton/Genesis (400 Hz vs 200 Hz).

_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton":  {"dt": 0.005, "substeps": 2, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 2, "decimation": 4},
    "mujoco":  {"dt": 0.005, "substeps": 2, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton":  "G1_29dof_Newton",
    "genesis": "G1_29dof_Genesis",
    "mujoco":  "G1_29Dof_Mujoco",
}

_SIM_RUNNER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "newton":  {"init_at_random_ep_len": False,  "save_interval": 1000},
    "genesis": {"init_at_random_ep_len": False, "save_interval": 1000},
    "mujoco":  {"init_at_random_ep_len": False,  "save_interval": 1000},
}


def _get_sim_builders(sim_type: str) -> G1SimBuilderProtocol:
    """Lazy-import the simulator-specific builders module.

    The returned module must satisfy :class:`G1SimBuilderProtocol` —
    see ``presets/_sim_builder_protocol.py`` for the full contract.
    """
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
    return mod  # type: ignore[return-value]


# ── Unified config ───────────────────────────────────────────────────


@dataclass
class G1FlatConfig:
    """Unified base configuration for G1 29-DOF flat-terrain locomotion.

    Set ``sim_type`` to choose the simulator backend. All simulator-
    agnostic dataclass fields and build methods live here; per-simulator
    differences are delegated to ``_{sim}_builders`` modules at build
    time.
    """

    # Simulator selection (must be set before build())
    sim_type: str = "newton"

    # Robot configuration
    robot: G1MujocoConfig = field(default_factory=G1MujocoConfig)

    # Environment / training settings (sim-agnostic)
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Command ranges (smaller ang_vel range than Go2)
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-0.5, 0.5)

    # Common event parameters (shared across all 3 sims; sim builders
    # may override reset_root via ``customize_reset_root_params`` hook).
    reset_pose_z_range: tuple[float, float] = (0.01, 0.05)
    reset_joint_position_noise: tuple[float, float] = (0.0, 0.0)
    push_interval_range_s: tuple[float, float] = (1.0, 3.0)

    # Algorithm
    algorithm_name: str = "PPO"
    max_iterations: int = 30000
    actor_class_name: str = "MLPActor"

    # Run name (None → auto from sim_type)
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
        """Constructor kwargs needed to reconstruct this config at eval time.

        Eval reads ``preset_module`` + ``preset_class_name`` + this dict
        from the checkpoint and rebuilds via ``Cls(**kwargs).build()``.
        Only fields whose value differs from the dataclass default are
        included so the dict stays small and forward-compatible.
        """
        from dataclasses import fields, MISSING
        kwargs: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "robot":
                # Nested dataclass — keep at default; eval rebuilds default.
                continue
            value = getattr(self, f.name)
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                # Required field with no default — always include.
                kwargs[f.name] = value
                continue
            if value != default:
                kwargs[f.name] = value
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict output."""
        return self.build().recursive_to_dict()

    # ── Shared build methods (variants may override) ──────────────────

    def _build_common_event_terms(self) -> Dict[str, EventTermConfig]:
        """Build the simulator-agnostic reset/interval event terms.

        Sim builders may register a ``customize_reset_root_params(cfg,
        params)`` hook to mutate the reset_root params dict in place.
        """
        builders = _get_sim_builders(self.sim_type)

        reset_root_params: Dict[str, Any] = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": self.reset_pose_z_range,
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "default_pos": (0.0, 0.0, self.robot.base_init_height),
        }
        if hasattr(builders, "customize_reset_root_params"):
            builders.customize_reset_root_params(self, reset_root_params)

        return {
            "reset_root": EventTermConfig(
                func=common_ef.reset_root_state_uniform,
                mode="reset",
                params=reset_root_params,
            ),
            "reset_dof_pos": EventTermConfig(
                func=common_ef.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": self.reset_joint_position_noise,
                    "velocity_range": (0.0, 0.0),
                },
            ),
            "push_robot": EventTermConfig(
                func=common_ef.push_by_setting_velocity,
                mode="interval",
                interval_range_s=self.push_interval_range_s,
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
        }

    def _build_event_config(self) -> EventConfig:
        """Build full event config = common reset/interval + sim-specific DR."""
        builders = _get_sim_builders(self.sim_type)
        common_terms = self._build_common_event_terms()
        dr_terms = builders.build_dr_terms(self)

        cfg = EventConfig()
        for name, term in {**common_terms, **dr_terms}.items():
            setattr(cfg, name, term)
        return cfg

    def _build_command_config(self) -> CommandConfig:
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
                    "ortho_init": True,
                    "output_gain": 0.01,
                    "hidden_dims": [512, 256, 128],
                },
                critic_kwargs={
                    "activation": "elu",
                    "ortho_init": True,
                    "output_gain": 0.01,
                    "hidden_dims": [1024, 512, 256],
                },
                init_noise_std=1.0,
                distribution_type="gaussian",
                std_type="scalar",
            ),
        )

    def _build_runner_config(self) -> RunnerConfig:
        run_name = self.run_name or _SIM_DEFAULT_RUN_NAMES[self.sim_type]
        overrides = _SIM_RUNNER_OVERRIDES[self.sim_type]
        return RunnerConfig(
            checkpoint=-1,
            log_interval=1,
            max_iterations=self.max_iterations,
            init_at_random_ep_len=overrides["init_at_random_ep_len"],
            resume=False,
            resume_path=None,
            run_name=run_name,
            logger="wandb",
            wandb_project="RLArchitecture",
            save_interval=overrides["save_interval"],
            output_dir="auto",
        )


