"""Unified Go2 flat-terrain locomotion config.

Single source of truth for go2_flat across Newton, Genesis, and MuJoCo
simulators. The shared dataclass fields and shared build methods are
defined here once. Per-simulator differences (scene, action, event,
reward function imports, env class, visualization class) are dispatched
to ``_{sim}_builders`` modules at build time.

Usage:
    from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig
    cfgs_for_run = Go2FlatConfig(sim_type="newton").build()

Variants (``gait_conditioned``) inherit ``Go2FlatConfig`` directly and
pin ``sim_type`` themselves; see
``presets/go2_flat/{newton,genesis,mujoco}/`` for the variant files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import (
    Activation,
    CommandConfig,
    DistributionType,
    EventConfig,
    GaitConfig,
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
from rlworld.rl.configs.presets._sim_builder_protocol import Go2SimBuilderProtocol
from rlworld.rl.configs.robots.go2 import Go2Config
from rlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from rlworld.rl.envs.mdp.events import common as common_ef
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_lin_vel,
    command as command_obs,
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)

# ── Per-simulator constants ──────────────────────────────────────────
_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "newton": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "mujoco": {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES: Dict[str, str] = {
    "newton": "Go2_Newton",
    "genesis": "Go2_Genesis",
    "mujoco": "Go2_Mujoco",
}


def _get_sim_builders(sim_type: str) -> Go2SimBuilderProtocol:
    """Lazy-import the simulator-specific builders module.

    Lazy imports avoid loading Newton/Genesis/MuJoCo dependencies that
    aren't installed in every environment.  The returned module must
    satisfy :class:`Go2SimBuilderProtocol` — see
    ``presets/_sim_builder_protocol.py`` for the full contract.
    """
    if sim_type == "newton":
        from . import _newton_builders as mod
    elif sim_type == "genesis":
        from . import _genesis_builders as mod
    elif sim_type == "mujoco":
        from . import _mujoco_builders as mod
    else:
        raise ValueError(f"Unknown sim_type: {sim_type!r}. Expected one of {sorted(_SIM_TIMINGS)}.")
    return mod  # type: ignore[return-value]


# ── Unified config ───────────────────────────────────────────────────


@dataclass
class Go2FlatConfig:
    """Unified base configuration for Go2 flat-terrain locomotion.

    Set ``sim_type`` to choose the simulator backend ("newton",
    "genesis", or "mujoco"). All simulator-agnostic dataclass fields and
    build methods live here; per-simulator differences are delegated to
    ``_{sim}_builders`` modules at ``build()`` time.
    """

    # Simulator selection (must be set before build())
    sim_type: str = "newton"

    # Robot configuration
    robot: Go2Config = field(default_factory=Go2Config)

    # Environment / training settings (sim-agnostic)
    num_envs: int = 4096
    episode_length_s: float = 20.0
    seed: int = 42

    # Command ranges
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)

    # Common event parameters (shared across all 3 sims; sim builders
    # override reset_root via ``customize_reset_root_params`` hook).
    reset_pose_z_range: tuple[float, float] = (0.0, 0.0)
    reset_joint_position_noise: tuple[float, float] = (math.pi / 360, math.pi / 120)
    push_interval_range_s: tuple[float, float] = (2.0, 20.0)

    # Algorithm
    algorithm_name: str = "PPO"
    max_iterations: int = 6000
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Run name (None → auto from sim_type)
    run_name: str | None = None

    # Actuator class selection
    use_ideal_pd_actuator: bool = True
    """If True, swap the default DelayedPDActuator pair for
    IdealPDActuator (no command delay). Default preserves the
    trained-policy behaviour; flip to True when actuator delay
    needs to be removed (e.g. for offline data collection)."""

    # ── Build entry point ─────────────────────────────────────────────

    def build(self):
        """Build the complete typed ConfigsForRun for the configured sim."""
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        kwargs: Dict[str, Any] = dict(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
            observation=self._build_observation_config(),
            action=builders.build_action(self),
            reward=self._build_reward_config(),
            command=self._build_command_config(),
            event=self._build_event_config(),
            gait=self._build_gait_config(),
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
        from dataclasses import MISSING, fields

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

    def _build_observation_config(self):
        """Standard actor/critic observation groups (sim-agnostic).

        Reads via ``mdp.observations.common.proprioception``, which
        dispatches through ``env.get_robot_data(...)`` and works on any
        ``World`` subclass.
        """
        builders = _get_sim_builders(self.sim_type)
        ObsCfgClass = builders.OBSERVATION_CFG_CLS

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
            enable_corruption = False
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=2.0)
            base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)

        @dataclass
        class _ObsCfg(ObsCfgClass):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

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

    def _build_common_event_terms(self) -> Dict[str, EventTermConfig]:
        """Build the simulator-agnostic reset/interval event terms.

        Sim builders may register a ``customize_reset_root_params(cfg,
        params)`` hook to mutate the reset_root params dict in place
        (e.g. Newton's xyzw→wxyz default quat, Genesis's spawn offset).
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

    def _build_reward_config(self):
        """Default reward config — delegates to the sim-specific builder.

        Variants override this to install their own (often sim-specific)
        reward terms.
        """
        builders = _get_sim_builders(self.sim_type)
        return builders.build_reward(self)

    def _build_gait_config(self) -> GaitConfig:
        builders = _get_sim_builders(self.sim_type)
        return GaitConfig(foot_names=builders.get_foot_names(self.robot))

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
                actor=MLPActorCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=list(self.actor_hidden_dims),
                ),
                critic=MLPCriticCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=list(self.actor_hidden_dims),
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
            wandb_project="Go2_Flat",
            save_interval=250,
            output_dir="auto",
        )
