"""Booster K1 joystick task — unified config, MDP-exact to upstream.

Replicates the upstream mujoco_playground ``K1JoystickFlatTerrain`` MDP
(observations, action, rewards, terminations, commands, events, DR) on
Newton / MuJoCo / Genesis. Physics-level parity is exact on the MuJoCo
backend (same MJCF); Newton/Genesis reproduce the identical MDP
definition on their own solvers.

Known deviations (all approved):
- push intervals resample per trigger (upstream: once per episode)
- the critic's 3-D IMU accelerometer entry is omitted (no cross-sim
  accelerometer observation exists yet) → critic is 171-D, not 174-D
- critic noise on the shared 82-D block is an independent draw from the
  actor's (upstream reuses the actor's realization); distributionally
  identical
- DR is applied per env at startup, fixed for the whole run — same as
  upstream's vectorized-model randomization
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict

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
    RewardConfig,
    RunnerConfig,
    StdType,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.k1 import K1Config
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from rlworld.rl.envs.mdp.commands.k1_gait_phase import K1GaitPhaseCommandTermCfg
from rlworld.rl.envs.mdp.events import common as ev
from rlworld.rl.envs.mdp.events.dr import unified as unified_dr
from rlworld.rl.envs.mdp.observations import k1_locomotion as k1_obs
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    applied_torque,
    base_ang_vel,
    base_height,
    base_lin_vel,
    dof_pos_nominal_difference,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards import k1_locomotion as k1_rf
from rlworld.rl.envs.mdp.rewards.common.reward_terms import raw_action_rate_l2, reward_alive

_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    # Upstream: sim_dt=0.002, ctrl_dt=0.02 → decimation 10.
    "newton": {"dt": 0.005, "substeps": 2, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 2, "decimation": 4},
    "mujoco": {"dt": 0.005, "substeps": 2, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES = {
    "newton": "K1_Newton",
    "genesis": "K1_Genesis",
    "mujoco": "K1_Mujoco",
}


def _get_sim_builders(sim_type: str):
    module = {
        "newton": "_newton_builders",
        "genesis": "_genesis_builders",
        "mujoco": "_mujoco_builders",
    }[sim_type]
    return importlib.import_module(f"{__package__}.{module}")


@dataclass
class K1JoystickConfig:
    """Task knobs. Values are the upstream defaults; do not retune here
    without re-checking parity (the parity diag pins these)."""

    sim_type: str = "newton"
    robot: K1Config = field(default_factory=K1Config)
    num_envs: int = 8192
    seed: int = 42
    episode_length_s: float = 20.0

    # Commands (resampled every 10 s; 10% standing envs).
    lin_vel_x_range: tuple = (-1.0, 1.0)
    lin_vel_y_range: tuple = (-0.8, 0.8)
    ang_vel_range: tuple = (-1.0, 1.0)

    # Push events.
    push_interval_range_s: tuple = (5.0, 10.0)
    push_magnitude_range: tuple = (0.1, 1.0)

    # Rewards (upstream reward_config; zero-weight terms omitted).
    tracking_sigma: float = 0.25
    max_foot_height: float = 0.15  # feet_phase bezier swing peak (foot-link z); ~0.11 m sole
    w_tracking_lin_vel: float = 1.0
    w_tracking_ang_vel: float = 0.5
    w_ang_vel_xy: float = -0.15
    w_orientation: float = -1.0
    w_feet_air_time: float = 2.0
    w_feet_slip: float = -0.25
    w_feet_phase: float = 1.0
    w_alive: float = 0.25
    w_joint_deviation_hip: float = -0.1
    w_joint_deviation_knee: float = -0.1
    w_dof_pos_limits: float = -1.0
    w_pose: float = -1.0
    w_feet_distance: float = -1.0
    w_collision: float = -1.0
    # Action-rate smoothness penalty (not in the upstream K1 MDP; added to
    # damp the high-frequency action jitter that upstream's tanh bound alone
    # does not suppress). POSITIVE weight: raw_action_rate_l2 already returns a
    # negative penalty. Mirrors the G1 recipe's value.
    w_raw_action_rate: float = 0.1

    # Action parameterization. The pal recipe pairs a tanh-squashed
    # policy with scale 1.0 and a (-1, 1) clip (identity rescale); the
    # G1-recipe variant swaps all three together (plain gaussian +
    # per-joint 0.25*effort/kp scale + wide clip) — they are a coherent
    # package, never mix-and-match.
    action_distribution: str = "squashed_gaussian"
    action_scale: Any = 1.0
    action_clip: tuple = (-1.0, 1.0)

    # Sim2real domain randomization knobs (0 disables each; see
    # _build_dr_terms / _build_observation_config / the sim builders).
    action_delay_max: int = 2  # per-env command delay U[0,N] PHYSICS steps (~N*5ms)
    obs_delay_max_lag: int = 1  # per-env sensor delay U[0,N] CONTROL steps (~N*20ms)

    # Training.
    algorithm_name: str = "PPO"
    max_iterations: int = 10_000
    actor_hidden_dims: tuple = (512, 256, 128)
    run_name: str | None = None

    def build(self):
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

    # ── Observations (sim-agnostic) ───────────────────────────────────
    #
    # Actor = upstream ``state`` (82-D), exact order:
    #   [linvel(3) gyro(3) gravity(3) command(3)
    #    dof_pos-default(22) dof_vel(22) last_act(22) phase(4)]
    # with upstream's uniform noise scales. No obs scaling (scale=1.0
    # everywhere) — upstream normalizes via the running normalizer only.
    #
    # Critic = the same 82-D block (noise: independent draw) + the
    # noiseless privileged extras in upstream order (accelerometer
    # omitted — see module docstring).

    def _uses_gait_phase(self) -> bool:
        """Whether the gait-phase clock feeds observations / command.

        pal's ``feet_phase`` reward consumes it. Recipes without a
        phase-based reward (the G1 recipe) override this to False, dropping
        the 4-D phase obs block and the ``gait_phase`` command term — a
        75-D actor with no deploy-side gait clock to reconstruct."""
        return True

    def _build_observation_config(self):
        builders = _get_sim_builders(self.sim_type)
        obs_cfg_cls = builders.OBSERVATION_CFG_CLS

        # Per-env stochastic sensor delay on the DEPLOYABLE proprioceptive
        # channels only (IMU gyro/gravity + joint encoders). Internal signals
        # (command / last_action / phase) are not sensor-derived → no delay.
        # Control-step units; hold_prob gives temporally-correlated lag.
        _sd = dict(delay_max_lag=self.obs_delay_max_lag, delay_hold_prob=0.8, delay_update_period=50)
        _use_phase = self._uses_gait_phase()

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            gyro = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2), **_sd)
            gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05), **_sd)
            command = ObservationTermConfig(func=k1_obs.velocity_command, scale=1.0)
            joint_pos = ObservationTermConfig(
                func=dof_pos_nominal_difference, scale=1.0, noise=Unoise(-0.03, 0.03), **_sd
            )
            joint_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5), **_sd)
            last_action = ObservationTermConfig(func=raw_actions, scale=1.0)
            if _use_phase:
                phase = ObservationTermConfig(func=k1_obs.gait_phase_encoding, scale=1.0)

        # Standalone class (NOT inheriting _ActorObsCfg): iter_terms walks
        # the MRO subclass-first, which would put the privileged extras
        # BEFORE the shared 82-D block and scramble the upstream layout.
        @dataclass
        class _CriticObsCfg(ObservationGroupConfig):
            lin_vel = ObservationTermConfig(func=base_lin_vel, scale=1.0, noise=Unoise(-0.1, 0.1))
            gyro = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
            gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
            command = ObservationTermConfig(func=k1_obs.velocity_command, scale=1.0)
            joint_pos = ObservationTermConfig(func=dof_pos_nominal_difference, scale=1.0, noise=Unoise(-0.03, 0.03))
            joint_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
            last_action = ObservationTermConfig(func=raw_actions, scale=1.0)
            if _use_phase:
                phase = ObservationTermConfig(func=k1_obs.gait_phase_encoding, scale=1.0)
            # Privileged extras (noiseless), upstream order.
            gyro_clean = ObservationTermConfig(func=base_ang_vel, scale=1.0)
            gravity_clean = ObservationTermConfig(func=projected_gravity, scale=1.0)
            lin_vel_clean = ObservationTermConfig(func=base_lin_vel, scale=1.0)
            ang_vel_world = ObservationTermConfig(func=k1_obs.base_ang_vel_w, scale=1.0)
            joint_pos_clean = ObservationTermConfig(func=dof_pos_nominal_difference, scale=1.0)
            joint_vel_clean = ObservationTermConfig(func=dof_vel, scale=1.0)
            root_height = ObservationTermConfig(func=base_height, scale=1.0)
            actuator_force = ObservationTermConfig(func=applied_torque, scale=1.0)
            contact = ObservationTermConfig(
                func=k1_obs.feet_contact,
                scale=1.0,
                params={"contact_group": "feet_ground_contact"},
            )
            feet_vel = ObservationTermConfig(
                func=k1_obs.feet_lin_vel_w,
                scale=1.0,
                params={"asset_cfg": SceneEntitySelector(name="robot", body_names=tuple(self.robot.foot_names))},
            )
            air_time = ObservationTermConfig(
                func=k1_obs.feet_air_time,
                scale=1.0,
                params={"contact_group": "feet_ground_contact"},
            )

        @dataclass
        class _ObsCfg(obs_cfg_cls):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

        return _ObsCfg()

    # ── Rewards (sim-agnostic, upstream formulas) ─────────────────────

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        feet_selector = SceneEntitySelector(name="robot", body_names=tuple(r.foot_names))

        @dataclass
        class _RewardsCfg(RewardConfig):
            tracking_lin_vel = RewardTermConfig(
                func=k1_rf.track_lin_vel_xy_exp,
                weight=self.w_tracking_lin_vel,
                params={"tracking_sigma": self.tracking_sigma},
            )
            tracking_ang_vel = RewardTermConfig(
                func=k1_rf.track_ang_vel_z_exp,
                weight=self.w_tracking_ang_vel,
                params={"tracking_sigma": self.tracking_sigma},
            )
            ang_vel_xy = RewardTermConfig(func=k1_rf.ang_vel_xy_l2, weight=self.w_ang_vel_xy)
            orientation = RewardTermConfig(func=k1_rf.orientation_l2, weight=self.w_orientation)
            feet_air_time = RewardTermConfig(
                func=k1_rf.K1FeetAirTime,
                weight=self.w_feet_air_time,
                params={"contact_group": "feet_ground_contact"},
            )
            feet_slip = RewardTermConfig(
                func=k1_rf.feet_slip_base_vel,
                weight=self.w_feet_slip,
                params={"contact_group": "feet_ground_contact"},
            )
            feet_phase = RewardTermConfig(
                func=k1_rf.feet_phase_bezier,
                weight=self.w_feet_phase,
                params={
                    "swing_height": self.max_foot_height,
                    "asset_cfg": feet_selector,
                },
            )
            alive = RewardTermConfig(func=reward_alive, weight=self.w_alive)
            joint_deviation_hip = RewardTermConfig(
                func=k1_rf.joint_deviation_l1,
                weight=self.w_joint_deviation_hip,
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", joint_names=r.hip_joint_patterns),
                    "gate_column": 1,  # lateral velocity command
                },
            )
            joint_deviation_knee = RewardTermConfig(
                func=k1_rf.joint_deviation_l1,
                weight=self.w_joint_deviation_knee,
                params={"asset_cfg": SceneEntitySelector(name="robot", joint_names=r.knee_joint_patterns)},
            )
            dof_pos_limits = RewardTermConfig(func=k1_rf.dof_pos_limits_soft, weight=self.w_dof_pos_limits)
            pose = RewardTermConfig(
                func=k1_rf.K1PoseCost,
                weight=self.w_pose,
                params={"weights": _K1_POSE_WEIGHTS},
            )
            feet_distance = RewardTermConfig(
                func=k1_rf.feet_distance_lateral,
                weight=self.w_feet_distance,
                params={"asset_cfg": feet_selector},
            )
            collision = RewardTermConfig(
                func=k1_rf.contact_pair_penalty,
                weight=self.w_collision,
                params={"contact_group": "feet_pair_contact"},
            )
            action_rate = RewardTermConfig(func=raw_action_rate_l2, weight=self.w_raw_action_rate)

        # Upstream: reward = clip(sum * dt, 0, 10000). Per-term dt scaling
        # already happens in the manager, so the clip acts on the same
        # quantity. Passed via the constructor — a bare class attribute
        # would be shadowed by the dataclass field default (None).
        return _RewardsCfg(total_clip=(0.0, 10000.0))

    # ── Commands / events ─────────────────────────────────────────────

    def _build_command_config(self) -> CommandConfig:
        terms = {
            "velocity": VelocityCommandTermCfg(
                resampling_time_range=(10.0, 10.0),
                lin_vel_x_range=self.lin_vel_x_range,
                lin_vel_y_range=self.lin_vel_y_range,
                ang_vel_range=self.ang_vel_range,
                rel_standing_envs=0.1,
            ),
        }
        if self._uses_gait_phase():
            terms["gait_phase"] = K1GaitPhaseCommandTermCfg()
        return CommandConfig(terms=terms)

    def _build_event_config(self) -> EventConfig:
        terms: Dict[str, EventTermConfig] = {
            "reset_root": EventTermConfig(
                func=ev.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
                    "default_pos": (0.0, 0.0, self.robot.base_init_height),
                },
            ),
            "reset_joints": EventTermConfig(
                func=ev.reset_joints_by_scale,
                mode="reset",
                params={"position_range": (0.5, 1.5)},
            ),
            "push": EventTermConfig(
                func=ev.push_by_planar_impulse,
                mode="interval",
                interval_range_s=self.push_interval_range_s,
                params={"magnitude_range": self.push_magnitude_range},
            ),
        }
        terms.update(self._build_dr_terms())

        # Instance attributes are discovered by iter_terms (same pattern
        # as the Go2 preset's event assembly).
        events = EventConfig()
        for name, term in terms.items():
            setattr(events, name, term)
        return events

    def _build_dr_terms(self) -> Dict[str, EventTermConfig]:
        """Domain randomization for sim2real robustness.

        Context params (friction, joint friction, kp, kd) use ``reset_dr``
        — re-sampled per episode so the policy sees a fresh physics draw
        each reset (far broader coverage than a fixed per-env value). Build
        params (trunk/link mass, armature) stay ``startup`` (fixed per env)
        to avoid a mass-matrix recompute on every reset. Every backend
        samples from a captured baseline, so re-applying at reset does not
        compound. Command latency (DelayedPD) and sensor latency (obs delay)
        are wired in the sim builders / observation config, not here.


        Friction / trunk-mass semantics per sim (Genesis is scale-only):
        the effective ground-contact friction ends up U(0.9, 1.2) and
        the trunk mass ±1 kg on every backend.

        The friction range targets a high-friction ground surface
        (mu ~1.0+). The high band also encourages proper foot lifting: on
        grippy ground the stance foot does not slip, so the policy can
        clear the swing foot cleanly instead of learning a low dragging
        gait (which the near-slipping regime below ~0.7 rewards).
        """
        r = self.robot
        all_bodies = SceneEntitySelector(name="robot", body_names=(".*",))
        trunk = SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,))
        all_joints = SceneEntitySelector(name="robot")
        ankles = SceneEntitySelector(name="robot", joint_names=r.ankle_joint_patterns)

        if self.sim_type == "genesis":
            # set_friction_ratio / set_mass_shift are multiplicative, and
            # set_dofs_frictionloss is absolute-only. All three Genesis
            # variants below are exact equivalents of the upstream ranges
            # given the uniform MJCF defaults (friction 0.6,
            # frictionloss 0.1).
            joint_friction_term = EventTermConfig(
                func=unified_dr.randomize_joint_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": all_joints,
                    "friction_range": (0.09, 0.11),  # 0.1 x U(0.9, 1.1)
                    "operation": "abs",
                },
            )
            friction_term = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", body_names=tuple(r.foot_names)),
                    "friction_range": (0.9 / 0.6, 1.2 / 0.6),  # x0.6 default → U(0.9, 1.2)
                    "operation": "scale",
                },
            )
            trunk_mass_term = EventTermConfig(
                func=unified_dr.randomize_body_mass,
                mode="startup",
                params={
                    "asset_cfg": trunk,
                    "mass_range": (1.0 - 1.0 / 6.5, 1.0 + 1.0 / 6.5),  # ±1 kg on 6.5 kg
                    "operation": "scale",
                },
            )
        else:
            joint_friction_term = EventTermConfig(
                func=unified_dr.randomize_joint_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": all_joints,
                    "friction_range": (0.9, 1.1),
                    "operation": "scale",
                },
            )
            friction_term = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", geom_names=r.foot_geom_names),
                    "friction_range": (0.9, 1.2),
                    "operation": "abs",
                    "axes": [0],
                },
            )
            trunk_mass_term = EventTermConfig(
                func=unified_dr.randomize_body_mass,
                mode="startup",
                params={
                    "asset_cfg": trunk,
                    "mass_range": (-1.0, 1.0),
                    "operation": "add",
                },
            )

        return {
            "dr_friction": friction_term,
            "dr_trunk_mass": trunk_mass_term,
            "dr_link_mass": EventTermConfig(
                func=unified_dr.randomize_body_mass,
                mode="startup",
                params={
                    "asset_cfg": all_bodies,
                    "mass_range": (0.98, 1.02),
                    "operation": "scale",
                },
            ),
            "dr_joint_friction": joint_friction_term,
            "dr_armature": EventTermConfig(
                func=unified_dr.randomize_joint_armature,
                mode="startup",
                params={
                    "asset_cfg": all_joints,
                    "armature_range": (1.0, 1.05),
                    "operation": "scale",
                },
            ),
            # Passive joint (viscous) damping — the plant's own dof_damping
            # (MJCF default class 3 legs / 2 arms), distinct from the PD kd.
            # Randomized ABSOLUTE to [0, 1] to hedge the unknown real-robot
            # joint damping (Booster's own model uses 0; training inherited
            # 3/2 from mujoco_playground). reset_dr so every env sweeps the
            # range across episodes.
            "dr_joint_damping": EventTermConfig(
                func=unified_dr.randomize_joint_damping,
                mode="reset_dr",
                params={
                    "asset_cfg": all_joints,
                    "damping_range": (0.0, 1.0),
                    "operation": "abs",
                },
            ),
            "dr_kp": EventTermConfig(
                func=unified_dr.randomize_pd_gains,
                mode="reset_dr",
                params={
                    "asset_cfg": all_joints,
                    "kp_range": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            # All-joint kd DR (was ankle-only). dr_ankle_kd runs AFTER this
            # so ankles get the wider range (both sample from the same
            # baseline, so the later term wins on the overlapping columns).
            "dr_kd": EventTermConfig(
                func=unified_dr.randomize_pd_gains,
                mode="reset_dr",
                params={
                    "asset_cfg": all_joints,
                    "kd_range": (0.8, 1.25),
                    "operation": "scale",
                },
            ),
            "dr_ankle_kd": EventTermConfig(
                func=unified_dr.randomize_pd_gains,
                mode="reset_dr",
                params={
                    "asset_cfg": ankles,
                    "kd_range": (0.5, 2.0),
                    "operation": "scale",
                },
            ),
        }

    # ── Training configs ──────────────────────────────────────────────

    def _build_algorithm_config(self) -> PPOConfig:
        # Not part of the MDP-parity contract; gamma/entropy follow the
        # upstream brax config, the rest are framework-typical values.
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            obs_normalization=True,
            entropy_coef=0.005,
            gamma=0.97,
            lam=0.95,
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            desired_kl=0.01,
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
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
                # brax PPO's default is tanh_normal: the upstream K1 (and
                # every playground locomotion task) trains with actions
                # squashed to (-1, 1). With clip_actions=(-1, 1) the
                # runner's squashed-policy rescale is the identity, so the
                # effective action space matches upstream exactly
                # (motor target = home pose +- 1 rad).
                distribution_type=DistributionType(self.action_distribution),
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
            wandb_project="K1_Joystick",
            save_interval=2000,
            output_dir="auto",
        )


# Upstream per-joint pose weights (head 1,1; arms 0.1,1,1,1; legs
# 0.01,1,1,0.01,1,1 — hip pitch and knee are relaxed).
_K1_POSE_WEIGHTS: Dict[str, float] = {
    r".*AAHead_yaw": 1.0,
    r".*Head_pitch": 1.0,
    r".*_Shoulder_Pitch": 0.1,
    r".*_Shoulder_Roll": 1.0,
    r".*_Elbow_Pitch": 1.0,
    r".*_Elbow_Yaw": 1.0,
    r".*_Hip_Pitch": 0.01,
    r".*_Hip_Roll": 1.0,
    r".*_Hip_Yaw": 1.0,
    r".*_Knee_Pitch": 0.01,
    r".*_Ankle_Pitch": 1.0,
    r".*_Ankle_Roll": 1.0,
}
