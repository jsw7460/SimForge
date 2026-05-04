"""Configuration dataclasses for T1 NPMP distillation.

* :class:`CheckpointRef` — generic checkpoint reference (local path or
  wandb run). Used for both expert inputs and the trained NPMP module's
  own checkpoint when reloaded for downstream HL controller training.

* :class:`T1NPMPDistillConfig` — subclass of :class:`T1TrackingConfig`
  that builds the same env (scene / action / reward / termination /
  command / events) but swaps the observation config for the NPMP
  three-group layout (``actor`` reused from t1_tracking, plus
  ``decoder_input`` and ``encoder_input``). Adds NPMP architecture and
  training hyperparameters.

The ``actor`` group is reused via direct import of t1_tracking's
hoisted ``_ActorObsCfg`` so the distillation rollout obs matches the
expert checkpoints' actor obs bit-for-bit.

**Currently Newton-only.** ``sim_type`` other than ``"newton"`` raises
in ``__post_init__``. mjlab/Genesis-trained experts can be supported
later by adding sim-specific NPMP obs configs and dispatching from
``build()``; the t1_tracking builders are already hoisted so the wiring
is mechanical.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rlworld.rl.configs.newton_config_classes import NewtonObservationConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.presets.t1_tracking._newton_builders import (
    _ActorObsCfg as _T1TrackingActorObsCfg,
    _CriticObsCfg as _T1TrackingCriticObsCfg,
    _MOTION_PARAMS,
)
from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig
from rlworld.rl.envs.mdp.observations.common.motion_tracking import (
    motion_future_reference_window,
)
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.configs.common_config_classes import ObservationGroupConfig

if TYPE_CHECKING:
    pass


__all__ = [
    "CheckpointRef",
    "T1NPMPDistillConfig",
]


# ── Default expert cache + booster motion naming ────────────────────


_DEFAULT_EXPERT_CACHE_DIR = "outputs/npmp/expert_cache"

_BOOSTER_T1_NPZ_DIR = (
    "./JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted"
)
_BOOSTER_T1_NPZ_NAMES: tuple[str, ...] = (
    "goal_kick",
    "jogging",
    "kick_ball2",
    "kick_ball3",
    "pass_ball1",
    "powerful_kick",
    "running",
    "soccer_drill_run",
    "walking1",
)


# ── CheckpointRef ───────────────────────────────────────────────────


@dataclass
class CheckpointRef:
    """Generic checkpoint reference — local directory or wandb run.

    Mirrors :class:`PolicyEvaluator`'s ``(policy_path | wandb_run_path)``
    interface. Exactly one of ``local_path`` / ``wandb_run_path`` must
    be set. ``wandb_checkpoint_iter`` selects an iteration when pulling
    from wandb (``None`` → latest).

    Used as a single-source-of-truth checkpoint reference across the
    NPMP stack: input expert refs in :class:`T1NPMPDistillConfig` and
    NPMP-module reloads at eval / downstream HL controller time.
    """

    local_path: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_iter: int | None = None

    def resolve(self, cache_dir: str | None = None) -> str:
        """Return a local checkpoint directory.

        Local refs are passed through unchanged. Wandb refs are
        downloaded into ``cache_dir`` (default
        :data:`_DEFAULT_EXPERT_CACHE_DIR`) via
        :func:`get_wandb_checkpoint`, with the artifact digest used to
        skip redundant re-downloads on cache hit.
        """
        if (self.local_path is None) == (self.wandb_run_path is None):
            raise ValueError(
                "CheckpointRef requires exactly one of "
                "local_path or wandb_run_path to be set."
            )
        if self.local_path is not None:
            return self.local_path

        from rlworld.rl.utils.wandb_checkpoint import get_wandb_checkpoint
        path, _ = get_wandb_checkpoint(
            self.wandb_run_path,
            iteration=self.wandb_checkpoint_iter,
            cache_dir=cache_dir,
        )
        return path


# ── NPMP-specific observation groups ────────────────────────────────


@dataclass
class _DecoderInputObsCfg(ObservationGroupConfig):
    """Pure proprio for the NPMP decoder ``π(a | s, z)``.

    Excludes everything motion-related (command, anchor pos/ori, clip
    id, future window) — the encoder consumes the motion reference and
    the decoder receives that information through ``z``. Excludes
    ``dof_pos_nominal_difference`` since it is a pure linear shift of
    ``dof_pos`` and the decoder MLP can recover it internally.

    Noise is intentionally disabled (no ``noise=`` on any term) so the
    BC target is matched against a clean state vector.
    """

    base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0)
    projected_gravity_obs = ObservationTermConfig(
        func=projected_gravity, scale=1.0,
    )
    dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0)
    dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0)
    prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)


@dataclass
class _EncoderInputObsCfg(ObservationGroupConfig):
    """Future motion reference window for the NPMP encoder ``q(z | z_prev, x)``.

    Single term flattening :class:`MotionCommand`'s
    ``future_body_features_in_anchor_frame()`` to
    ``(num_envs, T_future · num_tracked_bodies · 9)``. The encoder
    reshapes back to ``(T, B, 9)`` if needed.
    """

    motion_future_window = ObservationTermConfig(
        func=motion_future_reference_window, scale=1.0,
        params=_MOTION_PARAMS,
    )


@dataclass
class _NPMPObsCfg(NewtonObservationConfig):
    """Four-group observation config for NPMP distillation in Newton.

    * ``actor`` — reused from t1_tracking so expert checkpoints work as-is
    * ``critic`` — reused from t1_tracking so expert ``PPOActorCritic``
      checkpoints deserialise without a critic-shape mismatch. Critic
      obs is computed each step but never read by the trainer; the
      cost is negligible relative to physics. (Stripping the critic
      from the loaded experts would be cleaner but requires a custom
      partial-deserialise path.)
    * ``decoder_input`` — proprio for ``π(a | s, z)``
    * ``encoder_input`` — future motion window for ``q(z | z_prev, x)``
    """

    actor: _T1TrackingActorObsCfg = field(default_factory=_T1TrackingActorObsCfg)
    critic: _T1TrackingCriticObsCfg = field(default_factory=_T1TrackingCriticObsCfg)
    decoder_input: _DecoderInputObsCfg = field(default_factory=_DecoderInputObsCfg)
    encoder_input: _EncoderInputObsCfg = field(default_factory=_EncoderInputObsCfg)


# ── Top-level distillation config ───────────────────────────────────


@dataclass
class T1NPMPDistillConfig(T1TrackingConfig):
    """Configuration for distilling a set of T1 tracking experts into a
    single NPMP motor primitive module.

    Inherits the env construction (scene / action / reward / termination
    / command / events) from :class:`T1TrackingConfig`. Overrides
    :meth:`build` to swap the observation config to :class:`_NPMPObsCfg`
    so that rollout collects ``actor`` (for expert query),
    ``decoder_input`` (NPMP decoder s_t), and ``encoder_input`` (NPMP
    encoder x_t) in one pass through the observation manager.

    ``expert_refs`` is a ``dict`` keyed by **motion clip basename**
    (the NPZ filename without the ``.npz`` extension). The keys must
    exactly match the basenames of every entry in ``motion_files``.
    Keying by name (instead of ordering a tuple in lockstep with
    ``motion_files``) eliminates index-mismatch footguns: each
    expert checkpoint is explicitly tagged with the clip it was
    trained on.
    """

    # ── Multi-motion default: the nine booster T1 clips. ──────────────
    motion_files: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            f"{_BOOSTER_T1_NPZ_DIR}/{name}.npz"
            for name in _BOOSTER_T1_NPZ_NAMES
        )
    )

    # ── Expert checkpoints, keyed by motion-clip basename. ────────────
    expert_refs: dict[str, CheckpointRef] = field(default_factory=dict)
    expert_cache_dir: str = _DEFAULT_EXPERT_CACHE_DIR

    # ── NPMP architecture. ────────────────────────────────────────────
    latent_dim: int = 60
    encoder_hidden: tuple[int, ...] = (256, 256)
    decoder_hidden: tuple[int, ...] = (512, 256, 128)
    ar1_alpha: float = 0.95
    decoder_log_std_init: float = 0.0

    # ── Loss. ─────────────────────────────────────────────────────────
    beta: float = 0.1

    # ── Distillation training schedule. ───────────────────────────────
    rollout_steps: int = 64
    num_grad_steps: int = 4
    batch_traj: int = 2048
    traj_len: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    expert_noise_std: float = 0.025  # DART noise for env step (label clean)

    num_iterations: int = 5000
    save_interval: int = 200
    run_name: str = "T1_NPMP"

    # ── Logging. ──────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "T1_NPMP"
    wandb_group: str | None = None  # None → auto from run_name
    upload_checkpoint_artifact: bool = True

    # ── In-training evaluation. ───────────────────────────────────────
    # Periodic deterministic NPMP rollout in a separate eval env to
    # measure tracking_reward / per-motion breakdown / action_gap to
    # experts / encoder z diagnostics. Mirrors RL pipeline's
    # ``BaseRunner._run_evaluation`` cadence.
    eval_interval: int = 100
    eval_steps: int = 200
    eval_num_envs: int = 90  # 10 envs × 9 motions for per-motion breakdown
    eval_compute_action_gap: bool = True  # requires expert dispatcher

    # ── Validation ────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.sim_type != "newton":
            raise NotImplementedError(
                f"T1NPMPDistillConfig currently supports only "
                f"sim_type='newton'; got {self.sim_type!r}. "
                f"Mujoco / Genesis distillation needs a sim-specific "
                f"_NPMPObsCfg (the t1_tracking builders are already "
                f"hoisted, so adding one is mechanical)."
            )

        expected_keys = {self._motion_key(p) for p in self.motion_files}
        if len(expected_keys) != len(self.motion_files):
            # Two motion files share a basename — would collide as
            # expert_refs keys. Force users to disambiguate upstream.
            raise ValueError(
                "motion_files contains entries with duplicate basenames. "
                "Rename or move so every clip has a unique basename "
                "(used as the expert_refs key)."
            )

        provided_keys = set(self.expert_refs.keys())
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        if missing or extra:
            msg = ["expert_refs keys must exactly match motion_files basenames."]
            if missing:
                msg.append(f"  missing: {sorted(missing)}")
            if extra:
                msg.append(f"  extra:   {sorted(extra)}")
            raise ValueError("\n".join(msg))

    @staticmethod
    def _motion_key(motion_path: str) -> str:
        """Strip directory + ``.npz`` to yield the canonical clip key."""
        return os.path.splitext(os.path.basename(motion_path))[0]

    # ── Build override: swap observation only. ────────────────────────

    def build(self):
        cfgs = super().build()
        cfgs.observation = _NPMPObsCfg()
        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        return cfgs

    # ── Resolve checkpoint paths once at trainer init time. ───────────

    def resolve_expert_paths(self) -> tuple[str, ...]:
        """Return local checkpoint directories ordered to match
        ``motion_files``, downloading any wandb refs into
        :attr:`expert_cache_dir`. The dispatcher's ``experts[i]`` thus
        ends up paired with ``motion_files[i]`` regardless of dict
        insertion order in ``expert_refs``.
        """
        os.makedirs(self.expert_cache_dir, exist_ok=True)
        return tuple(
            self.expert_refs[self._motion_key(p)].resolve(self.expert_cache_dir)
            for p in self.motion_files
        )
