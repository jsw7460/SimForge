"""NPMP evaluation — standalone post-training and in-training periodic eval.

Three public entry points:

* :func:`run_npmp_eval` — one-shot deterministic NPMP rollout that
  collects rich diagnostics (tracking reward, per-reward-term
  breakdown, episode length, termination breakdown, action gap to
  experts, latent z norm, encoder posterior log-std), grouped per
  motion clip when ``per_motion=True``. Reusable both by the trainer's
  in-training eval loop and the standalone batch evaluator below.

* :class:`NPMPPolicyWrapper` — adapter that exposes a trained
  :class:`NPMPModule` through the ``PolicyWrapper`` interface so the
  existing :class:`ViserPlayViewer` can drive it. Maintains the
  per-env latent ``z_prev`` between steps and zeroes it on env reset
  so the AR(1) chain restarts at the prior origin.

* :class:`NPMPEvaluator` — convenience wrapper for the entry script:
  loads a checkpoint, builds the env, runs ``evaluate()`` (batch) or
  ``play()`` (viser).

Notes
-----
* The eval rollout drives the env directly with the NPMP module's
  deterministic action mean — no DART noise, no expert dispatch in the
  control loop. Experts, when provided, are queried *off-line* on the
  same observations to compute the action gap; their actions never
  reach the env.

* ``ep_starts`` is True at env reset *and* motion-command rollover,
  matching the trainer's convention. The encoder's z prior resets at
  every kinematic-motion discontinuity.

* Per-motion grouping is exact when ``num_envs`` is divisible by the
  motion count; otherwise the last motion absorbs the remainder. The
  evaluator sets ``MotionCommand.set_motion_clip(motion_id, env_ids)``
  for each group at reset, then leaves the assignment fixed for the
  rollout window.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.imitation.npmp.config import CheckpointRef, T1NPMPDistillConfig
from rlworld.imitation.npmp.expert_dispatch import MultiExpertDispatcher
from rlworld.imitation.npmp.module import NPMPModule
from rlworld.imitation.npmp.trainer import NPMPTrainer
from rlworld.rl.evals.policy_wrappers import PolicyWrapper
from rlworld.rl.runners import BaseRunner
from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


__all__ = [
    "NPMPEvalStats",
    "NPMPEvaluator",
    "NPMPPolicyWrapper",
    "run_npmp_eval",
]


# ── Eval stats ──────────────────────────────────────────────────────


@dataclass
class NPMPEvalStats:
    """Diagnostic metrics from one deterministic NPMP eval rollout."""

    # Aggregated env reward signal.
    tracking_reward_mean: float
    tracking_reward_std: float
    episode_length_mean: float
    completed_episodes: int

    # Per-term reward breakdown (anchor_pos, anchor_ori, body_pos, ...).
    reward_terms: dict[str, float]

    # Termination breakdown (per-term reset counts within the eval
    # window, normalised by total). Pulled from
    # ``termination_manager.consume_episode_stats``.
    termination_rates: dict[str, float]

    # Distillation fidelity. ``None`` when no dispatcher was provided.
    action_gap_mean: float | None

    # Latent diagnostics (encoder output).
    z_norm_mean: float
    z_norm_std: float
    q_log_std_mean: float

    # Per-motion breakdown. Keys are motion clip basenames (NPZ stems);
    # values mirror the top-level scalars but restricted to that clip.
    per_motion: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_wandb_dict(self, prefix: str = "Eval") -> dict[str, float]:
        d: dict[str, float] = {
            f"{prefix}/tracking_reward": self.tracking_reward_mean,
            f"{prefix}/tracking_reward_std": self.tracking_reward_std,
            f"{prefix}/episode_length": self.episode_length_mean,
            f"{prefix}/completed_episodes": float(self.completed_episodes),
            f"{prefix}/z_norm": self.z_norm_mean,
            f"{prefix}/z_norm_std": self.z_norm_std,
            f"{prefix}/q_log_std": self.q_log_std_mean,
        }
        if self.action_gap_mean is not None:
            d[f"{prefix}/action_gap"] = self.action_gap_mean
        for term, val in self.reward_terms.items():
            d[f"{prefix}/Reward/{term}"] = val
        for term, rate in self.termination_rates.items():
            d[f"{prefix}/Term/{term}"] = rate
        for motion_name, ms in self.per_motion.items():
            for k, v in ms.items():
                d[f"{prefix}/per_motion/{motion_name}/{k}"] = v
        return d

    def format_table(self) -> str:
        """Human-readable table — used by the standalone entry script."""
        lines = [
            "─" * 80,
            f"NPMP Evaluation",
            "─" * 80,
            f"  tracking_reward      {self.tracking_reward_mean:8.4f}  ± {self.tracking_reward_std:.4f}",
            f"  episode_length       {self.episode_length_mean:8.1f}     ({self.completed_episodes} episodes)",
            f"  z_norm               {self.z_norm_mean:8.4f}  ± {self.z_norm_std:.4f}",
            f"  q_log_std (mean)     {self.q_log_std_mean:+8.4f}",
        ]
        if self.action_gap_mean is not None:
            lines.append(
                f"  action_gap           {self.action_gap_mean:8.4f}"
            )
        if self.reward_terms:
            lines.append("")
            lines.append("Reward terms:")
            for name in sorted(self.reward_terms):
                lines.append(f"  {name:<32s} {self.reward_terms[name]:+8.4f}")
        if self.termination_rates:
            lines.append("")
            lines.append("Termination breakdown:")
            for name in sorted(self.termination_rates):
                lines.append(f"  {name:<32s} {self.termination_rates[name]:8.4f}")
        if self.per_motion:
            lines.append("")
            lines.append("Per-motion:")
            header = f"  {'motion':<20s} {'reward':>10s} {'length':>10s} {'z_norm':>10s} {'gap':>10s}"
            lines.append(header)
            lines.append("  " + "─" * (len(header) - 2))
            for name in sorted(self.per_motion):
                ms = self.per_motion[name]
                rew = ms.get("reward", float("nan"))
                length = ms.get("episode_length", float("nan"))
                zn = ms.get("z_norm", float("nan"))
                gap = ms.get("action_gap", float("nan"))
                lines.append(
                    f"  {name:<20s} {rew:>10.4f} {length:>10.1f} {zn:>10.4f} "
                    f"{gap:>10.4f}"
                )
        lines.append("─" * 80)
        return "\n".join(lines)


# ── JIT-compiled diagnostic step ────────────────────────────────────


@eqx.filter_jit
def _eval_step_batched(
    module: NPMPModule,
    z_prev: jax.Array,
    s_t: jax.Array,
    x_t: jax.Array,
    episode_start: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Vmapped :meth:`NPMPModule.eval_step` across the env axis.

    Returns ``(z_t, action_mean, q_log_std)`` each shaped
    ``(num_envs, ...)``.
    """
    return jax.vmap(module.eval_step)(z_prev, s_t, x_t, episode_start)


# ── Core eval loop ──────────────────────────────────────────────────


def run_npmp_eval(
    module: NPMPModule,
    env: "World",
    num_steps: int,
    *,
    dispatcher: MultiExpertDispatcher | None = None,
    per_motion: bool = True,
) -> NPMPEvalStats:
    """One deterministic NPMP rollout with full diagnostics.

    Args:
        module: Trained NPMP module.
        env: Distillation env (must expose ``decoder_input`` /
            ``encoder_input`` / ``actor`` obs groups and a ``motion``
            command term).
        num_steps: Length of the rollout.
        dispatcher: Optional expert dispatcher. When provided, every
            step also queries the per-motion expert at the same
            observation and accumulates ``||action_NPMP - μ_E||₂`` as
            a fidelity metric.
        per_motion: When True, explicitly assigns env groups to each
            motion clip at reset so per-motion breakdown is exact.
    """
    cmd = env.command_manager.get_term("motion")
    n_motions = cmd._n_motions
    num_envs = env.num_envs
    motion_names = [Path(p).stem for p in cmd.cfg.motion_files]
    latent_dim = module.latent_dim

    # ── Reset env. ───────────────────────────────────────────────────
    env.reset()
    if per_motion:
        envs_per_motion = num_envs // n_motions
        for mi in range(n_motions):
            start = mi * envs_per_motion
            end = (mi + 1) * envs_per_motion if mi < n_motions - 1 else num_envs
            env_ids = torch.arange(start, end, device=env.device)
            cmd.set_motion_clip(mi, env_ids=env_ids)

    z_prev = jnp.zeros((num_envs, latent_dim))
    just_reset = jnp.ones(num_envs, dtype=jnp.bool_)

    # ── Per-step buffers. ────────────────────────────────────────────
    rew_buf: list[jax.Array] = []
    rew_term_bufs: dict[str, list[jax.Array]] = defaultdict(list)
    z_norm_buf: list[jax.Array] = []
    q_log_std_buf: list[jax.Array] = []
    motion_id_buf: list[jax.Array] = []
    action_gap_buf: list[jax.Array] | None = (
        [] if dispatcher is not None else None
    )

    # ── Per-env episode tracking. ────────────────────────────────────
    episode_lengths = torch.zeros(
        num_envs, dtype=torch.long, device=env.device,
    )
    completed_lengths: list[int] = []
    completed_motion_ids: list[int] = []

    # ── Rollout. ─────────────────────────────────────────────────────
    for step in range(num_steps):
        obs = env.obs_manager.get_observation()
        actor_obs = torch_to_jax(obs["actor"])
        decoder_s = torch_to_jax(obs["decoder_input"])
        encoder_x = torch_to_jax(obs["encoder_input"])
        motion_ids = torch_to_jax(cmd.motion_ids)

        prev_time = cmd.time_steps.clone()

        z_t, action_mean, q_log_std = _eval_step_batched(
            module, z_prev, decoder_s, encoder_x, just_reset,
        )

        z_norm_buf.append(jnp.linalg.norm(z_t, axis=-1))
        q_log_std_buf.append(jnp.mean(q_log_std, axis=-1))
        motion_id_buf.append(motion_ids)

        if dispatcher is not None:
            mu_E = dispatcher.deterministic_mean(actor_obs, motion_ids)
            action_gap_buf.append(
                jnp.linalg.norm(action_mean - mu_E, axis=-1)
            )

        action_torch = jax_to_torch(action_mean, env.device)
        _, reward, term, trunc, infos = env.step(action_torch)

        rew_buf.append(torch_to_jax(reward))
        for name, val in infos.get("rewards_per_type", {}).items():
            rew_term_bufs[name].append(torch_to_jax(val))

        episode_lengths = episode_lengths + 1
        dones = term | trunc
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).flatten()
            done_lens = episode_lengths[done_indices].cpu().numpy().tolist()
            done_motions = (
                cmd.motion_ids[done_indices].cpu().numpy().tolist()
            )
            completed_lengths.extend(done_lens)
            completed_motion_ids.extend(done_motions)
            episode_lengths[done_indices] = 0

        new_time = cmd.time_steps
        rollover = new_time != (prev_time + 1)
        next_just_reset = torch_to_jax((term | trunc) | rollover)
        z_prev = jnp.where(
            next_just_reset[:, None], jnp.zeros_like(z_t), z_t,
        )
        just_reset = next_just_reset

    # ── Aggregate. ───────────────────────────────────────────────────
    rew = jnp.stack(rew_buf, axis=0)             # (T, num_envs)
    z_norms = jnp.stack(z_norm_buf, axis=0)
    q_log_stds = jnp.stack(q_log_std_buf, axis=0)
    motion_ids_all = jnp.stack(motion_id_buf, axis=0)
    action_gaps = (
        jnp.stack(action_gap_buf, axis=0)
        if action_gap_buf is not None else None
    )
    rew_terms = {
        name: jnp.stack(vals, axis=0)
        for name, vals in rew_term_bufs.items()
    }

    tracking_reward_mean = float(jnp.mean(rew))
    tracking_reward_std = float(jnp.std(rew))
    z_norm_mean = float(jnp.mean(z_norms))
    z_norm_std = float(jnp.std(z_norms))
    q_log_std_mean = float(jnp.mean(q_log_stds))
    action_gap_mean = (
        float(jnp.mean(action_gaps)) if action_gaps is not None else None
    )

    reward_terms = {
        name: float(jnp.mean(vals)) for name, vals in rew_terms.items()
    }

    if hasattr(env.termination_manager, "consume_episode_stats"):
        raw = env.termination_manager.consume_episode_stats()
        termination_rates = {
            k.split("/")[-1]: float(v) for k, v in raw.items()
        }
    else:
        termination_rates = {}

    episode_length_mean = (
        float(np.mean(completed_lengths))
        if completed_lengths else float(num_steps)
    )

    # ── Per-motion breakdown. ────────────────────────────────────────
    per_motion_stats: dict[str, dict[str, float]] = {}
    if per_motion:
        for mi in range(n_motions):
            mask = motion_ids_all == mi
            n_samples = int(jnp.sum(mask))
            if n_samples == 0:
                continue
            ms: dict[str, float] = {
                "reward": float(jnp.sum(rew * mask) / n_samples),
                "z_norm": float(jnp.sum(z_norms * mask) / n_samples),
                "q_log_std": float(jnp.sum(q_log_stds * mask) / n_samples),
            }
            if action_gaps is not None:
                ms["action_gap"] = float(
                    jnp.sum(action_gaps * mask) / n_samples
                )
            mi_lens = [
                length for length, mid
                in zip(completed_lengths, completed_motion_ids)
                if mid == mi
            ]
            if mi_lens:
                ms["episode_length"] = float(np.mean(mi_lens))
            per_motion_stats[motion_names[mi]] = ms

    return NPMPEvalStats(
        tracking_reward_mean=tracking_reward_mean,
        tracking_reward_std=tracking_reward_std,
        episode_length_mean=episode_length_mean,
        completed_episodes=len(completed_lengths),
        reward_terms=reward_terms,
        termination_rates=termination_rates,
        action_gap_mean=action_gap_mean,
        z_norm_mean=z_norm_mean,
        z_norm_std=z_norm_std,
        q_log_std_mean=q_log_std_mean,
        per_motion=per_motion_stats,
    )


# ── Viser policy adapter ────────────────────────────────────────────


class NPMPPolicyWrapper(PolicyWrapper):
    """Stateful adapter that lets :class:`ViserPlayViewer` drive an
    :class:`NPMPModule` directly. Bypasses ``PolicyWrapper.__init__``
    (which expects a ``BaseRunner``) — NPMP carries its own state and
    has no critic / squash / joint_perm machinery to mirror.
    """

    def __init__(
        self,
        module: NPMPModule,
        num_envs: int,
        device: torch.device,
    ):
        # Skip parent __init__ — set the attributes the play viewer reads.
        self.device = device
        self.is_squashed = False
        self._joint_perm = None

        self._module = module
        self._num_envs = num_envs
        self._latent_dim = module.latent_dim
        self._z_prev = jnp.zeros((num_envs, self._latent_dim))
        self._just_reset = jnp.ones(num_envs, dtype=jnp.bool_)

        self._step_fn = jax.jit(jax.vmap(module.act_step_deterministic))

    def get_action(
        self,
        env_obs: dict[str, torch.Tensor],
        robot_states: torch.Tensor | None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        s_t = torch_to_jax(env_obs["decoder_input"])
        x_t = torch_to_jax(env_obs["encoder_input"])
        z_t, action = self._step_fn(self._z_prev, s_t, x_t, self._just_reset)
        self._z_prev = z_t
        # ``just_reset`` is consumed; ``notify_reset`` will set it again
        # for any envs the play viewer resets externally.
        self._just_reset = jnp.zeros_like(self._just_reset)
        return jax_to_torch(action, self.device)

    def notify_reset(self, reset_mask: np.ndarray) -> None:
        mask = jnp.asarray(reset_mask, dtype=jnp.bool_)
        self._z_prev = jnp.where(
            mask[:, None], jnp.zeros_like(self._z_prev), self._z_prev,
        )
        self._just_reset = self._just_reset | mask


# ── Standalone evaluator ────────────────────────────────────────────


class NPMPEvaluator:
    """Owns env + module for the standalone eval entry script.

    Both eval modes:

    * :meth:`evaluate` — deterministic batch rollout via
      :func:`run_npmp_eval`. Returns :class:`NPMPEvalStats`.

    * :meth:`play` — wires :class:`NPMPPolicyWrapper` into
      :class:`ViserPlayViewer`. The viewer's motion picker tab
      switches the env's tracked clip via
      ``MotionCommand.set_motion_clip``; the encoder picks up the new
      ``motion_future_reference_window`` automatically.
    """

    def __init__(
        self,
        npmp_ckpt: CheckpointRef,
        cfg: T1NPMPDistillConfig | None = None,
        seed: int = 42,
        dispatcher: MultiExpertDispatcher | None = None,
    ):
        if cfg is None:
            cfg = T1NPMPDistillConfig()  # defaults: 9 motions, 90 envs
        self._cfg = cfg
        self._seed = seed

        ckpt_path = npmp_ckpt.resolve(cfg.expert_cache_dir)
        self._module = NPMPTrainer.load_module(ckpt_path)

        cfgs_for_run = cfg.build()
        self._env = BaseRunner._create_env_from_config(cfgs_for_run)

        self._dispatcher = dispatcher

    @property
    def env(self) -> "World":
        return self._env

    @property
    def module(self) -> NPMPModule:
        return self._module

    def attach_dispatcher(self, dispatcher: MultiExpertDispatcher) -> None:
        """Optional — enables ``action_gap`` diagnostic in
        :meth:`evaluate`. Resolves expert checkpoints via
        ``self._cfg.expert_refs`` when called without a pre-built
        dispatcher.
        """
        self._dispatcher = dispatcher

    def evaluate(self, num_steps: int = 500) -> NPMPEvalStats:
        return run_npmp_eval(
            module=self._module,
            env=self._env,
            num_steps=num_steps,
            dispatcher=self._dispatcher,
            per_motion=True,
        )

    def play(self, port: int = 2026) -> None:
        from rlworld.rl.evals.sim_initializers import get_initializer
        from rlworld.rl.vis.viser.play_viewer import ViserPlayViewer

        sim_name = self._env.sim_name
        initializer = get_initializer(sim_name)
        play_scene = initializer.create_play_scene(self._env)

        policy = NPMPPolicyWrapper(
            module=self._module,
            num_envs=self._env.num_envs,
            device=self._env.device,
        )

        viewer = ViserPlayViewer(
            env=self._env,
            play_scene=play_scene,
            policy=policy,
            port=port,
        )
        viewer.run()
