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
    """Diagnostic metrics from one deterministic eval rollout.

    Frame-0 protocol: each env starts at frame 0 of its assigned motion
    (via ``MotionCommand.reset_to_frame``). For each env we count *only
    the first episode* — from frame 0 to the first ``term | trunc`` (or
    to the end of the eval window if the env never dies). After that
    first done, the env keeps running (auto-reset to a random RSI start
    on a random motion) but its data is masked out of every aggregate
    so the stats reflect "from frame 0, how far did the policy track"
    rather than "average over many partial-episode fragments".
    """

    # Aggregated env reward signal — first episode only.
    tracking_reward_mean: float  # per-step mean across (env, step) of first ep
    tracking_reward_std: float
    episode_return_mean: float  # per-episode SUM mean (first episode)
    episode_return_std: float
    episode_length_mean: float  # frame 0 → first done step count
    completed_episodes: int

    # Per-term reward breakdown (anchor_pos, anchor_ori, body_pos, ...) —
    # per-step mean within first-episode steps.
    reward_terms: dict[str, float]

    # Termination breakdown — pulled from
    # ``termination_manager.consume_episode_stats`` (counts every reset
    # in the eval window, including auto-reset after the first death).
    termination_rates: dict[str, float]

    # Distillation fidelity. ``None`` when no dispatcher was provided
    # or when ``policy="expert"`` (gap is trivially zero there).
    action_gap_mean: float | None = None

    # Latent diagnostics (encoder output). ``None`` for ``policy="expert"``
    # where the NPMP module never runs.
    z_norm_mean: float | None = None
    z_norm_std: float | None = None
    q_log_std_mean: float | None = None

    # Per-motion breakdown. Keys are motion clip basenames (NPZ stems);
    # value dict has ``episode_return`` / ``episode_length`` / ``reward``
    # plus diagnostics where available.
    per_motion: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_wandb_dict(self, prefix: str = "Eval") -> dict[str, float]:
        d: dict[str, float] = {
            f"{prefix}/tracking_reward": self.tracking_reward_mean,
            f"{prefix}/tracking_reward_std": self.tracking_reward_std,
            f"{prefix}/episode_return": self.episode_return_mean,
            f"{prefix}/episode_return_std": self.episode_return_std,
            f"{prefix}/episode_length": self.episode_length_mean,
            f"{prefix}/completed_episodes": float(self.completed_episodes),
        }
        if self.z_norm_mean is not None:
            d[f"{prefix}/z_norm"] = self.z_norm_mean
        if self.z_norm_std is not None:
            d[f"{prefix}/z_norm_std"] = self.z_norm_std
        if self.q_log_std_mean is not None:
            d[f"{prefix}/q_log_std"] = self.q_log_std_mean
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
            "NPMP Evaluation",
            "─" * 80,
            f"  episode_return       {self.episode_return_mean:8.4f}  ± {self.episode_return_std:.4f}",
            f"  episode_length       {self.episode_length_mean:8.1f}     ({self.completed_episodes} episodes)",
            f"  tracking_reward      {self.tracking_reward_mean:8.4f}  ± {self.tracking_reward_std:.4f}",
        ]
        if self.z_norm_mean is not None:
            lines.append(f"  z_norm               {self.z_norm_mean:8.4f}  ± {self.z_norm_std:.4f}")
        if self.q_log_std_mean is not None:
            lines.append(f"  q_log_std (mean)     {self.q_log_std_mean:+8.4f}")
        if self.action_gap_mean is not None:
            lines.append(f"  action_gap           {self.action_gap_mean:8.4f}")
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
            header = f"  {'motion':<20s} {'return':>10s} {'length':>10s} {'reward':>10s} {'z_norm':>10s} {'gap':>10s}"
            lines.append(header)
            lines.append("  " + "─" * (len(header) - 2))
            for name in sorted(self.per_motion):
                ms = self.per_motion[name]
                ret = ms.get("episode_return", float("nan"))
                length = ms.get("episode_length", float("nan"))
                rew = ms.get("reward", float("nan"))
                zn = ms.get("z_norm", float("nan"))
                gap = ms.get("action_gap", float("nan"))
                lines.append(f"  {name:<20s} {ret:>10.4f} {length:>10.1f} {rew:>10.4f} {zn:>10.4f} {gap:>10.4f}")
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
    env: World,
    num_steps: int,
    *,
    dispatcher: MultiExpertDispatcher | None = None,
    per_motion: bool = True,
    policy: str = "student",
) -> NPMPEvalStats:
    """Frame-0 deterministic eval rollout with full diagnostics.

    Args:
        module: NPMP module. Used as control policy when
            ``policy="student"``; for ``policy="expert"`` only its
            ``latent_dim`` attribute is read (z is never produced).
        env: Distillation env (must expose ``decoder_input`` /
            ``encoder_input`` / ``actor`` obs groups and a ``motion``
            command term).
        num_steps: Rollout length in env steps. Bound the eval window;
            envs that don't terminate within ``num_steps`` have their
            partial first-episode return recorded as if they survived
            the full window.
        dispatcher: Expert dispatcher. Required when
            ``policy="expert"``. Optional when ``policy="student"`` —
            providing it enables the action-gap diagnostic.
        per_motion: When True, explicitly assigns env groups to each
            motion clip via :meth:`MotionCommand.reset_to_frame` (so
            ``motion_ids[env_ids]=mi`` *and* ``time_steps=0`` *and*
            robot qpos/qvel are written to motion ``mi``'s frame 0).
            When False the env's normal RSI applies — useful as a
            no-frame-0 baseline but not what the per-motion table
            expects.
        policy: ``"student"`` (default) — env is driven by the NPMP
            module's deterministic action mean; ``"expert"`` — env is
            driven by ``dispatcher.deterministic_mean`` instead.

    Counting protocol — each env counts only its **first episode**
    (frame 0 → first ``term | trunc`` or to ``num_steps`` end). After
    the first done, the env auto-resets to a random RSI start on a
    random clip but its data is masked from every aggregator so the
    stats reflect "from frame 0, how far did the policy track".
    """
    if policy not in {"student", "expert"}:
        raise ValueError(f"policy must be 'student' or 'expert', got {policy!r}")
    if policy == "expert" and dispatcher is None:
        raise ValueError("policy='expert' requires a dispatcher to query expert means.")

    cmd = env.command_manager.get_term("motion")
    n_motions = cmd._n_motions
    num_envs = env.num_envs
    motion_names = [Path(p).stem for p in cmd.cfg.motion_files]
    latent_dim = module.latent_dim

    # ── Reset env to frame 0 of each motion. ────────────────────────
    # ``env.reset()`` first runs the standard RSI which samples a
    # random frame — overwritten immediately by ``reset_to_frame``
    # below so the actual robot pose lands on motion[mi] frame 0.
    env.reset()
    if per_motion:
        envs_per_motion = num_envs // n_motions
        for mi in range(n_motions):
            start = mi * envs_per_motion
            end = (mi + 1) * envs_per_motion if mi < n_motions - 1 else num_envs
            env_ids = torch.arange(start, end, device=env.device)
            cmd.reset_to_frame(env_ids, frame=0, motion_id=mi)

    # Snapshot motion_id at frame-0 so we can group first-episode
    # returns by motion even after the env auto-resets later.
    initial_motion_ids = cmd.motion_ids.clone()

    # ── State carried across steps. ─────────────────────────────────
    z_prev = jnp.zeros((num_envs, latent_dim))
    just_reset = jnp.ones(num_envs, dtype=jnp.bool_)

    # First-episode mask: True until the env's first ``done``; after
    # that env's data is excluded from aggregates.
    first_episode_alive = torch.ones(
        num_envs,
        dtype=torch.bool,
        device=env.device,
    )
    ep_returns = torch.zeros(num_envs, device=env.device)
    ep_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    # Per-step buffers (indexed by step axis, masked by
    # ``first_episode_alive`` at aggregation time).
    rew_buf: list[torch.Tensor] = []
    rew_term_bufs: dict[str, list[torch.Tensor]] = defaultdict(list)
    z_norm_buf: list[jax.Array] = []
    q_log_std_buf: list[jax.Array] = []
    action_gap_buf: list[jax.Array] = []
    active_mask_buf: list[torch.Tensor] = []  # bool, (num_envs,) per step

    # First-episode completion records.
    completed_returns: list[tuple[int, float]] = []  # (motion_id, return)
    completed_lengths: list[tuple[int, int]] = []  # (motion_id, length)

    # ── Rollout. ─────────────────────────────────────────────────────
    track_action_gap = dispatcher is not None and policy == "student"

    for step in range(num_steps):
        obs = env.obs_manager.get_observation()
        actor_obs = torch_to_jax(obs["actor"])
        motion_ids_jax = torch_to_jax(cmd.motion_ids)

        prev_time = cmd.time_steps.clone()

        # ── Choose action source. ───────────────────────────────────
        if policy == "expert":
            action_jax = dispatcher.deterministic_mean(
                actor_obs,
                motion_ids_jax,
            )
        else:
            decoder_s = torch_to_jax(obs["decoder_input"])
            encoder_x = torch_to_jax(obs["encoder_input"])
            z_t, action_jax, q_log_std = _eval_step_batched(
                module,
                z_prev,
                decoder_s,
                encoder_x,
                just_reset,
            )
            z_norm_buf.append(jnp.linalg.norm(z_t, axis=-1))
            q_log_std_buf.append(jnp.mean(q_log_std, axis=-1))
            if track_action_gap:
                mu_E = dispatcher.deterministic_mean(
                    actor_obs,
                    motion_ids_jax,
                )
                action_gap_buf.append(jnp.linalg.norm(action_jax - mu_E, axis=-1))

        # ── Step env. ───────────────────────────────────────────────
        action_torch = jax_to_torch(action_jax, env.device)
        _, reward, term, trunc, infos = env.step(action_torch)

        # ── Accumulate first-episode stats only. ────────────────────
        active_mask = first_episode_alive
        active_mask_buf.append(active_mask.clone())

        rew_buf.append(reward.detach())
        for name, val in infos.get("rewards_per_type", {}).items():
            rew_term_bufs[name].append(val.detach())

        active_f = active_mask.to(reward.dtype)
        ep_returns = ep_returns + reward * active_f
        ep_lengths = ep_lengths + active_mask.long()

        # ── Detect first-time done for each env. ────────────────────
        dones = term | trunc
        first_done_now = dones & first_episode_alive
        if first_done_now.any():
            done_indices = first_done_now.nonzero(as_tuple=False).flatten()
            done_motion = initial_motion_ids[done_indices].cpu().numpy()
            done_ret = ep_returns[done_indices].cpu().numpy()
            done_len = ep_lengths[done_indices].cpu().numpy()
            for mi, ret, ln in zip(done_motion, done_ret, done_len):
                completed_returns.append((int(mi), float(ret)))
                completed_lengths.append((int(mi), int(ln)))
            first_episode_alive[done_indices] = False

        # ── Latent state update (only matters for student). ─────────
        if policy == "student":
            new_time = cmd.time_steps
            rollover = new_time != (prev_time + 1)
            next_just_reset = torch_to_jax((term | trunc) | rollover)
            z_prev = jnp.where(
                next_just_reset[:, None],
                jnp.zeros_like(z_t),
                z_t,
            )
            just_reset = next_just_reset

    # ── Envs that never died — record partial returns. ──────────────
    still_alive_indices = first_episode_alive.nonzero(
        as_tuple=False,
    ).flatten()
    if still_alive_indices.numel() > 0:
        alive_motion = initial_motion_ids[still_alive_indices].cpu().numpy()
        alive_ret = ep_returns[still_alive_indices].cpu().numpy()
        alive_len = ep_lengths[still_alive_indices].cpu().numpy()
        for mi, ret, ln in zip(alive_motion, alive_ret, alive_len):
            completed_returns.append((int(mi), float(ret)))
            completed_lengths.append((int(mi), int(ln)))

    # ── Aggregate. ───────────────────────────────────────────────────
    # Masked per-step tensors for first-episode-only stats.
    rew_stacked = torch.stack(rew_buf, dim=0)  # (T, num_envs)
    active_stacked = torch.stack(active_mask_buf, dim=0)  # (T, num_envs)
    n_active = float(active_stacked.sum().item())

    if n_active > 0:
        rew_active = rew_stacked[active_stacked]
        tracking_reward_mean = float(rew_active.mean().item())
        tracking_reward_std = float(rew_active.std().item())
    else:
        tracking_reward_mean = 0.0
        tracking_reward_std = 0.0

    reward_terms: dict[str, float] = {}
    for name, vals in rew_term_bufs.items():
        v_stacked = torch.stack(vals, dim=0)
        if n_active > 0:
            reward_terms[name] = float(v_stacked[active_stacked].mean().item())
        else:
            reward_terms[name] = 0.0

    if hasattr(env.termination_manager, "consume_episode_stats"):
        raw = env.termination_manager.consume_episode_stats()
        termination_rates = {k.split("/")[-1]: float(v) for k, v in raw.items()}
    else:
        termination_rates = {}

    # Episode-return aggregation.
    all_returns = [r for _, r in completed_returns]
    all_lengths = [l for _, l in completed_lengths]
    episode_return_mean = float(np.mean(all_returns)) if all_returns else 0.0
    episode_return_std = float(np.std(all_returns)) if all_returns else 0.0
    episode_length_mean = float(np.mean(all_lengths)) if all_lengths else float(num_steps)

    # Latent diagnostics — student-only.
    z_norm_mean: float | None = None
    z_norm_std: float | None = None
    q_log_std_mean: float | None = None
    if policy == "student" and z_norm_buf:
        # Same first-episode masking — convert active_stacked to JAX once.
        active_jax = torch_to_jax(active_stacked)
        z_norms = jnp.stack(z_norm_buf, axis=0)
        q_log_stds = jnp.stack(q_log_std_buf, axis=0)
        n_active_j = jnp.sum(active_jax)
        if float(n_active_j) > 0:
            z_norm_mean = float(jnp.sum(z_norms * active_jax) / n_active_j)
            # std over masked subset
            sq = (z_norms - z_norm_mean) ** 2
            z_norm_std = float(jnp.sqrt(jnp.sum(sq * active_jax) / n_active_j))
            q_log_std_mean = float(jnp.sum(q_log_stds * active_jax) / n_active_j)
        else:
            z_norm_mean = z_norm_std = q_log_std_mean = 0.0

    action_gap_mean: float | None = None
    if track_action_gap and action_gap_buf:
        active_jax = torch_to_jax(active_stacked)
        gaps = jnp.stack(action_gap_buf, axis=0)
        n_active_j = jnp.sum(active_jax)
        if float(n_active_j) > 0:
            action_gap_mean = float(jnp.sum(gaps * active_jax) / n_active_j)
        else:
            action_gap_mean = 0.0

    # ── Per-motion breakdown. ────────────────────────────────────────
    per_motion_stats: dict[str, dict[str, float]] = {}
    if per_motion:
        # Group completed first-episode entries by motion.
        returns_by_motion: dict[int, list[float]] = defaultdict(list)
        lengths_by_motion: dict[int, list[int]] = defaultdict(list)
        for mi, ret in completed_returns:
            returns_by_motion[mi].append(ret)
        for mi, ln in completed_lengths:
            lengths_by_motion[mi].append(ln)

        # Per-step aggregates restricted by initial_motion_ids and
        # first-episode mask.
        initial_motion_ids_t = initial_motion_ids  # (num_envs,)

        for mi in range(n_motions):
            if mi not in returns_by_motion:
                continue
            ms: dict[str, float] = {
                "episode_return": float(np.mean(returns_by_motion[mi])),
                "episode_length": float(np.mean(lengths_by_motion[mi])),
            }

            # Per-step reward / z_norm / etc. for envs whose initial
            # motion was mi *and* still in their first episode.
            env_mask = (initial_motion_ids_t == mi).unsqueeze(0)  # (1, num_envs)
            step_mask = active_stacked & env_mask  # (T, num_envs)
            n_step = float(step_mask.sum().item())
            if n_step > 0:
                ms["reward"] = float(rew_stacked[step_mask].mean().item())
                if policy == "student" and z_norm_buf:
                    step_mask_j = torch_to_jax(step_mask)
                    z_stacked = jnp.stack(z_norm_buf, axis=0)
                    ms["z_norm"] = float(jnp.sum(z_stacked * step_mask_j) / jnp.sum(step_mask_j))
                    qls_stacked = jnp.stack(q_log_std_buf, axis=0)
                    ms["q_log_std"] = float(jnp.sum(qls_stacked * step_mask_j) / jnp.sum(step_mask_j))
                if track_action_gap and action_gap_buf:
                    step_mask_j = torch_to_jax(step_mask)
                    gaps_stacked = jnp.stack(action_gap_buf, axis=0)
                    ms["action_gap"] = float(jnp.sum(gaps_stacked * step_mask_j) / jnp.sum(step_mask_j))

            per_motion_stats[motion_names[mi]] = ms

    return NPMPEvalStats(
        tracking_reward_mean=tracking_reward_mean,
        tracking_reward_std=tracking_reward_std,
        episode_return_mean=episode_return_mean,
        episode_return_std=episode_return_std,
        episode_length_mean=episode_length_mean,
        completed_episodes=len(completed_returns),
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
            mask[:, None],
            jnp.zeros_like(self._z_prev),
            self._z_prev,
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
    def env(self) -> World:
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
