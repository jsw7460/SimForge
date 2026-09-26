"""The adversarial motion prior: discriminator, expert windows, replay, policy history.

:class:`AdversarialMotionPrior` owns everything AMP adds to PPO and exposes
two operations the algorithm calls:

- :meth:`shape_rewards` once per rollout step, with the step's ``amp``
  observations, done flags and task rewards: it appends the frame to every
  env's K-frame history (backfilling envs that just reset with their reset
  frame, so a window never spans two episodes), scores the window with the
  discriminator, keeps the window for the discriminator's next update and
  returns the blended reward ``(1 - w) r_task + w dt r_style``;
- :meth:`update_discriminator` once per PPO update: the rollout's windows
  are folded into the replay, then ``num_updates`` gradient steps are taken
  on the discriminator, each on a policy batch (half current rollout, half
  replay) against an expert batch drawn from the reference windows, with the
  running input normalizer advanced after every step.

The discriminator loss is independent of the policy parameters and the PPO
loss is independent of the discriminator (rewards were fixed at rollout
time), so taking the discriminator steps after the PPO scan instead of
interleaved with it changes nothing but the order of random draws.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrlworld.rl.algorithms.amp_ppo.discriminator import (
    Discriminator,
    DiscriminatorStats,
    discriminator_loss,
    discriminator_reward,
)
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import ExpertMotionSet, history_windows
from jaxrlworld.rl.algorithms.amp_ppo.metrics import AmpMetrics
from jaxrlworld.rl.algorithms.metrics.base import host_scalars
from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpConfig
from jaxrlworld.rl.modules.normalization import EmpiricalNormalization

_DISC_FILE = "amp_discriminator.eqx"
_NORM_FILE = "amp_normalizer.eqx"


def expert_frame_probabilities(expert: ExpertMotionSet, dataset_weights: Sequence[float] | None) -> np.ndarray:
    """Sampling probability of every expert frame.

    Each clip variant carries its source file's weight and spreads it
    uniformly over its own frames, so every variant has the same total mass
    (for equal weights) regardless of length.
    """
    n_files = int(expert.source_index.max()) + 1
    if dataset_weights is None:
        weights = np.ones(n_files)
    else:
        weights = np.asarray(dataset_weights, dtype=np.float64)
        if weights.shape != (n_files,):
            raise ValueError(f"{weights.shape[0]} dataset weights for {n_files} motion files")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("dataset weights must be non-negative with a positive sum")
    per_frame = np.repeat(weights[expert.source_index] / expert.clip_lengths, expert.clip_lengths)
    return per_frame / per_frame.sum()


def _discriminator_optimizer(
    cfg: AmpConfig, learning_rate: float, params: Any
) -> tuple[optax.GradientTransformation, Any]:
    """Adam with coupled L2 (torch ``weight_decay``) per group, rate injected so it can be rewritten."""

    def label(path) -> str:
        return "head" if "head" in ".".join(str(p) for p in path) else "trunk"

    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    labels = jax.tree_util.tree_unflatten(treedef, [label(path) for path, _ in flat])

    def group(weight_decay: float) -> optax.GradientTransformation:
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8),
        )

    optimizer = optax.multi_transform(
        {"trunk": group(cfg.discriminator_weight_decay_trunk), "head": group(cfg.discriminator_weight_decay_head)},
        labels,
    )
    return optimizer, labels


def velocity_from_positions(frames: jax.Array, fd: tuple[tuple[int, int], tuple[int, int]], dt: float) -> jax.Array:
    """``(.., K, D)`` frames with the velocity block rewritten as ``(q_t - q_{t-1}) / dt``.

    ``fd = ((pos_start, pos_end), (vel_start, vel_end))`` are the two
    blocks' column ranges; the first frame of a window takes the forward
    difference (its predecessor is outside the window). Applied identically
    to policy and expert windows, so the discriminator compares like with
    like whatever velocity either side started from.
    """
    (ps, pe), (vs, ve) = fd
    q = frames[..., ps:pe]
    backward = (q[..., 1:, :] - q[..., :-1, :]) / dt
    first = backward[..., :1, :]
    vel = jnp.concatenate([first, backward], axis=-2)
    return jnp.concatenate([frames[..., :vs], vel, frames[..., ve:]], axis=-1)


@eqx.filter_jit
def _shape_step(
    disc: Discriminator,
    norm: EmpiricalNormalization | None,
    history: jax.Array,
    amp_obs: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    style_weight: jax.Array,
    dt: jax.Array,
    fd: tuple[tuple[int, int], tuple[int, int]] | None,
):
    """Push one frame, score the window, blend the reward.

    ``history`` is chronological ``(N, K, D)``; the new frame enters at the
    end. Envs flagged done received their post-reset observation, so their
    whole history becomes that frame. ``fd`` (static) selects the
    position-difference velocity of :func:`velocity_from_positions`.
    """
    pushed = jnp.concatenate([history[:, 1:], amp_obs[:, None, :]], axis=1)
    history = jnp.where(dones[:, None, None], amp_obs[:, None, :], pushed)
    frames = history if fd is None else velocity_from_positions(history, fd, dt)
    window = frames.reshape(history.shape[0], -1)
    style_raw = discriminator_reward(disc, norm, window)
    task = (1.0 - style_weight) * rewards
    style = style_weight * dt * style_raw
    return history, window, task + style, task.mean(), style.mean(), style_raw.mean()


@partial(jax.jit, static_argnums=(1,))
def _replay_write(replay: jax.Array, n: int, rows: jax.Array, ptr: jax.Array) -> jax.Array:
    idx = (ptr + jnp.arange(n)) % replay.shape[0]
    return replay.at[idx].set(rows[:n])


@partial(jax.jit, static_argnums=(2, 6, 7, 8))
def _discriminator_updates(
    params: Any,
    static: Any,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    norm: EmpiricalNormalization | None,
    key: jax.Array,
    minibatch_size: int,
    num_updates: int,
    grad_penalty_lambda: float,
    current: jax.Array,
    replay: jax.Array,
    replay_count: jax.Array,
    expert_windows: jax.Array,
    expert_probs: jax.Array,
):
    """``num_updates`` discriminator steps as one scan; returns the per-step statistics."""

    def step(carry, _):
        params, opt_state, norm, key = carry
        key, k_cur, k_rep, k_exp, k_gp = jax.random.split(key, 5)
        idx_cur = jax.random.randint(k_cur, (minibatch_size,), 0, current.shape[0])
        idx_rep = jax.random.randint(k_rep, (minibatch_size,), 0, replay_count)
        idx_exp = jax.random.choice(k_exp, expert_windows.shape[0], (minibatch_size,), p=expert_probs)
        policy = jnp.concatenate([current[idx_cur], replay[idx_rep]], axis=0)
        expert = expert_windows[idx_exp]

        def loss_fn(p):
            return discriminator_loss(eqx.combine(p, static), norm, policy, expert, grad_penalty_lambda, k_gp)

        (_, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if norm is not None:
            # Statistics advance AFTER the step, on the raw batches, expert first.
            norm = norm.update(expert).update(policy)
        return (params, opt_state, norm, key), stats

    (params, opt_state, norm, key), stats = jax.lax.scan(step, (params, opt_state, norm, key), None, length=num_updates)
    return params, opt_state, norm, key, stats


class AdversarialMotionPrior:
    """All AMP state; see the module docstring for the two operations."""

    def __init__(
        self,
        cfg: AmpConfig,
        expert: ExpertMotionSet,
        feature_dim: int,
        control_dt: float,
        learning_rate: float,
        key: jax.Array,
        fd_blocks: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ):
        """``fd_blocks`` are the ``(joint_pos, joint_vel)`` column ranges of the
        feature vector when ``cfg.joint_velocity_from_positions`` is on
        (the algorithm derives them from the layout); ``None`` otherwise."""
        if cfg.joint_velocity_from_positions != (fd_blocks is not None):
            raise ValueError("fd_blocks must be given exactly when joint_velocity_from_positions is on")
        if fd_blocks is not None:
            (ps, pe), (vs, ve) = fd_blocks
            if pe - ps != ve - vs or not (0 <= ps < pe <= feature_dim and 0 <= vs < ve <= feature_dim):
                raise ValueError(f"joint position block {(ps, pe)} and velocity block {(vs, ve)} do not match")
        if cfg.num_amp_obs_steps < 1:
            raise ValueError(f"num_amp_obs_steps must be >= 1, got {cfg.num_amp_obs_steps}")
        if not 0.0 <= cfg.style_reward_weight <= 1.0:
            raise ValueError(f"style_reward_weight must lie in [0, 1], got {cfg.style_reward_weight}")
        if cfg.replay_insert_size < 1 or cfg.replay_buffer_size < 1:
            raise ValueError("replay_buffer_size and replay_insert_size must be >= 1")
        if expert.features.shape[1] != feature_dim:
            raise ValueError(f"expert features are {expert.features.shape[1]} wide, the amp group {feature_dim}")
        self.cfg = cfg
        self.feature_dim = int(feature_dim)
        self.num_steps = int(cfg.num_amp_obs_steps)
        self.window_dim = self.feature_dim * self.num_steps
        self.control_dt = float(control_dt)
        self.style_weight = float(cfg.style_reward_weight)
        self.learning_rate = float(learning_rate)

        self.expert = expert
        self.fd_blocks = fd_blocks
        windows = jnp.asarray(history_windows(expert.features, expert.clip_start, self.num_steps))
        if fd_blocks is not None:
            frames = windows.reshape(windows.shape[0], self.num_steps, self.feature_dim)
            windows = velocity_from_positions(frames, fd_blocks, self.control_dt).reshape(windows.shape[0], -1)
        self.expert_windows = windows
        self.expert_probs = jnp.asarray(expert_frame_probabilities(expert, cfg.dataset_weights), dtype=jnp.float32)

        key, k_disc = jax.random.split(key)
        self.disc = Discriminator(
            self.window_dim,
            tuple(cfg.discriminator_hidden_dims),
            cfg.discriminator_activation,
            cfg.use_minibatch_std,
            cfg.loss_type,
            cfg.wasserstein_eta,
            cfg.reward_scale,
            key=k_disc,
        )
        self.normalizer = EmpiricalNormalization(self.window_dim) if cfg.empirical_normalization else None
        params, _ = eqx.partition(self.disc, eqx.is_inexact_array)
        self.optimizer, self.param_labels = _discriminator_optimizer(cfg, self.learning_rate, params)
        self.opt_state = self.optimizer.init(params)
        self.key = key

        self.replay = jnp.zeros((cfg.replay_buffer_size, self.window_dim), dtype=jnp.float32)
        self.replay_count = 0
        self.replay_ptr = 0
        self._history: jax.Array | None = None
        self._current: list[jax.Array] = []
        self._reset_stats()

    # ── rollout side ─────────────────────────────────────────────────

    def _reset_stats(self) -> None:
        self._task_sum = jnp.zeros(())
        self._style_sum = jnp.zeros(())
        self._raw_sum = jnp.zeros(())
        self._num_steps_seen = 0

    def reset_history(self, amp_obs: jax.Array) -> None:
        """Backfill every env's history with ``amp_obs`` ``(N, D)``."""
        if amp_obs.ndim != 2 or amp_obs.shape[1] != self.feature_dim:
            raise ValueError(f"expected amp observations (N, {self.feature_dim}), got {tuple(amp_obs.shape)}")
        self._history = jnp.broadcast_to(amp_obs[:, None, :], (amp_obs.shape[0], self.num_steps, self.feature_dim))

    def shape_rewards(self, amp_obs: jax.Array, dones: jax.Array, rewards: jax.Array) -> jax.Array:
        """Blend the style reward of the window ending at this step into ``rewards``.

        The first call of a run backfills the history with this step's
        frame; later resets are handled per env through ``dones``.
        """
        if self._history is None:
            self.reset_history(amp_obs)
        elif self._history.shape[0] != amp_obs.shape[0]:
            raise ValueError(f"amp observation batch changed from {self._history.shape[0]} to {amp_obs.shape[0]}")
        self._history, window, blended, task_mean, style_mean, raw_mean = _shape_step(
            self.disc,
            self.normalizer,
            self._history,
            amp_obs,
            dones,
            rewards,
            jnp.asarray(self.style_weight, dtype=jnp.float32),
            jnp.asarray(self.control_dt, dtype=jnp.float32),
            self.fd_blocks,
        )
        self._current.append(window)
        self._task_sum = self._task_sum + task_mean
        self._style_sum = self._style_sum + style_mean
        self._raw_sum = self._raw_sum + raw_mean
        self._num_steps_seen += 1
        return blended

    # ── update side ──────────────────────────────────────────────────

    def _insert_replay(self, current: jax.Array) -> None:
        """Fill the replay eagerly, then keep only a random subset of each rollout."""
        capacity = self.replay.shape[0]
        if self.replay_count >= capacity:
            n = min(current.shape[0], self.cfg.replay_insert_size)
            self.key, k = jax.random.split(self.key)
            rows = current[jax.random.permutation(k, current.shape[0])[:n]]
        else:
            n = min(current.shape[0], capacity)
            rows = current[-n:]
        self.replay = _replay_write(self.replay, n, rows, jnp.asarray(self.replay_ptr))
        self.replay_ptr = (self.replay_ptr + n) % capacity
        self.replay_count = min(capacity, self.replay_count + n)

    def update_discriminator(self, num_updates: int, minibatch_size: int) -> AmpMetrics:
        """Fold the rollout into the replay and take ``num_updates`` discriminator steps."""
        if not self._current:
            raise RuntimeError("no rollout windows collected since the last update")
        if minibatch_size > self.replay.shape[0]:
            raise ValueError(f"minibatch {minibatch_size} exceeds the replay capacity {self.replay.shape[0]}")
        current = jnp.concatenate(self._current, axis=0)
        self._current = []
        self._insert_replay(current)

        params, static = eqx.partition(self.disc, eqx.is_inexact_array)
        self.key, k = jax.random.split(self.key)
        params, self.opt_state, self.normalizer, _, stats = _discriminator_updates(
            params,
            static,
            self.optimizer,
            self.opt_state,
            self.normalizer,
            k,
            int(minibatch_size),
            int(num_updates),
            float(self.cfg.grad_penalty_lambda),
            current,
            self.replay,
            jnp.asarray(self.replay_count),
            self.expert_windows,
            self.expert_probs,
        )
        self.disc = eqx.combine(params, static)
        return self._metrics(stats)

    def _metrics(self, stats: DiscriminatorStats) -> AmpMetrics:
        n = max(self._num_steps_seen, 1)
        values = host_scalars(
            {
                "disc_loss": stats.amp_loss.mean(),
                "grad_penalty": stats.grad_penalty.mean(),
                "policy_pred": stats.policy_pred.mean(),
                "expert_pred": stats.expert_pred.mean(),
                "accuracy_policy": stats.accuracy_policy.mean(),
                "accuracy_expert": stats.accuracy_expert.mean(),
                "style_reward": self._style_sum / n,
                "task_reward": self._task_sum / n,
                "style_reward_raw": self._raw_sum / n,
            }
        )
        self._reset_stats()
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise FloatingPointError(
                f"adversarial motion prior produced non-finite values {bad}: {values}. "
                "Stopping rather than training on NaN rewards."
            )
        return AmpMetrics(**values, style_weight=self.style_weight, discriminator_lr=self.learning_rate)

    # ── learning rate / checkpoint ───────────────────────────────────

    def set_learning_rate(self, lr: float) -> None:
        """Rewrite the injected rate of both groups; Adam moments stay."""
        self.learning_rate = float(lr)
        for label in ("trunk", "head"):
            # multi_transform -> per-label chain(add_decayed_weights, inject_hyperparams(adam)).
            hyperparams = self.opt_state.inner_states[label].inner_state[1].hyperparams
            hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)

    def save(self, checkpoint_dir: str) -> None:
        eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, _DISC_FILE), self.disc)
        if self.normalizer is not None:
            eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, _NORM_FILE), self.normalizer)

    def load(self, checkpoint_dir: str) -> None:
        self.disc = eqx.tree_deserialise_leaves(os.path.join(checkpoint_dir, _DISC_FILE), self.disc)
        if self.normalizer is not None:
            self.normalizer = eqx.tree_deserialise_leaves(os.path.join(checkpoint_dir, _NORM_FILE), self.normalizer)
        params, _ = eqx.partition(self.disc, eqx.is_inexact_array)
        self.opt_state = self.optimizer.init(params)
        self.set_learning_rate(self.learning_rate)
