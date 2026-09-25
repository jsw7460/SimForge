"""The AMP discriminator: a small MLP scoring K-frame feature windows.

Pure functions over an :class:`Discriminator` module and an
:class:`EmpiricalNormalization` held OUTSIDE it (so the optimizer never sees
the running statistics and the gradient penalty differentiates the network
alone), all ``jit``-able:

- :func:`discriminator_logits` -- raw scores of already-normalized windows,
  with the minibatch-std feature optionally detached (the gradient-penalty
  path);
- :func:`discriminator_reward` -- the style reward of raw windows;
- :func:`discriminator_loss` -- the adversarial loss plus gradient penalty
  for one policy / expert batch, with the prediction statistics.

Loss types: ``bce`` (logistic, expert = 1) and ``hinge`` take an R1 penalty
``0.5 * lambda * E||d score / d x||^2`` on expert samples; ``wasserstein``
squashes the critic through ``tanh(eta * D)`` and penalizes
``lambda * E(||grad|| - 1)^2`` on interpolates of expert and policy samples.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxrlworld.rl.modules.normalization import EmpiricalNormalization
from jaxrlworld.rl.modules.utils import MLP

LOSS_TYPES = ("bce", "hinge", "wasserstein")


class Discriminator(eqx.Module):
    """``trunk`` (every layer activated) then a linear ``head`` to one logit."""

    trunk: MLP
    head: eqx.nn.Linear
    use_minibatch_std: bool = eqx.field(static=True)
    loss_type: str = eqx.field(static=True)
    eta: float = eqx.field(static=True)
    reward_scale: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        use_minibatch_std: bool,
        loss_type: str,
        eta: float,
        reward_scale: float,
        *,
        key: jax.Array,
    ):
        if len(hidden_dims) < 1:
            raise ValueError("the discriminator needs at least one hidden layer")
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type {loss_type!r}; one of {LOSS_TYPES}")
        k_trunk, k_head = jax.random.split(key)
        self.trunk = MLP(
            input_dim,
            list(hidden_dims[:-1]),
            hidden_dims[-1],
            activation=activation,
            output_activation=activation,
            key=k_trunk,
        )
        self.head = eqx.nn.Linear(hidden_dims[-1] + (1 if use_minibatch_std else 0), 1, key=k_head)
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type
        self.eta = float(eta)
        self.reward_scale = float(reward_scale)


def minibatch_std(h: jax.Array) -> jax.Array:
    """Mean over features of the per-feature standard deviation across the batch (scalar).

    A feature that is constant over the batch (a dead ReLU unit) has zero
    variance, where ``sqrt`` has an infinite derivative and JAX would hand
    back ``inf * 0 = NaN`` for the whole batch. torch's ``std`` backward
    masks a zero result to a zero gradient; the double ``where`` below does
    the same, so a dead unit contributes no gradient instead of poisoning
    the update.
    """
    var = h.var(axis=0)
    positive = var > 0.0
    std = jnp.where(positive, jnp.sqrt(jnp.where(positive, var, 1.0)), 0.0)
    return std.mean()


def discriminator_logits(disc: Discriminator, x_norm: jax.Array, batch_std: jax.Array | None = None) -> jax.Array:
    """``(B,)`` logits of normalized windows ``x_norm`` ``(B, F)``.

    ``batch_std`` substitutes the minibatch-std feature (the gradient-penalty
    path passes it stop-gradient'd, computed on the same batch).
    """
    h = disc.trunk(x_norm)
    if disc.use_minibatch_std:
        s = minibatch_std(h) if batch_std is None else batch_std
        h = jnp.concatenate([h, jnp.broadcast_to(s, (h.shape[0], 1))], axis=-1)
    return jax.vmap(disc.head)(h)[:, 0]


def _normalize(norm: EmpiricalNormalization | None, x: jax.Array) -> jax.Array:
    return x if norm is None else norm.normalize(x)


@eqx.filter_jit
def discriminator_reward(disc: Discriminator, norm: EmpiricalNormalization | None, x_raw: jax.Array) -> jax.Array:
    """Style reward ``(B,)`` of raw windows: ``softplus(D)`` for bce/hinge
    (``= -log(1 - sigmoid(D))``), ``exp(tanh(eta D))`` for wasserstein, times
    ``reward_scale``."""
    logits = discriminator_logits(disc, _normalize(norm, x_raw))
    if disc.loss_type == "wasserstein":
        reward = jnp.exp(jnp.tanh(disc.eta * logits))
    else:
        reward = jax.nn.softplus(logits)
    return disc.reward_scale * reward


class DiscriminatorStats(NamedTuple):
    amp_loss: jax.Array
    grad_penalty: jax.Array
    policy_pred: jax.Array
    """Mean squashed policy score (sigmoid, or tanh for wasserstein)."""
    expert_pred: jax.Array
    accuracy_policy: jax.Array
    """Fraction of policy samples classified as policy."""
    accuracy_expert: jax.Array


def _gradient_penalty(disc: Discriminator, data_norm: jax.Array, lambda_: float) -> jax.Array:
    """Penalty on ``d score / d input`` at ``data_norm`` (already normalized).

    The minibatch-std feature is computed once on the batch and held
    constant, so each sample's score is a function of that sample alone
    and the per-sample gradient is well defined.
    """
    s = None
    if disc.use_minibatch_std:
        s = jax.lax.stop_gradient(minibatch_std(disc.trunk(data_norm)))

    def score(x: jax.Array) -> jax.Array:
        h = disc.trunk(x)
        if disc.use_minibatch_std:
            h = jnp.concatenate([h, s[None]], axis=-1)
        out = disc.head(h)[0]
        if disc.loss_type == "wasserstein":
            out = jnp.tanh(disc.eta * out)
        return out

    grads = jax.vmap(jax.grad(score))(data_norm)  # (B, F)
    if disc.loss_type == "wasserstein":
        return lambda_ * jnp.mean((jnp.linalg.norm(grads, axis=1) - 1.0) ** 2)
    return 0.5 * lambda_ * jnp.mean(jnp.sum(grads**2, axis=1))


def discriminator_loss(
    disc: Discriminator,
    norm: EmpiricalNormalization | None,
    policy_raw: jax.Array,
    expert_raw: jax.Array,
    lambda_: float,
    key: jax.Array,
) -> tuple[jax.Array, DiscriminatorStats]:
    """``amp_loss + grad_penalty`` for raw policy ``(P, F)`` and expert ``(E, F)`` windows.

    Returns the total and the statistics. The normalizer is applied as a
    constant (it is not a parameter of ``disc``); ``key`` draws the
    interpolation weights of the wasserstein penalty.
    """
    policy_norm = _normalize(norm, policy_raw)
    expert_norm = _normalize(norm, expert_raw)
    n_policy = policy_norm.shape[0]
    logits = discriminator_logits(disc, jnp.concatenate([policy_norm, expert_norm], axis=0))
    policy_d, expert_d = logits[:n_policy], logits[n_policy:]

    if disc.loss_type == "bce":
        expert_loss = jnp.mean(jax.nn.softplus(-expert_d))  # BCE(expert, 1)
        policy_loss = jnp.mean(jax.nn.softplus(policy_d))  # BCE(policy, 0)
        amp_loss = 0.5 * (expert_loss + policy_loss)
    elif disc.loss_type == "hinge":
        amp_loss = 0.5 * (jnp.mean(jax.nn.relu(1.0 - expert_d)) + jnp.mean(jax.nn.relu(1.0 + policy_d)))
    else:
        amp_loss = jnp.mean(jnp.tanh(disc.eta * policy_d)) - jnp.mean(jnp.tanh(disc.eta * expert_d))

    if disc.loss_type == "wasserstein":
        # Interpolates need one expert row per policy row; the expert batch
        # is tiled when it is the smaller one.
        repeats = -(-n_policy // expert_norm.shape[0])
        expert_tiled = jnp.tile(expert_norm, (repeats, 1))[:n_policy]
        alpha = jax.random.uniform(key, (n_policy, 1))
        data = alpha * expert_tiled + (1.0 - alpha) * policy_norm
    else:
        data = expert_norm
    grad_penalty = _gradient_penalty(disc, jax.lax.stop_gradient(data), lambda_)

    if disc.loss_type == "wasserstein":
        policy_prob, expert_prob = jnp.tanh(disc.eta * policy_d), jnp.tanh(disc.eta * expert_d)
        policy_target, expert_target = -1.0, 1.0
    else:
        policy_prob, expert_prob = jax.nn.sigmoid(policy_d), jax.nn.sigmoid(expert_d)
        policy_target, expert_target = 0.0, 1.0
    stats = DiscriminatorStats(
        amp_loss=amp_loss,
        grad_penalty=grad_penalty,
        policy_pred=policy_prob.mean(),
        expert_pred=expert_prob.mean(),
        accuracy_policy=jnp.mean(jnp.round(policy_prob) == policy_target),
        accuracy_expert=jnp.mean(jnp.round(expert_prob) == expert_target),
    )
    return amp_loss + grad_penalty, stats
