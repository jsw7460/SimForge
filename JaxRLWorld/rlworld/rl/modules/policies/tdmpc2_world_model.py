"""
TD-MPC2 World Model (JAX/Equinox).

Implicit world model with:
- State encoder (MLP with SimNorm)
- Latent dynamics model
- Reward predictor (discrete regression)
- Q-function ensemble with dropout
- Gaussian policy prior

All networks use NormedLinear (LayerNorm + Mish activation).
"""

from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
    two_hot_inv,
    log_std_transform,
    gaussian_logprob,
    squash,
)
from rlworld.rl.modules.normalization import EmpiricalNormalization


# ==================== Custom Layers ====================


class SimNorm(eqx.Module):
    """
    Simplicial normalization.
    Reshapes input into groups of `dim` and applies softmax within each group.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int = 8):
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        shp = x.shape
        x = x.reshape(*shp[:-1], -1, self.dim)
        x = jax.nn.softmax(x, axis=-1)
        return x.reshape(*shp)


class NormedLinear(eqx.Module):
    """
    Linear layer with LayerNorm, Mish activation, and optional dropout.
    TD-MPC2's core building block.
    Handles both [dim] and [batch, dim] inputs directly.
    """
    weight: jax.Array
    bias: jax.Array
    ln_weight: jax.Array
    ln_bias: jax.Array
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    custom_act: Optional[SimNorm]
    use_custom_act: bool = eqx.field(static=True)
    ln_eps: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        act: Optional[SimNorm] = None,
        *,
        key: jax.Array,
    ):
        self.in_features = in_features
        self.out_features = out_features
        k1, k2 = jax.random.split(key)
        # Truncated normal, std=0.02 (matches author's init.weight_init)
        self.weight = jax.random.truncated_normal(k1, -2.0, 2.0, (out_features, in_features)) * 0.02
        self.bias = jnp.zeros(out_features)
        # LayerNorm params
        self.ln_weight = jnp.ones(out_features)
        self.ln_bias = jnp.zeros(out_features)
        self.ln_eps = 1e-5
        self.dropout_rate = dropout
        self.custom_act = act
        self.use_custom_act = act is not None

    def __call__(
        self,
        x: jax.Array,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        # Linear
        x = x @ self.weight.T + self.bias

        # Dropout (only during training)
        if self.dropout_rate > 0.0 and not inference and key is not None:
            mask = jax.random.bernoulli(key, 1.0 - self.dropout_rate, x.shape)
            x = x * mask / (1.0 - self.dropout_rate)

        # LayerNorm
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.ln_eps)
        x = x * self.ln_weight + self.ln_bias

        if self.use_custom_act:
            x = self.custom_act(x)
        else:
            # Mish activation
            x = x * jnp.tanh(jax.nn.softplus(x))

        return x


class OutputLinear(eqx.Module):
    """Plain linear layer (no norm, no activation) for output heads."""
    weight: jax.Array
    bias: jax.Array
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, key: jax.Array):
        self.in_features = in_features
        self.out_features = out_features
        # Truncated normal, std=0.02 (matches author's init.weight_init)
        self.weight = jax.random.truncated_normal(key, -2.0, 2.0, (out_features, in_features)) * 0.02
        self.bias = jnp.zeros(out_features)

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return x @ self.weight.T + self.bias


# ==================== MLP Builder ====================


class TDMPC2MLP(eqx.Module):
    """
    MLP with NormedLinear hidden layers and plain linear output.
    Mirrors the PyTorch `layers.mlp()` in TD-MPC2.
    """
    layers: tuple
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        act: Optional[SimNorm] = None,
        dropout: float = 0.0,
        *,
        key: jax.Array,
    ):
        """
        Args:
            in_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            out_dim: Output dimension
            act: Custom activation for the LAST layer (SimNorm for encoder/dynamics).
                 If None, last layer is plain linear.
            dropout: Dropout rate (applied only to the first hidden layer)
            key: JAX random key
        """
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        num_layers = len(dims) - 1
        keys = jax.random.split(key, num_layers)

        layers = []
        for i in range(num_layers - 1):
            # Hidden layers: NormedLinear with Mish
            layer_dropout = dropout if i == 0 else 0.0
            layers.append(
                NormedLinear(
                    dims[i], dims[i + 1],
                    dropout=layer_dropout,
                    key=keys[i],
                )
            )

        # Output layer
        if act is not None:
            # NormedLinear with custom activation (e.g., SimNorm for encoder)
            layers.append(
                NormedLinear(
                    dims[-2], dims[-1],
                    act=act,
                    key=keys[-1],
                )
            )
        else:
            # Plain linear output
            layers.append(OutputLinear(dims[-2], dims[-1], key=keys[-1]))

        self.layers = tuple(layers)
        self.num_layers = num_layers

    def __call__(
        self,
        x: jax.Array,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """
        Forward pass. All layers handle both [dim] and [batch, dim].

        Args:
            x: [in_dim] or [batch_size, in_dim]

        Returns:
            [out_dim] or [batch_size, out_dim]
        """

        if key is not None:
            keys = jax.random.split(key, self.num_layers)
        else:
            keys = [None] * self.num_layers

        for layer, k in zip(self.layers, keys):
            x = layer(x, key=k, inference=inference)
        return x


# ==================== Q-Function Ensemble ====================


class QFunction(eqx.Module):
    """Single Q-function network."""
    net: TDMPC2MLP

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
        *,
        key: jax.Array,
    ):
        self.net = TDMPC2MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            act=None,
            dropout=dropout,
            key=key,
        )

    def __call__(
        self, x: jax.Array, *, key: Optional[jax.Array] = None, inference: bool = False
    ) -> jax.Array:
        """Forward pass. Handles both [dim] and [batch, dim]."""
        return self.net(x, key=key, inference=inference)


class QEnsemble(eqx.Module):
    """
    Ensemble of Q-functions.
    Each Q-function processes batched inputs via internal vmap.
    """
    q_functions: tuple
    num_q: int = eqx.field(static=True)

    def __init__(
        self,
        num_q: int,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_q)
        self.q_functions = tuple(
            QFunction(in_dim, hidden_dims, out_dim, dropout=dropout, key=k)
            for k in keys
        )
        self.num_q = num_q

    def __call__(
        self,
        x: jax.Array,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """
        Forward pass through all Q-functions.

        Args:
            x: Input [batch_size, in_dim] or [in_dim]
            key: JAX random key (split per Q-function for dropout)
            inference: If True, disable dropout

        Returns:
            Q-values [num_q, batch_size, out_dim] or [num_q, out_dim]
        """
        if key is not None:
            keys = jax.random.split(key, self.num_q)
        else:
            keys = [None] * self.num_q

        outputs = []
        for q_fn, k in zip(self.q_functions, keys):
            outputs.append(q_fn(x, key=k, inference=inference))
        return jnp.stack(outputs, axis=0)


# ==================== World Model ====================


class TDMPC2WorldModel(eqx.Module):
    """
    TD-MPC2 implicit world model.

    Components:
    - encoder: obs -> latent z
    - dynamics: (z, a) -> z'
    - reward: (z, a) -> r (discrete regression)
    - Qs: (z, a) -> Q (ensemble, discrete regression)
    - pi: z -> (mean, log_std) -> action (Gaussian)

    All predictions are in latent space (no decoder).
    """

    # Networks
    encoder: TDMPC2MLP
    dynamics: TDMPC2MLP
    reward_head: TDMPC2MLP
    q_ensemble: QEnsemble
    policy: TDMPC2MLP

    # Configuration (static)
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    num_q: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    simnorm_dim: int = eqx.field(static=True)

    # Log-std bounds
    log_std_min: float = eqx.field(static=True)
    log_std_dif: float = eqx.field(static=True)

    # Squash control: True = tanh squashing (original), False = raw Gaussian output
    squash_action: bool = eqx.field(static=True)

    # Action bounds (static tuples for JIT compatibility)
    # Used by MPPI planning when squash_action=False.
    # When squash_action=True, these are (-1.0, ...) and (1.0, ...) (original behavior).
    action_low_tuple: tuple = eqx.field(static=True)
    action_high_tuple: tuple = eqx.field(static=True)

    # Observation normalizer (optional)
    obs_normalizer: EmpiricalNormalization | None

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        num_enc_layers: int = 2,
        num_q: int = 5,
        num_bins: int = 101,
        simnorm_dim: int = 8,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_action: bool = True,
        action_low: tuple = None,
        action_high: tuple = None,
        obs_normalization: bool = False,
        *,
        key: jax.Array,
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            latent_dim: Latent state dimension
            mlp_dim: Hidden layer dimension for all MLPs
            num_enc_layers: Number of encoder hidden layers
            num_q: Number of Q-functions in ensemble
            num_bins: Number of bins for discrete regression (0 = continuous)
            simnorm_dim: SimNorm group dimension
            dropout: Dropout rate for Q-functions
            log_std_min: Minimum log std for policy
            log_std_max: Maximum log std for policy
            squash_action: If True, apply tanh squashing (original TD-MPC2).
                           If False, output raw Gaussian actions (no tanh).
            action_low: Action lower bounds as tuple. Required when squash_action=False.
                        Ignored when squash_action=True (uses -1.0).
            action_high: Action upper bounds as tuple. Required when squash_action=False.
                         Ignored when squash_action=True (uses 1.0).
            key: JAX random key
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.simnorm_dim = simnorm_dim
        self.log_std_min = log_std_min
        self.log_std_dif = log_std_max - log_std_min
        self.squash_action = squash_action

        # Action bounds: static tuples for JIT compatibility
        if squash_action:
            # Original TD-MPC2: tanh outputs in [-1, 1]
            self.action_low_tuple = tuple([-1.0] * action_dim)
            self.action_high_tuple = tuple([1.0] * action_dim)
        else:
            if action_low is None or action_high is None:
                raise ValueError(
                    "action_low and action_high must be provided when squash_action=False"
                )
            self.action_low_tuple = tuple(float(x) for x in action_low)
            self.action_high_tuple = tuple(float(x) for x in action_high)

        # Observation normalizer
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim)
        else:
            self.obs_normalizer = None

        out_bins = max(num_bins, 1)
        sim_norm = SimNorm(dim=simnorm_dim)

        k_enc, k_dyn, k_rew, k_q, k_pi = jax.random.split(key, 5)

        # Encoder: obs -> latent (with SimNorm output activation)
        enc_hidden = max(num_enc_layers - 1, 1) * [mlp_dim]
        self.encoder = TDMPC2MLP(
            in_dim=obs_dim,
            hidden_dims=enc_hidden,
            out_dim=latent_dim,
            act=sim_norm,
            key=k_enc,
        )

        # Dynamics: (z, a) -> z' (with SimNorm output activation)
        self.dynamics = TDMPC2MLP(
            in_dim=latent_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=latent_dim,
            act=sim_norm,
            key=k_dyn,
        )

        # Reward: (z, a) -> reward logits
        self.reward_head = TDMPC2MLP(
            in_dim=latent_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            act=None,
            key=k_rew,
        )

        # Q-ensemble: (z, a) -> Q logits
        self.q_ensemble = QEnsemble(
            num_q=num_q,
            in_dim=latent_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            dropout=dropout,
            key=k_q,
        )

        # Policy: z -> (mean, log_std)
        self.policy = TDMPC2MLP(
            in_dim=latent_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=2 * action_dim,
            act=None,
            key=k_pi,
        )

        # Zero out reward head last layer weight
        self.reward_head = eqx.tree_at(
            lambda m: m.layers[-1].weight,
            self.reward_head,
            jnp.zeros_like(self.reward_head.layers[-1].weight),
        )

        # Zero out each Q-function's last layer weight
        q_fns = list(self.q_ensemble.q_functions)
        for i in range(num_q):
            q_fns[i] = eqx.tree_at(
                lambda q: q.net.layers[-1].weight,
                q_fns[i],
                jnp.zeros_like(q_fns[i].net.layers[-1].weight),
            )
        self.q_ensemble = eqx.tree_at(
            lambda m: m.q_functions,
            self.q_ensemble,
            tuple(q_fns),
        )

    # ==================== Forward Methods ====================

    def encode(self, obs: jax.Array) -> jax.Array:
        """
        Encode observation to latent state.

        Args:
            obs: [obs_dim] or [batch_size, obs_dim]

        Returns:
            z: [latent_dim] or [batch_size, latent_dim]
        """
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return self.encoder(obs)

    def next_latent(self, z: jax.Array, a: jax.Array) -> jax.Array:
        """
        Predict next latent state.

        Args:
            z: [latent_dim] or [batch_size, latent_dim]
            a: [action_dim] or [batch_size, action_dim]

        Returns:
            z': Same shape as input z
        """
        za = jnp.concatenate([z, a], axis=-1)
        return self.dynamics(za)

    def predict_reward(self, z: jax.Array, a: jax.Array) -> jax.Array:
        """
        Predict reward logits.

        Args:
            z: [latent_dim] or [batch_size, latent_dim]
            a: [action_dim] or [batch_size, action_dim]

        Returns:
            Reward logits [num_bins] or [batch_size, num_bins]
        """
        za = jnp.concatenate([z, a], axis=-1)
        return self.reward_head(za)

    def predict_q(
        self,
        z: jax.Array,
        a: jax.Array,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """
        Predict Q-values from ensemble.

        Args:
            z: Latent [batch_size, latent_dim]
            a: Action [batch_size, action_dim]
            key: Random key for dropout
            inference: Disable dropout if True

        Returns:
            Q logits [num_q, batch_size, num_bins]
        """
        za = jnp.concatenate([z, a], axis=-1)
        return self.q_ensemble(za, key=key, inference=inference)

    def q_value(
        self,
        z: jax.Array,
        a: jax.Array,
        two_hot_cfg: TwoHotConfig,
        return_type: str = "min",
        *,
        key: jax.Array,
        inference: bool = True,
    ) -> jax.Array:
        """
        Get scalar Q-value from ensemble.

        Args:
            z: Latent [batch_size, latent_dim]
            a: Action [batch_size, action_dim]
            two_hot_cfg: Config for two-hot decoding
            return_type: 'min', 'avg', or 'all'
            key: Random key for subsampling and dropout
            inference: Disable dropout if True

        Returns:
            Q-value [batch_size, 1] (for 'min'/'avg') or [num_q, batch_size, 1]
        """
        key1, key2 = jax.random.split(key)
        q_logits = self.predict_q(z, a, key=key1, inference=inference)

        if return_type == "all":
            return q_logits

        q_idx = jax.random.permutation(key2, self.num_q)[:2]
        q_selected = q_logits[q_idx]
        q_values = two_hot_inv(q_selected, two_hot_cfg)

        if return_type == "min":
            return q_values.min(axis=0)
        else:
            return q_values.mean(axis=0)

    def pi(
        self,
        z: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, dict]:
        """
        Sample action from Gaussian policy prior.

        Args:
            z: Latent [batch_size, latent_dim]
            key: Random key

        Returns:
            (action, info_dict)
            - squash_action=True: action is squashed to [-1, 1] via tanh
            - squash_action=False: action is raw Gaussian output (unbounded)
        """
        raw = self._pi_forward(z)
        mean, log_std_raw = jnp.split(raw, 2, axis=-1)

        log_std = log_std_transform(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = jax.random.normal(key, mean.shape)

        log_prob = gaussian_logprob(eps, log_std)

        # Scaled entropy (scale by action dimensions)
        action_dims = float(self.action_dim)
        scaled_log_prob = log_prob * action_dims

        # Reparameterization trick
        action = mean + eps * jnp.exp(log_std)

        if self.squash_action:
            # Original TD-MPC2: tanh squashing with log_prob correction
            mean, action, log_prob = squash(mean, action, log_prob)
            entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        else:
            # Raw Gaussian output: no tanh, no log_prob correction
            # entropy_scale is computed without squash correction
            entropy_scale = scaled_log_prob / (log_prob + 1e-8)

        info = {
            "mean": mean,
            "log_std": log_std,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def _pi_forward(self, z: jax.Array) -> jax.Array:
        """Raw policy forward pass. Handles [dim] and [batch, dim]."""
        return self.policy(z)

    def act_inference(self, obs: jax.Array, *, key: jax.Array) -> tuple[jax.Array, dict]:
        """
        Deterministic action from policy mean.

        - squash_action=True: applies tanh to mean (original)
        - squash_action=False: returns raw mean (unbounded)
        """
        z = self.encode(obs)
        raw = self._pi_forward(z)
        mean, _ = jnp.split(raw, 2, axis=-1)

        if self.squash_action:
            # Original: tanh squashing
            action = jnp.tanh(mean)
        else:
            # Raw mean output (no tanh)
            action = mean

        return action, {"mean": action}