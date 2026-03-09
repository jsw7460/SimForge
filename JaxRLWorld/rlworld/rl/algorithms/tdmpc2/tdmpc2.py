from typing import Any, Dict, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.rl.algorithms.base import (
    OffPolicyAlgorithm,
    ActInput,
    copy_params,
)
from rlworld.rl.algorithms.metrics import BatchMetrics
from rlworld.rl.algorithms.tdmpc2.math import make_two_hot_config
from rlworld.rl.algorithms.tdmpc2.metrics import (
    TDMPC2Metrics,
    TDMPC2WorldModelMetrics,
    TDMPC2QMetrics,
    TDMPC2PolicyMetrics,
)
from rlworld.rl.algorithms.tdmpc2.update import (
    plan_mppi_batched,
    unified_update
)
from rlworld.rl.modules.normalization import EmpiricalNormalization
from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel
from rlworld.rl.storages.sequence_replay_buffer import SequenceReplayBuffer, SequenceBatch


# ==================== Running Scale ====================


class RunningScale:
    """Running trimmed scale estimator (matches author's implementation)."""

    def __init__(self, tau: float = 0.01):
        self.tau = tau
        self._value = 1.0

    @property
    def value(self) -> float:
        return self._value

    def update(self, qs: jax.Array) -> None:
        """Update scale using P95 - P05 percentile range."""
        qs_flat = qs.flatten()
        p05 = float(jnp.percentile(qs_flat, 5))
        p95 = float(jnp.percentile(qs_flat, 95))
        new_scale = max(p95 - p05, 1.0)
        self._value = (1.0 - self.tau) * self._value + self.tau * new_scale


# ==================== Train State ====================


class TDMPC2TrainState(NamedTuple):
    """Training state for TD-MPC2."""
    model: TDMPC2WorldModel
    target_q_params: Any
    wm_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    key: jax.Array


# ==================== Algorithm ====================


class TDMPC2(OffPolicyAlgorithm):
    """
    TD-MPC2: Scalable, Robust World Models for Continuous Control.

    Combines model-based planning (MPPI) with model-free TD-learning.
    Uses an implicit world model operating entirely in latent space.
    """

    ActInput = ActInput

    def __init__(
        self,
        world_model: TDMPC2WorldModel,
        num_envs: int,
        # Discount / episode
        gamma: float = 0.99,
        episode_length: int = 1000,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        discount_denom: float = 5.0,
        # Learning rates
        lr: float = 3e-4,
        pi_lr: float = 3e-4,
        # Target network
        tau: float = 0.01,
        # Planning (MPPI)
        mpc: bool = True,
        horizon: int = 3,
        num_samples: int = 512,
        num_pi_trajs: int = 24,
        num_elites: int = 64,
        num_iterations: int = 6,
        temperature: float = 0.5,
        min_std: float = 0.05,
        max_std: float = 2.0,
        # Losses
        consistency_coef: float = 2.0,
        reward_coef: float = 0.5,
        value_coef: float = 0.1,
        entropy_coef: float = 1e-4,
        rho: float = 0.5,
        # Discrete regression
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        # Training
        batch_size: int = 256,
        grad_clip_norm: float = 20.0,
        max_grad_norm: float = 20.0,
        # Key
        key: jax.Array = None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        super().__init__(
            actor_critic=world_model,
            gamma=gamma,
            tau=tau,
            key=key,
        )

        self.world_model = world_model
        self.lr = lr
        self.pi_lr = pi_lr
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.max_grad_norm = max_grad_norm

        # Planning
        self.mpc = mpc
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_pi_trajs = num_pi_trajs
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std

        # Heuristic: increase iterations for large action spaces
        if world_model.action_dim >= 20:
            self.num_iterations += 2
        # Loss coefficients
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rho = rho

        # Discount (heuristic based on episode length)
        self.discount = self._compute_discount(
            episode_length, discount_min, discount_max, discount_denom
        )

        # Discrete regression config
        self.two_hot_cfg = make_two_hot_config(num_bins, vmin, vmax)

        # Running Q-value scale for policy loss normalization
        self.scale = RunningScale()

        # Action bounds are stored in world_model as static tuple fields:
        # world_model.action_low_tuple, world_model.action_high_tuple
        # Used by MPPI planning (plan_mppi_inner reads from model directly)

        # Create optimizers
        self._create_optimizers(world_model)

        # Initialize train state
        self._init_train_state(key)

        # Previous MPPI mean (warm-starting across steps)
        self._prev_mean = np.zeros(
            (num_envs, horizon, world_model.action_dim), dtype=np.float32
        )

        # Observation normalization
        self.obs_normalization = world_model.obs_normalizer is not None

        # Storage
        self.replay_buffer: Optional[SequenceReplayBuffer] = None

        # Update counter
        self.total_it = 0

        self._print_config()

    # ==================== Initialization ====================

    def _compute_discount(
        self, episode_length: int, discount_min: float,
        discount_max: float, discount_denom: float,
    ) -> float:
        frac = episode_length / discount_denom
        return min(max((frac - 1) / frac, discount_min), discount_max)

    def _create_optimizers(self, model: TDMPC2WorldModel):
        param_filter = eqx.is_inexact_array

        all_params = eqx.filter(model, param_filter)
        labels = jax.tree.map(lambda _: "rest", all_params)

        encoder_labels = jax.tree.map(lambda _: "encoder", eqx.filter(model.encoder, param_filter))
        labels = eqx.tree_at(lambda m: m.encoder, labels, encoder_labels)

        policy_labels = jax.tree.map(lambda _: "frozen", eqx.filter(model.policy, param_filter))
        labels = eqx.tree_at(lambda m: m.policy, labels, policy_labels)

        # Freeze obs normalizer (stats updated manually, not via gradient)
        if model.obs_normalizer is not None:
            norm_params = eqx.filter(model.obs_normalizer, param_filter)
            norm_labels = jax.tree.map(lambda _: "frozen", norm_params)
            labels = eqx.tree_at(
                lambda m: m.obs_normalizer, labels, norm_labels,
                is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
            )

        self.wm_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.multi_transform(
                transforms={
                    "encoder": optax.adam(learning_rate=self.lr * 0.3),
                    "rest": optax.adam(learning_rate=self.lr),
                    "frozen": optax.set_to_zero(),
                },
                param_labels=labels,
            ),
        )

        self.pi_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adam(learning_rate=self.pi_lr, eps=1e-5),
        )

    def _init_train_state(self, key: jax.Array):
        key, subkey = jax.random.split(key)

        # Target Q parameters
        q_params, _ = eqx.partition(self.world_model.q_ensemble, eqx.is_inexact_array)
        target_q_params = copy_params(q_params)

        # World model optimizer: init on full model
        # (eqx.filter_value_and_grad + eqx.apply_updates operate on full model)
        wm_opt_state = self.wm_optimizer.init(
            eqx.filter(self.world_model, eqx.is_inexact_array)
        )

        # Policy optimizer
        pi_params, _ = eqx.partition(self.world_model.policy, eqx.is_inexact_array)
        pi_opt_state = self.pi_optimizer.init(pi_params)

        self.train_state = TDMPC2TrainState(
            model=self.world_model,
            target_q_params=target_q_params,
            wm_opt_state=wm_opt_state,
            pi_opt_state=pi_opt_state,
            key=subkey,
        )

    def _print_config(self):
        print(f"\n🔧 TD-MPC2 Configuration:")
        print(f"  Discount: {self.discount:.4f}")
        print(f"  LR: {self.lr}, Pi LR: {self.pi_lr}")
        print(f"  Tau: {self.tau}")
        print(f"  MPC: {self.mpc}, Horizon: {self.horizon}")
        print(f"  Samples: {self.num_samples}, Elites: {self.num_elites}, Iters: {self.num_iterations}")
        print(f"  Num bins: {self.two_hot_cfg.num_bins}, vmin: {self.two_hot_cfg.vmin}, vmax: {self.two_hot_cfg.vmax}")
        print(f"  Batch size: {self.batch_size}")
        print(
            f"  Obs: {self.world_model.obs_dim}, Act: {self.world_model.action_dim}, Latent: {self.world_model.latent_dim}")
        print(f"  Num Q: {self.world_model.num_q}")
        print(f"  Loss coefs: consistency={self.consistency_coef}, reward={self.reward_coef}, value={self.value_coef}")
        print(f"  Squash action: {self.world_model.squash_action}")
        print(f"  Obs normalization: {self.obs_normalization}")
        if not self.world_model.squash_action:
            print(f"  Action bounds: [{self.world_model.action_low_tuple}, {self.world_model.action_high_tuple}]")

    # ==================== Properties ====================

    @property
    def model(self) -> TDMPC2WorldModel:
        return self.train_state.model

    # ==================== Storage ====================

    def init_storage(self, cfg: Dict[str, Any]) -> None:
        """Initialize sequence replay buffer."""
        self.replay_buffer = SequenceReplayBuffer(
            num_envs=cfg["num_envs"],
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            size_per_env=cfg["size_per_env"],
            horizon=self.horizon,
        )

    def store_transition(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.store_parallel(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, terminated=terminated, truncated=truncated,
        )

    def sample_batch(self, batch_size: int, key: jax.Array) -> SequenceBatch:
        """Sample batch from replay buffer."""
        return self.replay_buffer.sample_batch(batch_size, key)

    # ==================== Acting ====================

    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """
        Select action. Uses actor_obs for world model encoding.

        For MPPI planning, call act_with_t0 instead (requires t0 flag).
        Without MPPI, uses direct policy forward pass.
        """
        model = self.train_state.model
        key = self.train_state.key
        key, subkey = jax.random.split(key)
        self.train_state = self.train_state._replace(key=key)

        z = model.encode(obs.actor_obs)
        action, info = model.pi(z, key=subkey)

        if deterministic:
            action = info["mean"]

        return action

    def act_with_t0(
        self,
        obs: jax.Array,
        t0_mask: np.ndarray = False,
        eval_mode: bool = False,
    ) -> jax.Array:
        model = self.train_state.model
        key = self.train_state.key
        key, subkey = jax.random.split(key)
        self.train_state = self.train_state._replace(key=key)

        if not self.mpc:
            z = model.encode(obs)
            action, info = model.pi(z, key=subkey)
            if eval_mode:
                return info["mean"]
            return action

        num_envs = obs.shape[0]
        keys = jax.random.split(subkey, num_envs)

        # Single JIT call: encode + MPPI
        # Action bounds are read from model.action_low_tuple/action_high_tuple
        # inside plan_mppi_inner (static fields, no JIT trace impact)
        actions, new_means = plan_mppi_batched(
            model=model,
            obs=obs,
            prev_mean=jnp.asarray(self._prev_mean),
            two_hot_cfg=self.two_hot_cfg,
            discount=self.discount,
            horizon=self.horizon,
            num_samples=self.num_samples,
            num_pi_trajs=self.num_pi_trajs,
            num_elites=self.num_elites,
            num_iterations=self.num_iterations,
            temperature=self.temperature,
            min_std=self.min_std,
            max_std=self.max_std,
            t0_mask=jnp.asarray(t0_mask),
            eval_mode=jnp.array(eval_mode),
            keys=keys,
        )

        self._prev_mean = np.array(new_means)
        return actions

    # ==================== Environment Step Processing ====================

    def process_env_step(
        self,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        infos: Dict[str, Any],
    ) -> None:
        """Process environment step (no-op, transitions stored via store_transition)."""
        pass

    # ==================== Update ====================

    def update(self, batch: SequenceBatch, build_metrics: bool = False) -> TDMPC2Metrics | None:
        self.total_it += 1

        # Update observation normalizer stats from batch
        if self.obs_normalization:
            # batch.observations: [H+1, B, obs_dim] — flatten to update stats
            all_obs = batch.observations.reshape(-1, batch.observations.shape[-1])
            new_normalizer = self.train_state.model.obs_normalizer.update(all_obs)
            new_model = eqx.tree_at(
                lambda m: m.obs_normalizer,
                self.train_state.model,
                new_normalizer,
                is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
            )
            self.train_state = self.train_state._replace(model=new_model)

        key = self.train_state.key
        key, update_key = jax.random.split(key)

        new_model, new_wm_opt, new_pi_opt, new_target_q, info = unified_update(
            model=self.train_state.model,
            wm_opt_state=self.train_state.wm_opt_state,
            wm_optimizer=self.wm_optimizer,
            pi_opt_state=self.train_state.pi_opt_state,
            pi_optimizer=self.pi_optimizer,
            target_q_params=self.train_state.target_q_params,
            tau=self.tau,
            batch=batch,
            two_hot_cfg=self.two_hot_cfg,
            discount=self.discount,
            rho=self.rho,
            entropy_coef=self.entropy_coef,
            consistency_coef=self.consistency_coef,
            reward_coef=self.reward_coef,
            value_coef=self.value_coef,
            scale_value=jnp.array(self.scale.value),
            scale_tau=self.scale.tau,
            key=update_key,
        )

        self.scale._value = float(info.updated_scale_value)

        self.train_state = TDMPC2TrainState(
            model=new_model,
            target_q_params=new_target_q,
            wm_opt_state=new_wm_opt,
            pi_opt_state=new_pi_opt,
            key=key,
        )

        if not build_metrics:
            return None

        return TDMPC2Metrics(
            world_model=TDMPC2WorldModelMetrics(
                consistency_loss=float(info.consistency_loss),
                reward_loss=float(info.reward_loss),
                value_loss=float(info.value_loss),
                total_loss=float(info.total_loss),
                grad_norm=float(info.wm_grad_norm),
            ),
            policy=TDMPC2PolicyMetrics(
                pi_loss=float(info.pi_loss),
                pi_entropy=float(info.pi_entropy),
                pi_scaled_entropy=float(info.pi_scaled_entropy),
                pi_grad_norm=float(info.pi_grad_norm),
                pi_scale=self.scale.value,
            ),
            q=TDMPC2QMetrics(
                mean=float(info.q_mean),
                std=float(info.q_std),
                p05=float(info.q_p05),
                p95=float(info.q_p95),
            ),
            batch=BatchMetrics(
                return_mean=float(batch.rewards.mean()),
                return_std=float(batch.rewards.std()),
                return_min=float(batch.rewards.min()),
                return_max=float(batch.rewards.max()),
            ),
            total_updates=self.total_it,
        )

    # ==================== Checkpointing ====================

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Save TD-MPC2 training state."""
        import os

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_q.eqx")
        eqx.tree_serialise_leaves(target_path, self.train_state.target_q_params)

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.train_state.key),
            "lr": self.lr,
            "pi_lr": self.pi_lr,
            "total_it": self.total_it,
            "scale_value": self.scale.value,
            "prev_mean": self._prev_mean.tolist(),
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """Load TD-MPC2 training state."""
        import os

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        new_model = eqx.tree_deserialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_q.eqx")
        new_target_q = eqx.tree_deserialise_leaves(
            target_path, self.train_state.target_q_params
        )

        # Re-initialize optimizer states
        wm_params, _ = eqx.partition(new_model, eqx.is_inexact_array)
        wm_opt_state = self.wm_optimizer.init(wm_params)

        pi_params, _ = eqx.partition(new_model.policy, eqx.is_inexact_array)
        pi_opt_state = self.pi_optimizer.init(pi_params)

        self.train_state = TDMPC2TrainState(
            model=new_model,
            target_q_params=new_target_q,
            wm_opt_state=wm_opt_state,
            pi_opt_state=pi_opt_state,
            key=jnp.array(metadata["alg_key"]),
        )

        self.total_it = metadata.get("total_it", 0)
        self.scale._value = metadata.get("scale_value", 1.0)
        prev_mean = metadata.get("prev_mean")
        if prev_mean is not None:
            self._prev_mean = np.array(prev_mean, dtype=np.float32)