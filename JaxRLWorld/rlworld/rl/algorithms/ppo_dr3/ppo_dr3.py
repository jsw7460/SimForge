from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.algorithms.metrics import BatchMetrics
from rlworld.rl.algorithms.ppo.ppo import PPO, PPOTrainState
from .metrics import (
    PPODR3Metrics,
    PPODR3CriticMetrics,
    PPODR3ActorMetrics,
    PPODR3KLMetrics,
)
from rlworld.rl.modules.policies import PPODR3ActorCritic
from .update import update_all_batches_dr3, ScanOutput


class PPODR3(PPO):
    """
    Proximal Policy Optimization with DR3 Regularization.

    DR3 adds an explicit regularizer to prevent feature co-adaptation
    between consecutive state features in the critic network:

        R_DR3 = E[phi(s)^T @ phi(s')]

    This regularizer is minimized to encourage dissimilar features
    for consecutive states, improving training stability.

    Features:
    - Inherits all PPO features (clipping, GAE, early stopping, etc.)
    - Adds DR3 regularization on critic features
    - Requires PPODR3ActorCritic with DR3Critic
    """

    def __init__(
        self,
        actor_critic: PPODR3ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        dr3_coef: float = 0.001,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        use_reward_scaling: bool = True,
        use_early_stop: bool = False,
        key: jax.Array = None,
        **kwargs,
    ):
        """
        Initialize PPO-DR3 algorithm.

        Args:
            actor_critic: PPODR3ActorCritic network with DR3-compatible critic
            num_learning_epochs: Number of epochs per update
            num_mini_batches: Number of minibatches per epoch
            clip_param: PPO clipping parameter
            gamma: Discount factor
            lam: GAE lambda parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            dr3_coef: DR3 regularization coefficient (default: 0.001)
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            max_grad_norm: Maximum gradient norm
            use_clipped_value_loss: Whether to clip value loss
            schedule: LR schedule ('fixed' or 'adaptive')
            desired_kl: Target KL for adaptive LR
            use_reward_scaling: Whether to scale rewards
            use_early_stop: Whether to use KL-based early stopping
            key: JAX random key
        """
        # Store DR3 coefficient before calling super().__init__
        self.dr3_coef = dr3_coef

        # Call parent init
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            use_reward_scaling=use_reward_scaling,
            use_early_stop=use_early_stop,
            key=key,
            **kwargs,
        )

        print(f"🔬 DR3 regularization enabled with coefficient: {self.dr3_coef}")

    def update(self) -> PPODR3Metrics:
        """Update policy and value networks with DR3 regularization."""
        key = self.train_state.key
        key, subkey = jax.random.split(key)

        stacked_batches = self.storage.get_stacked_batches(
            num_minibatches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
            key=subkey,
        )

        params, static = eqx.partition(self.train_state.model, eqx.is_inexact_array)

        # Handle None desired_kl
        desired_kl = self.desired_kl if self.desired_kl is not None else 1e10

        # Use DR3-specific update function
        new_params, new_opt_state, outputs, new_key = update_all_batches_dr3(
            params,
            static,
            self.train_state.opt_state,
            self.optimizer,
            self.clip_param,
            self.value_loss_coef,
            self.entropy_coef,
            self.dr3_coef,
            self.use_clipped_value_loss,
            True,  # normalize_advantages
            self.use_early_stop,
            desired_kl,
            stacked_batches,
            subkey,
        )

        new_model = eqx.combine(new_params, static)
        self.train_state = PPOTrainState(
            model=new_model,
            opt_state=new_opt_state,
            key=new_key,
        )

        # Compute metrics (DR3-specific)
        metrics = self._compute_dr3_metrics(outputs, stacked_batches)

        # Adaptive learning rate
        if self.schedule == "adaptive" and self.desired_kl is not None:
            self._adaptive_learning_rate(metrics.kl.approx_kl)

        self.storage.clear()

        return metrics

    def _compute_dr3_metrics(
        self, outputs: ScanOutput, stacked_batches
    ) -> PPODR3Metrics:
        """Compute metrics from update outputs including DR3 metrics."""
        did_update = outputs.did_update
        num_actual_updates = int(did_update.sum())
        num_expected_updates = self.num_learning_epochs * self.num_mini_batches

        # Compute means only from updated batches
        if num_actual_updates > 0:
            update_mask = did_update.astype(jnp.float32)
            mean_value_loss = float((outputs.value_loss * update_mask).sum() / num_actual_updates)
            mean_policy_loss = float((outputs.policy_loss * update_mask).sum() / num_actual_updates)
            mean_dr3_loss = float((outputs.dr3_loss * update_mask).sum() / num_actual_updates)
            mean_entropy = float((outputs.entropy * update_mask).sum() / num_actual_updates)
            mean_approx_kl = float((outputs.approx_kl * update_mask).sum() / num_actual_updates)
            mean_clip_fraction = float((outputs.clip_fraction * update_mask).sum() / num_actual_updates)
            mean_feature_dot = float((outputs.feature_dot_product * update_mask).sum() / num_actual_updates)
            mean_cosine_sim = float((outputs.feature_cosine_similarity * update_mask).sum() / num_actual_updates)
            mean_feature_norm = float((outputs.feature_norm * update_mask).sum() / num_actual_updates)
        else:
            mean_value_loss = float(outputs.value_loss.mean())
            mean_policy_loss = float(outputs.policy_loss.mean())
            mean_dr3_loss = float(outputs.dr3_loss.mean())
            mean_entropy = float(outputs.entropy.mean())
            mean_approx_kl = float(outputs.approx_kl.mean())
            mean_clip_fraction = float(outputs.clip_fraction.mean())
            mean_feature_dot = float(outputs.feature_dot_product.mean())
            mean_cosine_sim = float(outputs.feature_cosine_similarity.mean())
            mean_feature_norm = float(outputs.feature_norm.mean())

        # Get current std
        sample_obs = stacked_batches.actor_observations[0]
        current_std = float(self.train_state.model.std_module(sample_obs).mean())

        # Early stop ratio
        early_stop_ratio = 1.0 - (num_actual_updates / num_expected_updates)

        # Batch statistics
        actions = stacked_batches.actions
        returns = stacked_batches.returns

        return PPODR3Metrics(
            critic=PPODR3CriticMetrics(
                value_loss=mean_value_loss,
                dr3_loss=mean_dr3_loss,
                feature_dot_product=mean_feature_dot,
                feature_cosine_similarity=mean_cosine_sim,
                feature_norm=mean_feature_norm,
            ),
            actor=PPODR3ActorMetrics(
                policy_loss=mean_policy_loss,
                entropy=mean_entropy,
                std=current_std,
            ),
            kl=PPODR3KLMetrics(
                approx_kl=mean_approx_kl,
                clip_fraction=mean_clip_fraction,
                early_stop_ratio=early_stop_ratio,
                actual_updates=num_actual_updates,
                expected_updates=num_expected_updates,
            ),
            batch=BatchMetrics(
                return_mean=float(returns.mean()),
                return_std=float(returns.std()),
                return_min=float(returns.min()),
                return_max=float(returns.max()),
                action_mean=float(actions.mean()),
                action_std=float(actions.std()),
            ),
            learning_rate=self.actor_lr,
        )

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Save PPO-DR3 training state."""
        import os
        import numpy as np

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.train_state.model)

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.train_state.key),
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "dr3_coef": self.dr3_coef,
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """Load PPO-DR3 training state."""
        import os

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        new_model = eqx.tree_deserialise_leaves(model_path, self.train_state.model)

        new_params, _ = eqx.partition(new_model, eqx.is_inexact_array)
        new_opt_state = self.optimizer.init(new_params)

        self.train_state = PPOTrainState(
            model=new_model,
            opt_state=new_opt_state,
            key=jnp.array(metadata["alg_key"]),
        )

        self.actor_lr = metadata.get("actor_lr", self.actor_lr)
        self.critic_lr = metadata.get("critic_lr", self.critic_lr)
        self.dr3_coef = metadata.get("dr3_coef", self.dr3_coef)
