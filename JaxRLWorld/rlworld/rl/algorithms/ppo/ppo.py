import os
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.rl.algorithms.base import (
    ActInput,
    OnPolicyAlgorithm,
    create_optimizer_with_labels,
)
from rlworld.rl.algorithms.metrics import BatchMetrics
from rlworld.rl.algorithms.ppo.metrics import (
    PPOActorMetrics,
    PPOCriticMetrics,
    PPOKLMetrics,
    PPOMetrics,
)
from rlworld.rl.algorithms.ppo.update import (
    ScanOutput,
    forward_policy_and_value,
    forward_policy_and_value_deterministic,
    get_value,
    update_all_batches,
)
from rlworld.rl.modules.normalization import EmpiricalNormalization
from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.rl.storages.rollout_storage import RolloutStorage


def _as_obs_shape(shape):
    """Storage shape spec: one tuple, or one per observation group."""
    if isinstance(shape, dict):
        return {group: tuple(group_shape) for group, group_shape in shape.items()}
    return tuple(shape)


@eqx.filter_jit
def _evaluate_bootstrap_values(model: PPOActorCritic, critic_obs: jax.Array) -> jax.Array:
    """Critic forward for the timeout bootstrap.

    Compiled, like every other forward in the collection loop. Called as
    a plain method it runs eagerly — one dispatch per layer, per
    activation, per normalizer op — which cost 4.1 ms per step against
    0.79 ms for the jitted actor AND critic together, on the same batch.
    A truncation happens somewhere in the batch on essentially every
    step, so this runs on essentially every step.
    """
    values, _ = model.evaluate_value(critic_obs)
    return values.squeeze(-1)


@eqx.filter_jit
def _update_normalizers(
    model: PPOActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
) -> PPOActorCritic:
    """JIT-compiled normalizer update.

    Only the state vector feeds the running statistics — images are not
    normalized, matching rsl_rl's CNNModel.
    """
    new_actor_normalizer = model.actor_obs_normalizer.update(model._actor_vector(actor_obs))
    new_critic_normalizer = model.critic_obs_normalizer.update(model._critic_vector(critic_obs))
    return eqx.tree_at(
        lambda m: (m.actor_obs_normalizer, m.critic_obs_normalizer),
        model,
        (new_actor_normalizer, new_critic_normalizer),
    )


class PPOTrainState(NamedTuple):
    """Training state for PPO."""

    model: PPOActorCritic
    opt_state: optax.OptState
    key: jax.Array


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) with scan-based updates.

    Features:
    - Single JIT-compiled update over all minibatches
    - Separate learning rates for actor/critic
    - Reward scaling
    - GAE advantage estimation
    """

    ActInput = ActInput

    @dataclass
    class TransitionBuffer:
        """Buffer for current transition."""

        actor_observations: jax.Array = None
        critic_observations: jax.Array = None
        actions: jax.Array = None
        rewards: jax.Array = None
        dones: jax.Array = None
        values: jax.Array = None
        actions_log_prob: jax.Array = None
        action_mean: jax.Array = None
        action_sigma: jax.Array = None
        episode_starts: jax.Array = None

        def clear(self):
            self.actor_observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.episode_starts = None

    def __init__(
        self,
        actor_critic: PPOActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        use_value_normalization: bool = False,
        use_early_stop: bool = False,
        normalize_advantage_per_minibatch: bool = True,
        symmetry_spec=None,
        symmetry_coef: float = 0.0,
        optimizer_class=None,
        key: jax.Array = None,
        **kwargs,
    ):
        """
        Initialize PPO algorithm.

        Args:
            actor_critic: Neural network for policy and value function
            num_learning_epochs: Number of epochs per update
            num_mini_batches: Number of minibatches per epoch
            clip_param: PPO clipping parameter
            gamma: Discount factor
            lam: GAE lambda parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            max_grad_norm: Maximum gradient norm
            use_clipped_value_loss: Whether to clip value loss
            schedule: LR schedule ('fixed' or 'adaptive')
            desired_kl: Target KL for adaptive LR
            use_value_normalization: When True, the critic learns in normalized
                value space; collected V outputs are inverse-normalized for GAE
                and storage. When False (default), behavior is identical to a
                pure PPO with no return/value normalization.
            use_early_stop: Whether to use KL-based early stopping
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        super().__init__(
            actor_critic=actor_critic,
            gamma=gamma,
            gae_lambda=lam,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            key=key,
        )

        # Store hyperparameters
        self.clip_param = clip_param
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.use_value_normalization = use_value_normalization
        self.use_early_stop = use_early_stop
        self.normalize_advantage_per_minibatch = normalize_advantage_per_minibatch
        # Left/right mirror symmetry: static (perm/sign arrays) built once by the
        # runner from the env's obs layout; coef 0 disables (no trace cost).
        self.symmetry_spec = symmetry_spec
        self.symmetry_coef = symmetry_coef
        self.optimizer_class = optimizer_class or optax.adam

        # Check if model has normalizers enabled
        self.obs_normalization = actor_critic.actor_obs_normalizer is not None

        # Create optimizer
        model_for_optimizer = actor_critic
        if self.obs_normalization:
            model_for_optimizer = eqx.tree_at(
                lambda m: (m.actor_obs_normalizer, m.critic_obs_normalizer),
                actor_critic,
                (None, None),
            )
        self.optimizer = self._create_optimizers(model_for_optimizer)

        # Initialize training state
        params, static = eqx.partition(
            actor_critic,
            eqx.is_inexact_array,
            is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
        )
        self._static = static
        opt_state = self.optimizer.init(params)

        self.train_state = PPOTrainState(
            model=actor_critic,
            opt_state=opt_state,
            key=key,
        )

        # Storage (initialized later via init_storage)
        self.storage: RolloutStorage | None = None
        self.transition: PPO.TransitionBuffer | None = None

        # Value-target normalizer (skrl-style). Only instantiated when
        # ``use_value_normalization=True``. When None, the critic learns
        # in raw return space — identical to pure PPO with no value
        # normalization.
        self.value_normalizer: EmpiricalNormalization | None = (
            EmpiricalNormalization(shape=1) if use_value_normalization else None
        )

        # Last dones for GAE computation
        self._last_dones: jax.Array | None = None

    @property
    def actor_critic(self):
        return self.train_state.model

    def _create_optimizers(self, model: PPOActorCritic) -> optax.GradientTransformation:
        """Create optimizer with separate learning rates for actor/critic/std."""

        def label_fn(path):
            path_str = ".".join(str(p) for p in path)
            if "actor" in path_str:
                return "actor"
            elif "critic" in path_str:
                return "critic"
            elif "std_module" in path_str:
                return "std"
            return "actor"

        lr_config = {
            "actor": self.actor_lr,
            "critic": self.critic_lr,
            "std": self.actor_lr,
        }

        optimizer, self._param_labels = create_optimizer_with_labels(
            model=model,
            label_fn=label_fn,
            lr_config=lr_config,
            max_grad_norm=self.max_grad_norm,
            optimizer_class=self.optimizer_class,
        )

        return optimizer

    def init_storage(self, cfg: Dict[str, Any]) -> None:
        """Initialize rollout storage for experience collection."""
        self.storage = RolloutStorage(
            num_envs=cfg["num_envs"],
            num_steps=cfg["num_transitions_per_env"],
            actor_obs_shape=_as_obs_shape(cfg["actor_obs_shape"]),
            critic_obs_shape=_as_obs_shape(cfg["critic_obs_shape"]),
            action_shape=tuple(cfg["actions_shape"]),
        )
        self.transition = PPO.TransitionBuffer()

    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """Select action given observation.

        Returns env_actions (squashed if applicable) for environment stepping.
        Stores raw_actions (pre-tanh if squashed) in transition for PPO update.
        """
        model = self.train_state.model

        key = self.train_state.key
        key, subkey = jax.random.split(key)
        self.train_state = self.train_state._replace(key=key)

        if deterministic:
            env_actions, raw_actions, mean, std, log_prob, values, _ = forward_policy_and_value_deterministic(
                model, obs.actor_obs, obs.critic_obs, subkey
            )
        else:
            env_actions, raw_actions, mean, std, log_prob, values, _ = forward_policy_and_value(
                model, obs.actor_obs, obs.critic_obs, subkey
            )

        # Hook 1 (value normalization, opt-in): the critic outputs in
        # normalized return space; convert back to raw before storing
        # so GAE / bootstrap / storage all stay in raw space.
        if self.value_normalizer is not None:
            values = self.value_normalizer.unnormalize(values[..., None]).squeeze(-1)

        # Store raw (pre-tanh) actions for numerically stable PPO update
        self.transition.actions = raw_actions
        self.transition.values = values
        self.transition.actions_log_prob = log_prob
        self.transition.action_mean = mean
        self.transition.action_sigma = std
        self.transition.actor_observations = obs.actor_obs
        self.transition.critic_observations = obs.critic_obs
        # Return squashed actions for environment
        return env_actions

    def process_env_step(
        self,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        infos: Dict[str, Any],
        next_actor_obs: jax.Array = None,
        next_critic_obs: jax.Array = None,
    ) -> None:
        """Process environment step and store transition.

        NOTE: Observation normalizer is NOT updated here. It is updated
        once per iteration in update() after the gradient step, to ensure
        consistency between collection-time and update-time normalization
        (matching Brax PPO behavior).
        """
        dones = terminated | truncated

        # Transition assignment
        self.transition.rewards = rewards
        self.transition.dones = dones
        if self._last_dones is None:
            self.transition.episode_starts = jnp.zeros_like(dones)
        else:
            self.transition.episode_starts = self._last_dones

        # Handle timeout (truncated-only; vectorized — no host sync)
        self._handle_timeout(truncated, terminated, infos)

        # Add to storage
        self.storage.add_transition(
            actor_obs=self.transition.actor_observations,
            critic_obs=self.transition.critic_observations,
            actions=self.transition.actions,
            rewards=self.transition.rewards,
            dones=self.transition.dones,
            episode_starts=self.transition.episode_starts,
            values=self.transition.values,
            log_probs=self.transition.actions_log_prob,
            mu=self.transition.action_mean,
            sigma=self.transition.action_sigma,
        )

        # Clear and update
        self.transition.clear()
        self._last_dones = dones

    def _handle_timeout(
        self,
        truncated: jax.Array,
        terminated: jax.Array,
        infos: Dict[str, Any],
    ) -> None:
        """Bootstrap terminal value for truncated-only episodes (vectorized).

        Adds ``gamma * V(final_obs)`` to the reward for env-slots that hit a
        time-limit truncation but did NOT genuinely terminate. The
        ``truncated & ~terminated`` mask guards the rare but real case where
        an env emits both flags on the same step (e.g. max-episode-length is
        reached on the same physics step a fall is detected) — without the
        guard we would over-bootstrap a real termination.

        Implementation runs V on the full ``(num_envs, obs_dim)`` final-obs
        tensor and multiplies by the mask, so there is no host-device sync,
        no dynamic-shape indexing, and no JIT recompilation pressure on
        truncated counts.
        """
        if "final_observation" not in infos:
            return

        final_critic = infos["final_observation"]["critic"]
        bootstrap_values = _evaluate_bootstrap_values(self.train_state.model, final_critic)

        # Hook 2 (value normalization, opt-in): critic outputs in
        # normalized space; convert to raw before adding to raw rewards.
        if self.value_normalizer is not None:
            bootstrap_values = self.value_normalizer.unnormalize(bootstrap_values[..., None]).squeeze(-1)

        # Prefer the env-provided bootstrap mask (termination manager builds it
        # from per-term ``bootstrap_value``; ManiSkill adapter from ``success``).
        # Falls back to ``truncated & ~terminated`` for envs that don't provide
        # one — identical to the historical behaviour.
        bootstrap_mask = infos.get("bootstrap_mask")
        if bootstrap_mask is None:
            bootstrap_mask = truncated & ~terminated
        bonus = bootstrap_mask.astype(self.transition.rewards.dtype) * (self.gamma * bootstrap_values)
        self.transition.rewards = self.transition.rewards + bonus

    def compute_returns(self, last_critic_obs: jax.Array) -> None:
        """
        Compute returns and advantages using GAE.

        Args:
            last_critic_obs: Critic observations for the last state
        """
        last_values = _evaluate_bootstrap_values(self.train_state.model, last_critic_obs)

        # Hook 3a (value normalization, opt-in): last_values comes from the
        # critic in normalized space; GAE runs in raw space along with the
        # raw values stored in `act()`, so unnormalize first.
        if self.value_normalizer is not None:
            last_values = self.value_normalizer.unnormalize(last_values[..., None]).squeeze(-1)

        self.storage.compute_returns(
            last_values=last_values,
            last_dones=self._last_dones,
            gamma=self.gamma,
            gae_lambda=self.lam,
        )

        # Hook 3b (value normalization, opt-in): now that returns/values are
        # computed in raw space, fold raw returns into the running stats and
        # rewrite storage.values / storage.returns into normalized space so
        # the critic loss in compute_batch_loss naturally targets ~unit
        # variance. Advantages are NOT normalized here — they keep raw
        # scale (PPO's per-minibatch / per-rollout advantage normalization
        # below handles that separately).
        if self.value_normalizer is not None:
            raw_returns = self.storage.returns
            raw_values = self.storage.values
            flat_returns = raw_returns.reshape(-1, 1)
            self.value_normalizer = self.value_normalizer.update(flat_returns)
            self.storage.returns = self.value_normalizer.normalize(raw_returns[..., None]).squeeze(-1)
            self.storage.values = self.value_normalizer.normalize(raw_values[..., None]).squeeze(-1)

        # Per-rollout advantage normalization (rsl_rl default).
        # When per-minibatch is selected, the same statistic is computed inside
        # compute_batch_loss instead.
        if not self.normalize_advantage_per_minibatch:
            self.storage.normalize_advantages()

    def update(self) -> PPOMetrics:
        """Update policy and value networks with early stopping and adaptive LR."""
        key = self.train_state.key
        key, subkey = jax.random.split(key)

        # The rollout stays flat (views, no copy); the update's scan
        # gathers one shuffled minibatch per step via these index rows.
        flat_batch = self.storage.get_flat_batch()
        batch_indices = self.storage.get_minibatch_indices(
            num_minibatches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
            key=subkey,
        )

        params, static = eqx.partition(
            self.train_state.model,
            eqx.is_inexact_array,
            is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
        )

        # Handle None desired_kl (use large value to effectively disable early stop)
        desired_kl = self.desired_kl if self.desired_kl is not None else 1e10

        new_params, new_opt_state, outputs, new_key = update_all_batches(
            params,
            static,
            self.train_state.opt_state,
            self.optimizer,
            self.clip_param,
            self.value_loss_coef,
            self.entropy_coef,
            self.use_clipped_value_loss,
            self.normalize_advantage_per_minibatch,
            self.use_early_stop,
            desired_kl,
            self.symmetry_spec,
            self.symmetry_coef,
            flat_batch,
            batch_indices,
            subkey,
        )

        new_model = eqx.combine(new_params, static)

        self.train_state = PPOTrainState(
            model=new_model,
            opt_state=new_opt_state,
            key=new_key,
        )

        # Compute metrics
        metrics = self._compute_metrics(outputs, flat_batch, batch_indices)

        # Adaptive learning rate based on KL divergence.
        # Use the analytical KL averaged over actually-applied minibatches —
        # lower-variance signal than approx_kl, matches rsl_rl behavior.
        if self.schedule == "adaptive" and self.desired_kl is not None:
            did_update = outputs.did_update
            num_actual_updates = int(did_update.sum())
            if num_actual_updates > 0:
                update_mask = did_update.astype(jnp.float32)
                analytical_kl_mean = float((outputs.analytical_kl * update_mask).sum() / num_actual_updates)
            else:
                analytical_kl_mean = float(outputs.analytical_kl.mean())
            self._adaptive_learning_rate(analytical_kl_mean)

        # Update observation normalizers with all collected observations.
        # Done AFTER gradient update so that rollout, returns, and loss
        # computation all used the same (frozen) normalizer.
        if self.obs_normalization:
            flat_actor, flat_critic = self.storage.get_flat_observations()
            new_model = _update_normalizers(
                self.train_state.model,
                flat_actor,
                flat_critic,
            )
            self.train_state = self.train_state._replace(model=new_model)

        self.storage.clear()

        return metrics

    def _compute_metrics(self, outputs: ScanOutput, flat_batch, batch_indices) -> PPOMetrics:
        """Compute metrics from update outputs."""
        did_update = outputs.did_update
        num_actual_updates = int(did_update.sum())
        num_expected_updates = self.num_learning_epochs * self.num_mini_batches

        # Compute means only from updated batches
        if num_actual_updates > 0:
            update_mask = did_update.astype(jnp.float32)
            mean_value_loss = float((outputs.value_loss * update_mask).sum() / num_actual_updates)
            mean_policy_loss = float((outputs.policy_loss * update_mask).sum() / num_actual_updates)
            mean_entropy = float((outputs.entropy * update_mask).sum() / num_actual_updates)
            mean_approx_kl = float((outputs.approx_kl * update_mask).sum() / num_actual_updates)
            mean_clip_fraction = float((outputs.clip_fraction * update_mask).sum() / num_actual_updates)
            mean_mirror_loss = float((outputs.aux["mirror_loss"] * update_mask).sum() / num_actual_updates)
        else:
            mean_value_loss = float(outputs.value_loss.mean())
            mean_policy_loss = float(outputs.policy_loss.mean())
            mean_entropy = float(outputs.entropy.mean())
            mean_approx_kl = float(outputs.approx_kl.mean())
            mean_clip_fraction = float(outputs.clip_fraction.mean())
            mean_mirror_loss = float(outputs.aux["mirror_loss"].mean())

        # Get current std. Every std module reads the state vector (a
        # vision policy's images are not part of it), so take the first
        # minibatch of that — gathered from the flat rollout by the same
        # index row the update's first scan step used.
        model = self.train_state.model
        mb0_obs = jax.tree.map(lambda x: x[batch_indices[0]], flat_batch.actor_observations)
        sample_obs = model._actor_vector(mb0_obs)
        current_std = float(model.std_module(sample_obs).mean())

        # Early stop ratio
        early_stop_ratio = 1.0 - (num_actual_updates / num_expected_updates)

        # Batch statistics over the flat rollout.  (The old pre-gathered
        # layout averaged over num_epochs permuted copies of the same
        # samples — mathematically the same statistics.)
        actions = flat_batch.actions
        returns = flat_batch.returns

        return PPOMetrics(
            critic=PPOCriticMetrics(
                value_loss=mean_value_loss,
            ),
            actor=PPOActorMetrics(
                policy_loss=mean_policy_loss,
                mirror_loss=mean_mirror_loss,
                entropy=mean_entropy,
                std=current_std,
            ),
            kl=PPOKLMetrics(
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

    def _adaptive_learning_rate(self, kl_mean: float) -> None:
        """
        Adaptive learning rate based on KL divergence.

        Adjusts actor learning rate to maintain KL divergence near desired_kl.
        """
        if kl_mean > self.desired_kl * 2.0:
            self._set_actor_lr(max(1e-5, self.actor_lr / 1.5))
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self._set_actor_lr(min(1e-2, self.actor_lr * 1.5))

    def _set_actor_lr(self, lr: float) -> None:
        """Change the actor learning rate, keeping the optimizer's state.

        This used to rebuild the optimizer and call ``init`` again, which
        returns ZEROED Adam moments — so every adjustment threw away the
        gradient history, most often early on where the KL swings hardest
        and the rate is adjusted most. It also replaced the optimizer
        object, forcing a recompile of the update step each time.

        The rate now lives in the optimizer state (see
        ``create_optimizer_with_labels``), so it is written in place: the
        moments survive and nothing is retraced. The critic keeps its own
        rate, which the adaptive schedule does not touch.
        """
        self.actor_lr = lr
        # The std module is trained at the actor's rate, so it follows.
        self._write_injected_lr(("actor", "std"), lr)

    def _write_injected_lr(self, labels: tuple[str, ...], lr: float) -> None:
        """Write ``lr`` into the optimizer state for the given labels."""
        # create_optimizer_with_labels builds chain(clip, multi_transform),
        # so element 1 is the multi_transform state: one inner state per label.
        inner_states = self.train_state.opt_state[1].inner_states
        for label in labels:
            hyperparams = inner_states[label].inner_state.hyperparams
            hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)

    def get_value(self, critic_obs: jax.Array) -> jax.Array:
        """Get value estimate for observations."""
        return get_value(self.train_state.model, critic_obs)

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Save PPO training state.

        Args:
            checkpoint_dir: Directory to save state files

        Returns:
            Algorithm-specific metadata to include in checkpoint
        """
        model_path = os.path.join(checkpoint_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.train_state.model)

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.train_state.key),
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """
        Load PPO training state.

        Args:
            checkpoint_dir: Directory containing state files
            metadata: Metadata dictionary from checkpoint
        """
        model_path = os.path.join(checkpoint_dir, "model.eqx")
        new_model = eqx.tree_deserialise_leaves(model_path, self.train_state.model)

        new_params, _ = eqx.partition(
            new_model,
            eqx.is_inexact_array,
            is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
        )
        new_opt_state = self.optimizer.init(new_params)

        self.train_state = PPOTrainState(
            model=new_model,
            opt_state=new_opt_state,
            key=jnp.array(metadata["alg_key"], dtype=jnp.uint32),
        )

        self.actor_lr = metadata.get("actor_lr", self.actor_lr)
        self.critic_lr = metadata.get("critic_lr", self.critic_lr)

        # ``self.optimizer`` was built from the config's rates, so the fresh
        # opt_state carries those and not the ones the run had reached. The
        # adaptive schedule would eventually pull the actor back, but never
        # the critic, and a run resumed from a decayed rate would silently
        # restart at the config value.
        self._write_injected_lr(("actor", "std"), self.actor_lr)
        self._write_injected_lr(("critic",), self.critic_lr)
