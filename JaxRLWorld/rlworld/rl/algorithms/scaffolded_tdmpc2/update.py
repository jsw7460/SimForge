"""
Scaffolded unified update with ABD-Net dynamics.

Follows rlworld.rl.algorithms.tdmpc2.update exactly:
- ScaffoldedUnifiedUpdateInfo (NamedTuple, mirrors UnifiedUpdateInfo)
- @eqx.filter_jit on scaffolded_unified_update
- Same gradient flow: WM update -> latent rollout -> policy update -> Polyak

Steps:
1. Target world model update (ABD-Net on s-)
2. Scaffolded world model update (ABD-Net on s+)
3. Latent rollouts (stop_gradient, via lax.scan) for both models
4. Target policy update using scaffolded critic
5. Exploration policy update in s+ space
6. Polyak target Q updates for both models

NOTE: All horizon loops replaced with jax.lax.scan to reduce XLA compile time.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rlworld.rl.algorithms.scaffolded_tdmpc2.losses import (
    compute_abd_world_model_loss,
    compute_scaffolded_world_model_loss,
)
from rlworld.rl.algorithms.scaffolded_tdmpc2.scaffolded_world_model import (
    ScaffoldedWorldModel,
)
from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
)
from rlworld.rl.modules.policies.abd_world_model import ABDNetWorldModel
from rlworld.rl.storages.scaffolded_replay_buffer import (
    ScaffoldedSequenceBatch,
)
from rlworld.rl.storages.sequence_replay_buffer import SequenceBatch


class ScaffoldedUnifiedUpdateInfo(NamedTuple):
    """All metrics from scaffolded update. Mirrors UnifiedUpdateInfo."""

    # Target world model
    target_consistency_loss: jax.Array
    target_reward_loss: jax.Array
    target_value_loss: jax.Array
    target_ortho_loss: jax.Array
    target_total_loss: jax.Array
    target_wm_grad_norm: jax.Array
    # Scaffolded world model
    scaff_consistency_loss: jax.Array
    scaff_reward_loss: jax.Array
    scaff_value_loss: jax.Array
    scaff_ortho_loss: jax.Array
    scaff_total_loss: jax.Array
    scaff_wm_grad_norm: jax.Array
    # Target policy
    pi_loss: jax.Array
    pi_entropy: jax.Array
    pi_scaled_entropy: jax.Array
    pi_grad_norm: jax.Array
    updated_scale_value: jax.Array
    # Exploration policy
    explore_pi_loss: jax.Array
    explore_entropy: jax.Array
    # Q stats (scaffolded, used for policy update)
    q_mean: jax.Array
    q_std: jax.Array
    q_p05: jax.Array
    q_p95: jax.Array
    # Q stats (target, for comparison logging)
    target_q_mean: jax.Array
    target_q_std: jax.Array
    target_q_p05: jax.Array
    target_q_p95: jax.Array


@eqx.filter_jit
def scaffolded_unified_update(
    # Target model
    target_model: ABDNetWorldModel,
    target_wm_opt_state: optax.OptState,
    target_wm_optimizer: optax.GradientTransformation,
    target_q_params: Any,
    # Scaffolded model
    scaffolded_model: ScaffoldedWorldModel,
    scaff_wm_opt_state: optax.OptState,
    scaff_wm_optimizer: optax.GradientTransformation,
    scaff_target_q_params: Any,
    # Policy optimizers
    pi_opt_state: optax.OptState,
    pi_optimizer: optax.GradientTransformation,
    explore_pi_opt_state: optax.OptState,
    explore_pi_optimizer: optax.GradientTransformation,
    # Hyperparams
    tau: float,
    batch: ScaffoldedSequenceBatch,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    rho: float,
    entropy_coef: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    ortho_coef: float,
    scale_value: jax.Array,
    scale_tau: float,
    key: jax.Array,
) -> tuple[
    ABDNetWorldModel,
    optax.OptState,
    Any,
    ScaffoldedWorldModel,
    optax.OptState,
    Any,
    optax.OptState,
    optax.OptState,
    ScaffoldedUnifiedUpdateInfo,
]:
    key, twm_key, swm_key, pi_key, explore_key = jax.random.split(key, 5)

    # Extract batch components
    target_obs = batch.observations  # [H+1, B, obs_dim]
    priv_obs = batch.privileged_observations  # [H+1, B, priv_dim]
    actions = batch.actions  # [H, B, action_dim]
    rewards = batch.rewards  # [H, B, 1]
    terminated = batch.terminated  # [H, B, 1]
    horizon = actions.shape[0]

    # s+ = [s-, s_priv]
    scaffolded_obs = jnp.concatenate([target_obs, priv_obs], axis=-1)

    # Target batch (for target WM loss, uses SequenceBatch)
    target_batch = SequenceBatch(
        observations=target_obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
    )

    # ========== 1. Target World Model Update ==========
    tq_static = eqx.partition(target_model.q_ensemble, eqx.is_inexact_array)[1]
    target_q_ens = eqx.combine(target_q_params, tq_static)

    @eqx.filter_value_and_grad(has_aux=True)
    def target_wm_loss_fn(m: ABDNetWorldModel):
        m_stopped_pi = eqx.tree_at(
            lambda x: x.policy,
            m,
            jax.lax.stop_gradient(m.policy),
        )
        return compute_abd_world_model_loss(
            model=m_stopped_pi,
            target_q_ensemble=target_q_ens,
            batch=target_batch,
            two_hot_cfg=two_hot_cfg,
            discount=discount,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            ortho_coef=ortho_coef,
            key=twm_key,
        )

    (_, twm_info), twm_grads = target_wm_loss_fn(target_model)
    twm_grad_norm = optax.global_norm(twm_grads)
    twm_updates, new_twm_opt = target_wm_optimizer.update(twm_grads, target_wm_opt_state, target_model)
    new_target_model = eqx.apply_updates(target_model, twm_updates)

    # ========== 2. Scaffolded World Model Update ==========
    sq_static = eqx.partition(scaffolded_model.q_ensemble, eqx.is_inexact_array)[1]
    scaff_target_q_ens = eqx.combine(scaff_target_q_params, sq_static)

    @eqx.filter_value_and_grad(has_aux=True)
    def scaff_wm_loss_fn(sm: ScaffoldedWorldModel):
        sm_stopped_pi = eqx.tree_at(
            lambda x: x.exploration_policy,
            sm,
            jax.lax.stop_gradient(sm.exploration_policy),
        )
        return compute_scaffolded_world_model_loss(
            scaffolded_model=sm_stopped_pi,
            scaff_target_q_ensemble=scaff_target_q_ens,
            scaffolded_obs=scaffolded_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            two_hot_cfg=two_hot_cfg,
            discount=discount,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            ortho_coef=ortho_coef,
            key=swm_key,
        )

    (_, swm_info), swm_grads = scaff_wm_loss_fn(scaffolded_model)
    swm_grad_norm = optax.global_norm(swm_grads)
    swm_updates, new_swm_opt = scaff_wm_optimizer.update(swm_grads, scaff_wm_opt_state, scaffolded_model)
    new_scaffolded_model = eqx.apply_updates(scaffolded_model, swm_updates)

    # ========== 3. Latent Rollouts (stop_gradient, via lax.scan) ==========

    # Target rollout on s-
    tz0 = new_target_model.encode(target_obs[0])

    def _target_dynamics_step(z, a):
        z_next = new_target_model.next_latent(z, a)
        return z_next, z_next

    _, target_zs_rolled = jax.lax.scan(_target_dynamics_step, tz0, actions)
    # Prepend initial state: [H+1, B, obs_dim]
    target_zs = jax.lax.stop_gradient(jnp.concatenate([tz0[None], target_zs_rolled], axis=0))

    # Scaffolded rollout on s+
    sz0 = new_scaffolded_model.encode(scaffolded_obs[0])

    def _scaff_dynamics_step(z, a):
        z_next = new_scaffolded_model.next_latent(z, a)
        return z_next, z_next

    _, scaff_zs_rolled = jax.lax.scan(_scaff_dynamics_step, sz0, actions)
    scaff_zs = jax.lax.stop_gradient(jnp.concatenate([sz0[None], scaff_zs_rolled], axis=0))

    # ========== 4. Target Policy Update (scaffolded critic) ==========
    pi_params, pi_static = eqx.partition(new_target_model.policy, eqx.is_inexact_array)

    def pi_loss_fn(p_params):
        new_policy = eqx.combine(p_params, pi_static)
        full_model = eqx.tree_at(lambda m: m.policy, new_target_model, new_policy)

        horizon_plus_one = target_zs.shape[0]
        pi_keys = jax.random.split(pi_key, horizon_plus_one)
        q_keys = jax.random.split(jax.random.fold_in(pi_key, 999), horizon_plus_one)
        rho_weights = rho ** jnp.arange(horizon_plus_one)

        def _single_step(z_target, z_scaffolded, pi_k, q_k):
            # Target policy selects action from s-
            action, info = full_model.pi(z_target, key=pi_k)
            # TODO: Use scaffolded critic (scaff_q_val)
            scaff_q_val = new_scaffolded_model.q_value(
                z_scaffolded,
                action,
                two_hot_cfg,
                return_type="avg",
                key=q_k,
                inference=False,
            )

            # q_val = full_model.q_value(
            #     z_target, action, two_hot_cfg,
            #     return_type="avg", key=q_k, inference=False,
            # )

            # Target critic evaluates with s- (for logging only)
            target_q_val = jax.lax.stop_gradient(
                full_model.q_value(
                    z_target,
                    action,
                    two_hot_cfg,
                    return_type="avg",
                    key=q_k,
                    inference=False,
                )
            )
            return action, info, scaff_q_val, target_q_val

        _, infos, q_vals, target_q_vals = jax.vmap(_single_step)(target_zs, scaff_zs, pi_keys, q_keys)

        # Inline scale computation (matches tdmpc2/update.py exactly)
        qs_flat = jax.lax.stop_gradient(q_vals[0]).flatten()
        p05 = jnp.percentile(qs_flat, 5)
        p95 = jnp.percentile(qs_flat, 95)
        new_scale_val = jnp.maximum(p95 - p05, 1.0)
        inner_scale = (1.0 - scale_tau) * scale_value + scale_tau * new_scale_val

        q_normalized = q_vals / jnp.maximum(inner_scale, 1.0)
        step_losses = -(entropy_coef * infos["scaled_entropy"] + q_normalized).mean(axis=(-1, -2))

        pi_loss = jnp.sum(step_losses * rho_weights) / horizon_plus_one

        return pi_loss, (
            pi_loss,
            infos["entropy"].mean(),
            infos["scaled_entropy"].mean(),
            inner_scale,
            q_vals[0].mean(),
            q_vals[0].std(),
            jnp.percentile(q_vals[0].flatten(), 5),
            jnp.percentile(q_vals[0].flatten(), 95),
            target_q_vals[0].mean(),
            target_q_vals[0].std(),
            jnp.percentile(target_q_vals[0].flatten(), 5),
            jnp.percentile(target_q_vals[0].flatten(), 95),
        )

    (
        (
            _,
            (
                pi_loss,
                pi_entropy,
                pi_scaled_entropy,
                updated_scale,
                q_mean,
                q_std,
                q_p05,
                q_p95,
                target_q_mean,
                target_q_std,
                target_q_p05,
                target_q_p95,
            ),
        ),
        pi_grads,
    ) = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)

    pi_grad_norm = optax.global_norm(pi_grads)
    pi_updates, new_pi_opt = pi_optimizer.update(pi_grads, pi_opt_state, pi_params)
    new_pi_params = optax.apply_updates(pi_params, pi_updates)

    new_policy = eqx.combine(new_pi_params, pi_static)
    final_target_model = eqx.tree_at(lambda m: m.policy, new_target_model, new_policy)

    # ========== 5. Exploration Policy Update ==========
    explore_params, explore_static = eqx.partition(new_scaffolded_model.exploration_policy, eqx.is_inexact_array)

    def explore_loss_fn(ep_params):
        new_explore_pi = eqx.combine(ep_params, explore_static)
        full_scaff = eqx.tree_at(
            lambda m: m.exploration_policy,
            new_scaffolded_model,
            new_explore_pi,
        )

        horizon_plus_one = scaff_zs.shape[0]
        ep_keys = jax.random.split(explore_key, horizon_plus_one)
        eq_keys = jax.random.split(jax.random.fold_in(explore_key, 777), horizon_plus_one)
        rho_weights = rho ** jnp.arange(horizon_plus_one)

        def _single_step(z_plus, pk, qk):
            action, info = full_scaff.pi_explore(z_plus, key=pk)
            q_val = full_scaff.q_value(
                z_plus,
                action,
                two_hot_cfg,
                return_type="avg",
                key=qk,
                inference=False,
            )
            q_norm = q_val / jnp.maximum(updated_scale, 1.0)
            step_loss = -(entropy_coef * info["scaled_entropy"] + q_norm).mean()
            return step_loss, info["entropy"].mean()

        step_losses, entropies = jax.vmap(_single_step)(scaff_zs, ep_keys, eq_keys)
        explore_loss = jnp.sum(step_losses * rho_weights) / horizon_plus_one
        return explore_loss, (explore_loss, entropies.mean())

    (_, (explore_pi_loss, explore_entropy)), explore_grads = jax.value_and_grad(explore_loss_fn, has_aux=True)(
        explore_params
    )

    explore_updates, new_explore_opt = explore_pi_optimizer.update(explore_grads, explore_pi_opt_state, explore_params)
    new_explore_params = optax.apply_updates(explore_params, explore_updates)

    new_explore_pi = eqx.combine(new_explore_params, explore_static)
    final_scaffolded_model = eqx.tree_at(
        lambda m: m.exploration_policy,
        new_scaffolded_model,
        new_explore_pi,
    )

    # ========== 6. Polyak Target Q Updates ==========
    tq_params, _ = eqx.partition(final_target_model.q_ensemble, eqx.is_inexact_array)
    new_target_q_params = jax.tree.map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        tq_params,
        target_q_params,
    )

    sq_params, _ = eqx.partition(final_scaffolded_model.q_ensemble, eqx.is_inexact_array)
    new_scaff_target_q_params = jax.tree.map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        sq_params,
        scaff_target_q_params,
    )

    info = ScaffoldedUnifiedUpdateInfo(
        target_consistency_loss=twm_info.consistency_loss,
        target_reward_loss=twm_info.reward_loss,
        target_value_loss=twm_info.value_loss,
        target_ortho_loss=twm_info.orthogonality_loss,
        target_total_loss=twm_info.total_loss,
        target_wm_grad_norm=twm_grad_norm,
        scaff_consistency_loss=swm_info.consistency_loss,
        scaff_reward_loss=swm_info.reward_loss,
        scaff_value_loss=swm_info.value_loss,
        scaff_ortho_loss=swm_info.orthogonality_loss,
        scaff_total_loss=swm_info.total_loss,
        scaff_wm_grad_norm=swm_grad_norm,
        pi_loss=pi_loss,
        pi_entropy=pi_entropy,
        pi_scaled_entropy=pi_scaled_entropy,
        pi_grad_norm=pi_grad_norm,
        updated_scale_value=updated_scale,
        explore_pi_loss=explore_pi_loss,
        explore_entropy=explore_entropy,
        q_mean=q_mean,
        q_std=q_std,
        q_p05=q_p05,
        q_p95=q_p95,
        target_q_mean=target_q_mean,
        target_q_std=target_q_std,
        target_q_p05=target_q_p05,
        target_q_p95=target_q_p95,
    )

    return (
        final_target_model,
        new_twm_opt,
        new_target_q_params,
        final_scaffolded_model,
        new_swm_opt,
        new_scaff_target_q_params,
        new_pi_opt,
        new_explore_opt,
        info,
    )
