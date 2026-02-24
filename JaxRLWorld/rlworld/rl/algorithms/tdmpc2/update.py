# from typing import Any, NamedTuple
#
# import equinox as eqx
# import jax
# import jax.numpy as jnp
# import optax
#
# from rlworld.rl.algorithms.tdmpc2.losses import (
#     compute_world_model_loss,
# )
# from rlworld.rl.algorithms.tdmpc2.math import (
#     TwoHotConfig,
#     two_hot_inv,
#     gumbel_softmax_sample,
# )
# from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel, QEnsemble
# from rlworld.rl.storages.sequence_replay_buffer import SequenceBatch
#
#
# class UnifiedUpdateInfo(NamedTuple):
#     """All metrics from unified update."""
#     # World model
#     consistency_loss: jax.Array
#     reward_loss: jax.Array
#     value_loss: jax.Array
#     total_loss: jax.Array
#     wm_grad_norm: jax.Array
#     # Policy
#     pi_loss: jax.Array
#     pi_entropy: jax.Array
#     pi_scaled_entropy: jax.Array
#     pi_grad_norm: jax.Array
#     updated_scale_value: jax.Array
#     q_mean: jax.Array
#     q_std: jax.Array
#     q_p05: jax.Array
#     q_p95: jax.Array
#
#
# @eqx.filter_jit
# def unified_update(
#     model: TDMPC2WorldModel,
#     wm_opt_state: optax.OptState,
#     wm_optimizer: optax.GradientTransformation,
#     pi_opt_state: optax.OptState,
#     pi_optimizer: optax.GradientTransformation,
#     target_q_params: Any,
#     tau: float,
#     batch: SequenceBatch,
#     two_hot_cfg: TwoHotConfig,
#     discount: float,
#     rho: float,
#     entropy_coef: float,
#     consistency_coef: float,
#     reward_coef: float,
#     value_coef: float,
#     scale_value: jax.Array,
#     scale_tau: float,
#     key: jax.Array,
# ) -> tuple[
#     TDMPC2WorldModel,
#     optax.OptState,
#     optax.OptState,
#     Any,
#     UnifiedUpdateInfo,
# ]:
#     """
#     Full TD-MPC2 update in a single JIT boundary.
#
#     Steps:
#     1. World model update (encoder, dynamics, reward, Q-ensemble)
#     2. Latent rollout (for policy update, stop_gradient)
#     3. Policy update with inline scale computation (matches author)
#     4. Polyak target Q update
#
#     Returns:
#         (new_model, new_wm_opt, new_pi_opt, new_target_q_params, info)
#     """
#     key, wm_key, pi_key = jax.random.split(key, 3)
#
#     # Reconstruct target Q inside JIT
#     q_static = eqx.partition(model.q_ensemble, eqx.is_inexact_array)[1]
#     target_q_ensemble = eqx.combine(target_q_params, q_static)
#
#     # ========== 1. World Model Update ==========
#     @eqx.filter_value_and_grad(has_aux=True)
#     def wm_loss_fn(m: TDMPC2WorldModel):
#         m_with_stopped_pi = eqx.tree_at(
#             lambda x: x.policy,
#             m,
#             jax.lax.stop_gradient(m.policy),
#         )
#         _loss, _info = compute_world_model_loss(
#             model=m_with_stopped_pi,
#             target_q_ensemble=target_q_ensemble,
#             batch=batch,
#             two_hot_cfg=two_hot_cfg,
#             discount=discount,
#             rho=rho,
#             consistency_coef=consistency_coef,
#             reward_coef=reward_coef,
#             value_coef=value_coef,
#             key=wm_key,
#         )
#         return _loss, _info
#
#     (wm_loss, wm_info), wm_grads = wm_loss_fn(model)
#     wm_grad_norm = optax.global_norm(wm_grads)
#     wm_updates, new_wm_opt_state = wm_optimizer.update(wm_grads, wm_opt_state, model)
#     new_model = eqx.apply_updates(model, wm_updates)
#
#     # ========== 2. Latent Rollout ==========
#     obs = batch.observations
#     actions = batch.actions
#     horizon = actions.shape[0]
#
#     z = new_model.encode(obs[0])
#     zs = [z]
#     for t in range(horizon):
#         z = new_model.next_latent(z, actions[t])
#         zs.append(z)
#     zs = jax.lax.stop_gradient(jnp.stack(zs, axis=0))
#
#     # ========== 3. Policy Update ==========
#     pi_params, pi_static = eqx.partition(new_model.policy, eqx.is_inexact_array)
#
#     def pi_loss_fn(p_params):
#         new_policy = eqx.combine(p_params, pi_static)
#         full_model = eqx.tree_at(lambda m: m.policy, new_model, new_policy)
#
#         horizon_plus_one = zs.shape[0]
#         pi_keys = jax.random.split(pi_key, horizon_plus_one)
#         q_keys = jax.random.split(jax.random.fold_in(pi_key, 999), horizon_plus_one)
#         rho_weights = rho ** jnp.arange(horizon_plus_one)
#
#         def _single_step(z, pi_k, q_k):
#             action, info = full_model.pi(z, key=pi_k)
#             q_val = full_model.q_value(
#                 z, action, two_hot_cfg,
#                 return_type="avg", key=q_k,
#                 inference=False     # dropout ON
#             )
#             return action, info, q_val
#
#         actions_all, infos, q_vals = jax.vmap(_single_step)(zs, pi_keys, q_keys)
#
#         qs_flat = jax.lax.stop_gradient(q_vals[0]).flatten()
#         p05 = jnp.percentile(qs_flat, 5)
#         p95 = jnp.percentile(qs_flat, 95)
#         new_scale_val = jnp.maximum(p95 - p05, 1.0)
#         inner_scale = (1.0 - scale_tau) * scale_value + scale_tau * new_scale_val
#
#         q_normalized = q_vals / jnp.maximum(inner_scale, 1.0)
#
#         step_losses = -(entropy_coef * infos["scaled_entropy"] + q_normalized).mean(axis=(-1, -2))
#         pi_loss = jnp.sum(step_losses * rho_weights) / horizon_plus_one
#
#         return pi_loss, (
#             pi_loss,
#             infos["entropy"].mean(),
#             infos["scaled_entropy"].mean(),
#             inner_scale,
#             q_vals[0].mean(),
#             q_vals[0].std(),
#             jnp.percentile(q_vals[0].flatten(), 5),
#             jnp.percentile(q_vals[0].flatten(), 95),
#         )
#
#     (_, (pi_loss, pi_entropy, pi_scaled_entropy, updated_scale_value, q_mean, q_std, q_p05, q_p95)), pi_grads = \
#         jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
#
#     pi_grad_norm = optax.global_norm(pi_grads)
#     pi_updates, new_pi_opt_state = pi_optimizer.update(pi_grads, pi_opt_state, pi_params)
#     new_pi_params = optax.apply_updates(pi_params, pi_updates)
#
#     new_policy = eqx.combine(new_pi_params, pi_static)
#     final_model = eqx.tree_at(lambda m: m.policy, new_model, new_policy)
#
#     # ========== 4. Polyak Target Q ==========
#     q_params, _ = eqx.partition(final_model.q_ensemble, eqx.is_inexact_array)
#     new_target_q_params = jax.tree.map(
#         lambda p, tp: tau * p + (1 - tau) * tp,
#         q_params, target_q_params,
#     )
#
#     info = UnifiedUpdateInfo(
#         consistency_loss=wm_info.consistency_loss,
#         reward_loss=wm_info.reward_loss,
#         value_loss=wm_info.value_loss,
#         total_loss=wm_info.total_loss,
#         wm_grad_norm=wm_grad_norm,
#         pi_loss=pi_loss,
#         pi_entropy=pi_entropy,
#         pi_scaled_entropy=pi_scaled_entropy,
#         pi_grad_norm=pi_grad_norm,
#         updated_scale_value=updated_scale_value,
#         q_mean=q_mean,
#         q_std=q_std,
#         q_p05=q_p05,
#         q_p95=q_p95,
#     )
#
#     return final_model, new_wm_opt_state, new_pi_opt_state, new_target_q_params, info
#
#
# # ==================== Planning (MPPI) ====================
#
#
# def plan_mppi_inner(
#     model: TDMPC2WorldModel,
#     z: jax.Array,
#     prev_mean: jax.Array,
#     two_hot_cfg: TwoHotConfig,
#     discount: float,
#     horizon: int,
#     num_samples: int,
#     num_pi_trajs: int,
#     num_elites: int,
#     num_iterations: int,
#     temperature: float,
#     min_std: float,
#     max_std: float,
#     t0: jax.Array,
#     eval_mode: jax.Array,
#     key: jax.Array,
# ) -> tuple[jax.Array, jax.Array]:
#     action_dim = model.action_dim
#
#     key, pi_key, sample_key, select_key = jax.random.split(key, 4)
#
#     # Sample trajectory candidates from learned policy
#     pi_actions = _sample_policy_trajectories(
#         model, z, horizon, num_pi_trajs, pi_key,
#     )
#
#     # Replicate latent state for all sample trajectories
#     z_expanded = jnp.broadcast_to(z, (num_samples, z.shape[-1]))
#
#     # Initialize mean: warm-start from previous step or zero for new episode
#     mean = jnp.where(
#         t0,
#         jnp.zeros((horizon, action_dim)),
#         jnp.concatenate(
#             [prev_mean[1:], jnp.zeros((1, action_dim))],
#             axis=0,
#         ),
#     )
#
#     std = jnp.full((horizon, action_dim), max_std)
#
#     init_score = jnp.zeros(num_elites)
#     init_elite_actions = jnp.zeros((horizon, num_elites, action_dim))
#
#     def mppi_step(carry, _):
#         mean, std, score, elite_actions, key = carry
#
#         key, sample_key, value_key = jax.random.split(key, 3)
#
#         noise = jax.random.normal(
#             sample_key, (horizon, num_samples - num_pi_trajs, action_dim)
#         )
#
#         sampled_actions = jnp.clip(
#             mean[:, None, :] + std[:, None, :] * noise, -1.0, 1.0
#         )
#
#         actions = jnp.concatenate([pi_actions, sampled_actions], axis=1)
#
#         values = _estimate_trajectory_value(
#             model, z_expanded, actions, two_hot_cfg, discount, horizon, value_key,
#         )
#
#         values_flat = jnp.nan_to_num(values.squeeze(-1), nan=0.0)
#
#         elite_idx = jnp.argsort(values_flat)[-num_elites:]
#         elite_values = values_flat[elite_idx]
#         elite_actions = actions[:, elite_idx, :]
#
#         max_val = elite_values.max()
#         score = jnp.exp(temperature * (elite_values - max_val))
#         score = score / (score.sum() + 1e-9)
#
#         new_mean = (score[None, :, None] * elite_actions).sum(axis=1)
#
#         new_std = jnp.sqrt(
#             (score[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
#             / (score.sum() + 1e-9)
#         )
#         new_std = jnp.clip(new_std, min_std, max_std)
#
#         return (new_mean, new_std, score, elite_actions, key), None
#
#     (final_mean, final_std, final_score, final_elite_actions, _), _ = jax.lax.scan(
#         mppi_step,
#         (mean, std, init_score, init_elite_actions, sample_key),
#         None,
#         length=num_iterations,
#     )
#
#     # Select elite trajectory via Gumbel-Softmax (matches original TD-MPC2)
#     rand_idx = gumbel_softmax_sample(final_score, select_key)
#     action = final_elite_actions[0, rand_idx]
#
#     # Add exploration noise if not in eval mode
#     noise = jax.random.normal(select_key, (action_dim,)) * final_std[0]
#     action = jnp.where(eval_mode, action, action + noise)
#     action = jnp.clip(action, -1.0, 1.0)
#
#     return action, final_mean
#
#
# @eqx.filter_jit
# def plan_mppi_batched(
#     model: TDMPC2WorldModel,
#     obs: jax.Array,
#     prev_mean: jax.Array,
#     two_hot_cfg: TwoHotConfig,
#     discount: float,
#     horizon: int,
#     num_samples: int,
#     num_pi_trajs: int,
#     num_elites: int,
#     num_iterations: int,
#     temperature: float,
#     min_std: float,
#     max_std: float,
#     t0_mask: jax.Array,
#     eval_mode: jax.Array,
#     keys: jax.Array,
# ) -> tuple[jax.Array, jax.Array]:
#     """Batched MPPI over envs via vmap."""
#     z = jax.vmap(model.encode)(obs)
#
#     def single_env(z_i, prev_mean_i, t0_i, key_i):
#         return plan_mppi_inner(
#             model=model,
#             z=z_i[None],
#             prev_mean=prev_mean_i,
#             two_hot_cfg=two_hot_cfg,
#             discount=discount,
#             horizon=horizon,
#             num_samples=num_samples,
#             num_pi_trajs=num_pi_trajs,
#             num_elites=num_elites,
#             num_iterations=num_iterations,
#             temperature=temperature,
#             min_std=min_std,
#             max_std=max_std,
#             t0=t0_i,
#             eval_mode=eval_mode,
#             key=key_i,
#         )
#
#     return jax.vmap(single_env)(z, prev_mean, t0_mask, keys)
#
#
# def _sample_policy_trajectories(
#     model: TDMPC2WorldModel,
#     z: jax.Array,
#     horizon: int,
#     num_pi_trajs: int,
#     key: jax.Array,
# ) -> jax.Array:
#     """Sample trajectories from policy prior."""
#     z_rep = jnp.broadcast_to(z, (num_pi_trajs, z.shape[-1]))
#     pi_actions = []
#     keys = jax.random.split(key, horizon)
#
#     for t in range(horizon):
#         a, _ = model.pi(z_rep, key=keys[t])
#         pi_actions.append(a)
#         if t < horizon - 1:
#             z_rep = model.next_latent(z_rep, a)
#
#     return jnp.stack(pi_actions, axis=0)
#
#
# def _estimate_trajectory_value(
#     model: TDMPC2WorldModel,
#     z: jax.Array,
#     actions: jax.Array,
#     two_hot_cfg: TwoHotConfig,
#     discount: float,
#     horizon: int,
#     key: jax.Array,
# ) -> jax.Array:
#     """Estimate value of trajectories."""
#     key, final_key = jax.random.split(key)
#
#     G = jnp.zeros((z.shape[0], 1))
#     discount_acc = 1.0
#
#     for t in range(horizon):
#         reward_logits = model.predict_reward(z, actions[t])
#         reward = two_hot_inv(reward_logits, two_hot_cfg)
#         G = G + discount_acc * reward
#         z = model.next_latent(z, actions[t])
#         discount_acc = discount_acc * discount
#
#     # Terminal value
#     final_action, _ = model.pi(z, key=final_key)
#     final_q = model.q_value(z, final_action, two_hot_cfg, return_type="avg", key=final_key)
#     G = G + discount_acc * final_q
#
#     return G


from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rlworld.rl.algorithms.tdmpc2.losses import (
    compute_world_model_loss,
)
from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
    two_hot_inv,
    gumbel_softmax_sample,
)
from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel, QEnsemble
from rlworld.rl.storages.sequence_replay_buffer import SequenceBatch


class UnifiedUpdateInfo(NamedTuple):
    """All metrics from unified update."""
    # World model
    consistency_loss: jax.Array
    reward_loss: jax.Array
    value_loss: jax.Array
    total_loss: jax.Array
    wm_grad_norm: jax.Array
    # Policy
    pi_loss: jax.Array
    pi_entropy: jax.Array
    pi_scaled_entropy: jax.Array
    pi_grad_norm: jax.Array
    updated_scale_value: jax.Array
    q_mean: jax.Array
    q_std: jax.Array
    q_p05: jax.Array
    q_p95: jax.Array


@eqx.filter_jit
def unified_update(
    model: TDMPC2WorldModel,
    wm_opt_state: optax.OptState,
    wm_optimizer: optax.GradientTransformation,
    pi_opt_state: optax.OptState,
    pi_optimizer: optax.GradientTransformation,
    target_q_params: Any,
    tau: float,
    batch: SequenceBatch,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    rho: float,
    entropy_coef: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    scale_value: jax.Array,
    scale_tau: float,
    key: jax.Array,
) -> tuple[
    TDMPC2WorldModel,
    optax.OptState,
    optax.OptState,
    Any,
    UnifiedUpdateInfo,
]:
    """
    Full TD-MPC2 update in a single JIT boundary.

    Steps:
    1. World model update (encoder, dynamics, reward, Q-ensemble)
    2. Latent rollout (for policy update, stop_gradient)
    3. Policy update with inline scale computation (matches author)
    4. Polyak target Q update

    Returns:
        (new_model, new_wm_opt, new_pi_opt, new_target_q_params, info)
    """
    key, wm_key, pi_key = jax.random.split(key, 3)

    # Reconstruct target Q inside JIT
    q_static = eqx.partition(model.q_ensemble, eqx.is_inexact_array)[1]
    target_q_ensemble = eqx.combine(target_q_params, q_static)

    # ========== 1. World Model Update ==========
    @eqx.filter_value_and_grad(has_aux=True)
    def wm_loss_fn(m: TDMPC2WorldModel):
        m_with_stopped_pi = eqx.tree_at(
            lambda x: x.policy,
            m,
            jax.lax.stop_gradient(m.policy),
        )
        _loss, _info = compute_world_model_loss(
            model=m_with_stopped_pi,
            target_q_ensemble=target_q_ensemble,
            batch=batch,
            two_hot_cfg=two_hot_cfg,
            discount=discount,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            key=wm_key,
        )
        return _loss, _info

    (wm_loss, wm_info), wm_grads = wm_loss_fn(model)
    wm_grad_norm = optax.global_norm(wm_grads)
    wm_updates, new_wm_opt_state = wm_optimizer.update(wm_grads, wm_opt_state, model)
    new_model = eqx.apply_updates(model, wm_updates)

    # ========== 2. Latent Rollout ==========
    obs = batch.observations
    actions = batch.actions
    horizon = actions.shape[0]

    z = new_model.encode(obs[0])
    zs = [z]
    for t in range(horizon):
        z = new_model.next_latent(z, actions[t])
        zs.append(z)
    zs = jax.lax.stop_gradient(jnp.stack(zs, axis=0))

    # ========== 3. Policy Update ==========
    pi_params, pi_static = eqx.partition(new_model.policy, eqx.is_inexact_array)

    def pi_loss_fn(p_params):
        new_policy = eqx.combine(p_params, pi_static)
        full_model = eqx.tree_at(lambda m: m.policy, new_model, new_policy)

        horizon_plus_one = zs.shape[0]
        pi_keys = jax.random.split(pi_key, horizon_plus_one)
        q_keys = jax.random.split(jax.random.fold_in(pi_key, 999), horizon_plus_one)
        rho_weights = rho ** jnp.arange(horizon_plus_one)

        def _single_step(z, pi_k, q_k):
            action, info = full_model.pi(z, key=pi_k)
            q_val = full_model.q_value(
                z, action, two_hot_cfg,
                return_type="avg", key=q_k,
                inference=False  # dropout ON
            )
            return action, info, q_val

        actions_all, infos, q_vals = jax.vmap(_single_step)(zs, pi_keys, q_keys)

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
        )

    (_, (pi_loss, pi_entropy, pi_scaled_entropy, updated_scale_value, q_mean, q_std, q_p05, q_p95)), pi_grads = \
        jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)

    pi_grad_norm = optax.global_norm(pi_grads)
    pi_updates, new_pi_opt_state = pi_optimizer.update(pi_grads, pi_opt_state, pi_params)
    new_pi_params = optax.apply_updates(pi_params, pi_updates)

    new_policy = eqx.combine(new_pi_params, pi_static)
    final_model = eqx.tree_at(lambda m: m.policy, new_model, new_policy)

    # ========== 4. Polyak Target Q ==========
    q_params, _ = eqx.partition(final_model.q_ensemble, eqx.is_inexact_array)
    new_target_q_params = jax.tree.map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        q_params, target_q_params,
    )

    info = UnifiedUpdateInfo(
        consistency_loss=wm_info.consistency_loss,
        reward_loss=wm_info.reward_loss,
        value_loss=wm_info.value_loss,
        total_loss=wm_info.total_loss,
        wm_grad_norm=wm_grad_norm,
        pi_loss=pi_loss,
        pi_entropy=pi_entropy,
        pi_scaled_entropy=pi_scaled_entropy,
        pi_grad_norm=pi_grad_norm,
        updated_scale_value=updated_scale_value,
        q_mean=q_mean,
        q_std=q_std,
        q_p05=q_p05,
        q_p95=q_p95,
    )

    return final_model, new_wm_opt_state, new_pi_opt_state, new_target_q_params, info


# ==================== Planning (MPPI) ====================


def plan_mppi_inner(
    model: TDMPC2WorldModel,
    z: jax.Array,
    prev_mean: jax.Array,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    horizon: int,
    num_samples: int,
    num_pi_trajs: int,
    num_elites: int,
    num_iterations: int,
    temperature: float,
    min_std: float,
    max_std: float,
    t0: jax.Array,
    eval_mode: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    action_dim = model.action_dim

    # Action bounds from model (static fields, no JIT trace impact)
    # squash_action=True  -> (-1.0, ..., -1.0) and (1.0, ..., 1.0) (original behavior)
    # squash_action=False -> environment action bounds
    action_low = jnp.array(model.action_low_tuple)
    action_high = jnp.array(model.action_high_tuple)

    key, pi_key, sample_key, select_key = jax.random.split(key, 4)

    # Sample trajectory candidates from learned policy
    pi_actions = _sample_policy_trajectories(
        model, z, horizon, num_pi_trajs, pi_key,
    )

    # Replicate latent state for all sample trajectories
    z_expanded = jnp.broadcast_to(z, (num_samples, z.shape[-1]))

    # Initialize mean: warm-start from previous step or zero for new episode
    # NOTE: When squash_action=False, zero-init means center of action space
    # only if action space is symmetric. This is acceptable as MPPI will
    # quickly adapt via the CEM iterations.
    mean = jnp.where(
        t0,
        jnp.zeros((horizon, action_dim)),
        jnp.concatenate(
            [prev_mean[1:], jnp.zeros((1, action_dim))],
            axis=0,
        ),
    )

    std = jnp.full((horizon, action_dim), max_std)

    init_score = jnp.zeros(num_elites)
    init_elite_actions = jnp.zeros((horizon, num_elites, action_dim))

    def mppi_step(carry, _):
        mean, std, score, elite_actions, key = carry

        key, sample_key, value_key = jax.random.split(key, 3)

        noise = jax.random.normal(
            sample_key, (horizon, num_samples - num_pi_trajs, action_dim)
        )

        # Clip sampled actions to action bounds
        # Original: jnp.clip(..., -1.0, 1.0)
        sampled_actions = jnp.clip(
            mean[:, None, :] + std[:, None, :] * noise,
            action_low[None, None, :],  # broadcast [1, 1, action_dim]
            action_high[None, None, :],  # broadcast [1, 1, action_dim]
        )

        actions = jnp.concatenate([pi_actions, sampled_actions], axis=1)

        values = _estimate_trajectory_value(
            model, z_expanded, actions, two_hot_cfg, discount, horizon, value_key,
        )

        values_flat = jnp.nan_to_num(values.squeeze(-1), nan=0.0)

        elite_idx = jnp.argsort(values_flat)[-num_elites:]
        elite_values = values_flat[elite_idx]
        elite_actions = actions[:, elite_idx, :]

        max_val = elite_values.max()
        score = jnp.exp(temperature * (elite_values - max_val))
        score = score / (score.sum() + 1e-9)

        new_mean = (score[None, :, None] * elite_actions).sum(axis=1)

        new_std = jnp.sqrt(
            (score[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
            / (score.sum() + 1e-9)
        )
        new_std = jnp.clip(new_std, min_std, max_std)

        return (new_mean, new_std, score, elite_actions, key), None

    (final_mean, final_std, final_score, final_elite_actions, _), _ = jax.lax.scan(
        mppi_step,
        (mean, std, init_score, init_elite_actions, sample_key),
        None,
        length=num_iterations,
    )

    # Select elite trajectory via Gumbel-Softmax (matches original TD-MPC2)
    rand_idx = gumbel_softmax_sample(final_score, select_key)
    action = final_elite_actions[0, rand_idx]

    # Add exploration noise if not in eval mode
    noise = jax.random.normal(select_key, (action_dim,)) * final_std[0]
    action = jnp.where(eval_mode, action, action + noise)
    # Clip final action to action bounds
    # Original: jnp.clip(action, -1.0, 1.0)
    action = jnp.clip(action, action_low, action_high)

    return action, final_mean


@eqx.filter_jit
def plan_mppi_batched(
    model: TDMPC2WorldModel,
    obs: jax.Array,
    prev_mean: jax.Array,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    horizon: int,
    num_samples: int,
    num_pi_trajs: int,
    num_elites: int,
    num_iterations: int,
    temperature: float,
    min_std: float,
    max_std: float,
    t0_mask: jax.Array,
    eval_mode: jax.Array,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Batched MPPI over envs via vmap."""
    z = jax.vmap(model.encode)(obs)

    def single_env(z_i, prev_mean_i, t0_i, key_i):
        return plan_mppi_inner(
            model=model,
            z=z_i[None],
            prev_mean=prev_mean_i,
            two_hot_cfg=two_hot_cfg,
            discount=discount,
            horizon=horizon,
            num_samples=num_samples,
            num_pi_trajs=num_pi_trajs,
            num_elites=num_elites,
            num_iterations=num_iterations,
            temperature=temperature,
            min_std=min_std,
            max_std=max_std,
            t0=t0_i,
            eval_mode=eval_mode,
            key=key_i,
        )

    return jax.vmap(single_env)(z, prev_mean, t0_mask, keys)


def _sample_policy_trajectories(
    model: TDMPC2WorldModel,
    z: jax.Array,
    horizon: int,
    num_pi_trajs: int,
    key: jax.Array,
) -> jax.Array:
    """Sample trajectories from policy prior.

    NOTE: No clipping here. When squash_action=True, pi() outputs [-1, 1].
    When squash_action=False, pi() outputs raw Gaussian actions which are
    unbounded, but MPPI's CEM iterations will naturally select feasible
    trajectories via the value-based ranking.
    """
    z_rep = jnp.broadcast_to(z, (num_pi_trajs, z.shape[-1]))
    pi_actions = []
    keys = jax.random.split(key, horizon)

    for t in range(horizon):
        a, _ = model.pi(z_rep, key=keys[t])
        pi_actions.append(a)
        if t < horizon - 1:
            z_rep = model.next_latent(z_rep, a)

    return jnp.stack(pi_actions, axis=0)


def _estimate_trajectory_value(
    model: TDMPC2WorldModel,
    z: jax.Array,
    actions: jax.Array,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    horizon: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate value of trajectories."""
    key, final_key = jax.random.split(key)

    G = jnp.zeros((z.shape[0], 1))
    discount_acc = 1.0

    for t in range(horizon):
        reward_logits = model.predict_reward(z, actions[t])
        reward = two_hot_inv(reward_logits, two_hot_cfg)
        G = G + discount_acc * reward
        z = model.next_latent(z, actions[t])
        discount_acc = discount_acc * discount

    # Terminal value
    final_action, _ = model.pi(z, key=final_key)
    final_q = model.q_value(z, final_action, two_hot_cfg, return_type="avg", key=final_key)
    G = G + discount_acc * final_q

    return G