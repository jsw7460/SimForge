"""Expert checkpoint loading + per-env dispatch for NPMP distillation.

Two responsibilities, one class:

1. **Load** N pretrained ``PPOActorCritic`` checkpoints into memory.
   Each checkpoint is the output of a single ``T1TrackingConfig``
   training run on one motion clip — the "expert" for that clip.

2. **Dispatch** at every distillation rollout step: given
   ``(actor_obs, motion_ids)`` per env, return ``(num_envs, A)``
   noiseless mean actions where row ``e`` is supervised by
   ``experts[motion_ids[e]]``.

The dispatch implementation runs every expert on the full batch and
masks the output by ``motion_ids``. With ``N=9`` experts and
``num_envs ≈ 4096`` this is wasteful by a factor of ~9 in actor
forward compute, but the actor MLPs are small relative to physics
stepping; the simple constant graph is a worthwhile tradeoff over
ragged per-expert sub-batches. A vmap-stacked variant (one batched
module across the expert axis) is the natural follow-up if profiling
reveals a bottleneck.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.rl.utils.checkpoint import (
    load_checkpoint_metadata,
    load_config_from_checkpoint,
)

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


__all__ = ["MultiExpertDispatcher"]


# ── Per-expert deterministic forward ────────────────────────────────


@eqx.filter_jit
def _expert_mean(
    model: PPOActorCritic,
    actor_obs: jax.Array,
) -> jax.Array:
    """Deterministic action mean of one expert's actor.

    Uses ``PPOActorCritic.act(deterministic=True)`` — applies the
    actor obs normalizer and any tanh squashing the policy distribution
    requires. The PRNG key passed in is irrelevant for the
    deterministic path, so a fixed dummy key is used so the JIT cache
    key stays stable across calls.
    """
    mean, _ = model.act(
        actor_obs, key=jax.random.PRNGKey(0), deterministic=True,
    )
    return mean


# ── Checkpoint → PPOActorCritic loader ──────────────────────────────


def _load_expert_policy(
    checkpoint_path: str,
    env: "World",
    key: jax.Array,
) -> PPOActorCritic:
    """Reconstruct one expert's :class:`PPOActorCritic` from its checkpoint.

    Mirrors :meth:`OnPolicyRunner._init_ppo_actor_critic` but skips the
    optimizer / storage / runner-state setup — distillation only needs
    the policy weights for inference.

    Loads ``config.yaml`` + ``train_state.yaml`` to recover the original
    training config, builds an empty ``PPOActorCritic`` with matching
    architecture, then deserialises ``model.eqx`` into it.
    """
    metadata = load_checkpoint_metadata(checkpoint_path)
    cfgs = load_config_from_checkpoint(metadata)

    obs_dim = env.calculate_obs_dim()
    actor_obs_dim = obs_dim["actor"]
    critic_obs_dim = obs_dim["critic"]
    num_actions = env.num_actions

    policy_cfg = cfgs.nn.policy

    kinematic_tree = (
        env.scene_manager.trees.get("robot", None)
        if hasattr(env, "scene_manager")
        else None
    )
    actuated_joint_names = (
        list(env.act_manager.actuated_joint_names)
        if hasattr(env, "act_manager")
        else None
    )

    actor_critic = PPOActorCritic(
        num_actor_obs=actor_obs_dim,
        num_critic_obs=critic_obs_dim,
        num_actions=num_actions,
        actor_class_name=policy_cfg.actor_class_name,
        critic_class_name=policy_cfg.critic_class_name,
        init_noise_std=policy_cfg.init_noise_std,
        std_type=policy_cfg.std_type,
        distribution_type=policy_cfg.distribution_type,
        kinematic_tree=kinematic_tree,
        actuated_joint_names=actuated_joint_names,
        key=key,
        actor_kwargs=policy_cfg.actor_kwargs,
        critic_kwargs=policy_cfg.critic_kwargs,
        obs_normalization=cfgs.algorithm.obs_normalization,
    )

    model_path = os.path.join(checkpoint_path, "model.eqx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Expert weights not found at {model_path!r}. "
            f"Expected an ``eqx.tree_serialise_leaves`` output from a "
            f"completed PPO training run."
        )

    actor_critic = eqx.tree_deserialise_leaves(model_path, actor_critic)
    return actor_critic


# ── Multi-expert dispatcher ─────────────────────────────────────────


class MultiExpertDispatcher:
    """Owns ``N`` loaded expert ``PPOActorCritic`` modules and routes
    each env's actor obs to the expert assigned by ``motion_ids[e]``.

    The dispatcher is constructed once at trainer startup (after the
    distillation env exists) and called every rollout step.
    """

    def __init__(
        self,
        checkpoint_paths: tuple[str, ...] | list[str],
        env: "World",
        key: jax.Array,
    ):
        if len(checkpoint_paths) == 0:
            raise ValueError(
                "MultiExpertDispatcher requires at least one expert "
                "checkpoint path."
            )
        keys = jax.random.split(key, len(checkpoint_paths))
        self._experts: list[PPOActorCritic] = [
            _load_expert_policy(path, env, k)
            for path, k in zip(checkpoint_paths, keys)
        ]
        self._action_dim: int = env.num_actions
        self._actor_obs_dim: int = env.calculate_obs_dim()["actor"]

    @property
    def num_experts(self) -> int:
        return len(self._experts)

    @property
    def experts(self) -> tuple[PPOActorCritic, ...]:
        return tuple(self._experts)

    def deterministic_mean(
        self,
        actor_obs: jax.Array,    # (num_envs, D_actor)
        motion_ids: jax.Array,   # (num_envs,) int — value in [0, num_experts)
    ) -> jax.Array:
        """Return ``(num_envs, action_dim)`` expert means per env.

        For env ``e`` with ``motion_ids[e] = i``, the returned row equals
        ``experts[i]``'s deterministic actor mean evaluated on
        ``actor_obs[e]``. All other expert outputs at row ``e`` are
        masked away with :func:`jnp.where` so the gradient does not
        flow through them — but since the experts are frozen and never
        wrapped in ``jax.grad`` here, the practical effect is just a
        9× actor-forward overhead on row-by-row pruning.
        """
        if actor_obs.shape[1] != self._actor_obs_dim:
            raise ValueError(
                f"actor_obs dim {actor_obs.shape[1]} does not match the "
                f"distillation env's expected actor dim {self._actor_obs_dim}."
            )

        out = jnp.zeros((actor_obs.shape[0], self._action_dim))
        for i, expert in enumerate(self._experts):
            mean_i = _expert_mean(expert, actor_obs)
            mask_i = (motion_ids == i)[:, None]
            out = jnp.where(mask_i, mean_i, out)
        return out
