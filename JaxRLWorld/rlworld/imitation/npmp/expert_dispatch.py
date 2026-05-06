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

**motion_clip_id bridge.** Experts trained with single-clip
``T1TrackingConfig`` saw a ``motion_clip_id_onehot`` slot of width 1
(constant ``[1.0]``). The distillation env runs with all N motions, so
the same obs term widens to width N, shifting every actor first-layer
weight downstream and breaking ``eqx.tree_deserialise_leaves``. The
dispatcher detects the motion_clip_id slice in the env's actor obs
group and at runtime *bridges* it back to a single constant-1.0 column
before calling each expert. This is mathematically identical to the
expert's training input (the expert had baked the constant 1.0 into a
learned bias on that column), so dispatched mean actions match
training-time outputs bit-for-bit. The bridge is a no-op when the
motion_clip_id term is absent or already width 1 — i.e. once experts
are retrained without it, this file needs no further change.
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


_CLIP_ID_TERM_NAME = "motion_clip_id"


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
        actor_obs,
        key=jax.random.PRNGKey(0),
        deterministic=True,
    )
    return mean


# ── Checkpoint → PPOActorCritic loader ──────────────────────────────


def _load_expert_policy(
    checkpoint_path: str,
    env: World,
    key: jax.Array,
    *,
    actor_obs_dim_override: int | None = None,
    critic_obs_dim_override: int | None = None,
) -> PPOActorCritic:
    """Reconstruct one expert's :class:`PPOActorCritic` from its checkpoint.

    Mirrors :meth:`OnPolicyRunner._init_ppo_actor_critic` but skips the
    optimizer / storage / runner-state setup — distillation only needs
    the policy weights for inference.

    The two ``*_override`` kwargs let the caller specify the actor /
    critic obs dims that the expert was *trained* with, when the
    distillation env's obs differs in width (e.g. the
    ``motion_clip_id_onehot`` term widens with multi-motion). Built
    PPOActorCritic must match the saved shapes for
    ``eqx.tree_deserialise_leaves`` to succeed; the dispatcher's
    runtime bridge then reconciles the two layouts before forward.
    """
    metadata = load_checkpoint_metadata(checkpoint_path)
    cfgs = load_config_from_checkpoint(metadata)

    obs_dim = env.calculate_obs_dim()
    actor_obs_dim = actor_obs_dim_override if actor_obs_dim_override is not None else obs_dim["actor"]
    critic_obs_dim = critic_obs_dim_override if critic_obs_dim_override is not None else obs_dim["critic"]
    num_actions = env.num_actions

    policy_cfg = cfgs.nn.policy

    kinematic_tree = env.scene_manager.trees.get("robot", None) if hasattr(env, "scene_manager") else None
    actuated_joint_names = list(env.act_manager.actuated_joint_names) if hasattr(env, "act_manager") else None

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


# ── motion_clip_id bridge helpers ───────────────────────────────────


def _term_slice(
    env: World,
    group_name: str,
    term_name: str,
) -> tuple[int, int] | None:
    """Look up ``(start, end)`` for an obs term in a group, or ``None``.

    Reads from ``ObservationManager._group_term_indices``, which is
    populated by ``calculate_obs_dim`` / ``_build_term_indices``. The
    attribute is private but stable internal API.
    """
    om = env.obs_manager
    # Trigger the lazy index build if needed.
    om.calculate_obs_dim()
    group = getattr(om, "_group_term_indices", {}).get(group_name)
    if not group:
        return None
    return group.get(term_name)


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
        env: World,
        key: jax.Array,
    ):
        if len(checkpoint_paths) == 0:
            raise ValueError("MultiExpertDispatcher requires at least one expert checkpoint path.")

        env_obs_dim = env.calculate_obs_dim()
        env_actor_dim = env_obs_dim["actor"]
        env_critic_dim = env_obs_dim["critic"]

        # Detect the motion_clip_id slice in actor / critic groups so
        # we can map env obs (width N for multi-motion) to the
        # single-clip width 1 the experts were trained on.
        actor_clip_slice = _term_slice(env, "actor", _CLIP_ID_TERM_NAME)
        critic_clip_slice = _term_slice(env, "critic", _CLIP_ID_TERM_NAME)

        actor_clip_len = actor_clip_slice[1] - actor_clip_slice[0] if actor_clip_slice is not None else 0
        critic_clip_len = critic_clip_slice[1] - critic_clip_slice[0] if critic_clip_slice is not None else 0

        # Bridge only fires when the env's clip_id term is wider than
        # 1 (i.e. multi-motion). Width 0 (term absent) or width 1
        # (single motion) → identity passthrough at runtime.
        self._needs_actor_bridge = actor_clip_len > 1
        self._actor_clip_start = actor_clip_slice[0] if actor_clip_slice else 0
        self._actor_clip_end = actor_clip_slice[1] if actor_clip_slice else 0

        # Expert obs dims (the dims the saved weights were trained
        # with). When bridging, every clip_id segment collapses from
        # ``clip_len`` to 1.
        expert_actor_dim = env_actor_dim - actor_clip_len + 1 if self._needs_actor_bridge else env_actor_dim
        expert_critic_dim = env_critic_dim - critic_clip_len + 1 if critic_clip_len > 1 else env_critic_dim
        self._expert_actor_dim = expert_actor_dim

        # Load the experts.
        keys = jax.random.split(key, len(checkpoint_paths))
        self._experts: list[PPOActorCritic] = [
            _load_expert_policy(
                path,
                env,
                k,
                actor_obs_dim_override=expert_actor_dim,
                critic_obs_dim_override=expert_critic_dim,
            )
            for path, k in zip(checkpoint_paths, keys)
        ]

        self._action_dim: int = env.num_actions
        self._env_actor_dim: int = env_actor_dim

    @property
    def num_experts(self) -> int:
        return len(self._experts)

    @property
    def experts(self) -> tuple[PPOActorCritic, ...]:
        return tuple(self._experts)

    @property
    def expert_actor_obs_dim(self) -> int:
        """Actor obs dim each loaded expert was *trained* with.

        Equals the env actor obs dim when no bridging is needed, or
        ``env_actor_dim - clip_id_len + 1`` when the multi-motion
        clip_id slot is collapsed back to a single constant column.
        """
        return self._expert_actor_dim

    def _bridge_actor_obs(self, actor_obs: jax.Array) -> jax.Array:
        """Replace the multi-motion clip_id one-hot slot with a single
        constant ``1.0`` column so the obs matches the single-clip
        layout the experts were trained on.

        For an env tracking motion ``i``, the original clip_id segment
        is ``e_i`` (one-hot at position ``i``). Slot ``i`` of that
        segment is already 1.0; slots ``≠ i`` are 0. Replacing the
        whole segment with a constant 1.0 column is therefore both
        equivalent to slot ``i``'s value (since each expert's
        per-row output only matters when its index matches the env's
        ``motion_id``) *and* identical to what every expert saw at
        training time (single-clip → constant ``[1.0]``).
        """
        if not self._needs_actor_bridge:
            return actor_obs
        ones_col = jnp.ones(
            (actor_obs.shape[0], 1),
            dtype=actor_obs.dtype,
        )
        return jnp.concatenate(
            [
                actor_obs[:, : self._actor_clip_start],
                ones_col,
                actor_obs[:, self._actor_clip_end :],
            ],
            axis=-1,
        )

    def deterministic_mean(
        self,
        actor_obs: jax.Array,  # (num_envs, env_actor_dim)
        motion_ids: jax.Array,  # (num_envs,) int — value in [0, num_experts)
    ) -> jax.Array:
        """Return ``(num_envs, action_dim)`` expert means per env.

        The actor obs is bridged through :meth:`_bridge_actor_obs`
        before being handed to each expert (no-op when the env layout
        already matches the experts' training layout). For env ``e``
        with ``motion_ids[e] = i``, the returned row equals
        ``experts[i]``'s deterministic actor mean on the bridged obs.
        Other rows of ``experts[i]``'s output are masked away with
        :func:`jnp.where` so only the matching expert's prediction
        survives at each row.
        """
        if actor_obs.shape[1] != self._env_actor_dim:
            raise ValueError(
                f"actor_obs dim {actor_obs.shape[1]} does not match the "
                f"distillation env's expected actor dim {self._env_actor_dim}."
            )

        bridged = self._bridge_actor_obs(actor_obs)

        out = jnp.zeros((bridged.shape[0], self._action_dim))
        for i, expert in enumerate(self._experts):
            mean_i = _expert_mean(expert, bridged)
            mask_i = (motion_ids == i)[:, None]
            out = jnp.where(mask_i, mean_i, out)
        return out
