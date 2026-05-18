"""Cfg-type-keyed actor/critic registry + typed builders.

Migration note (from the old string-keyed ``ACTOR_REGISTRY`` /
``get_actor_class(name)`` API): the registry is now keyed by the
*config dataclass type* (``MLPActorCfg`` → ``MLPActor`` class),
because the cfg type itself unambiguously identifies the network
architecture. ``build_actor`` / ``build_critic`` are the single
entry points the runner / ActorCritic code should call — they
translate a typed cfg into the actor/critic-class constructor
arguments without any string lookup.

External packages can extend the registry by importing
``ACTOR_CLASS_BY_CFG`` / ``CRITIC_CLASS_BY_CFG`` and assigning a
new ``(cfg_type, network_class)`` pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from rlworld.rl.configs.common_config_classes import (
    ActorCfg,
    CriticCfg,
    DefaultInit,
    MLPActorCfg,
    MLPCriticCfg,
    OrthoInit,
    SpaceTimeTransformerActorCfg,
    SpaceTimeTransformerCriticCfg,
)
from rlworld.rl.modules.architectures.base import BaseActor, BaseCritic
from rlworld.rl.modules.architectures.mlp.actor import MLPActor, MLPCritic
from rlworld.rl.modules.architectures.space_time_transformer.actor import (
    SpaceTimeTransformerActor,
)
from rlworld.rl.modules.architectures.space_time_transformer.critic import (
    SpaceTimeTransformerCritic,
)

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


# ── Cfg-type → network class registries ─────────────────────────────


ACTOR_CLASS_BY_CFG: dict[type, type] = {
    MLPActorCfg: MLPActor,
    SpaceTimeTransformerActorCfg: SpaceTimeTransformerActor,
}


CRITIC_CLASS_BY_CFG: dict[type, type] = {
    MLPCriticCfg: MLPCritic,
    SpaceTimeTransformerCriticCfg: SpaceTimeTransformerCritic,
}


# ── Build helpers ───────────────────────────────────────────────────


def _ortho_init_args(init) -> dict:
    """Translate the typed InitScheme union into the MLP ``ortho_init`` /
    ``output_gain`` kwargs MLPActor/MLPCritic still expect.

    OrthoInit(gain) → ortho_init=True, output_gain=gain.
    DefaultInit()   → ortho_init=False, output_gain=1.0 (ignored).
    """
    if isinstance(init, OrthoInit):
        return {"ortho_init": True, "output_gain": init.output_gain}
    if isinstance(init, DefaultInit):
        return {"ortho_init": False, "output_gain": 1.0}
    raise TypeError(f"Unknown InitScheme: {type(init).__name__}")


def build_actor(
    actor_cfg: ActorCfg,
    *,
    num_obs: int,
    num_actions: int,
    key: jax.Array,
    kinematic_tree: KinematicTree | None = None,
    actuated_joint_names: list[str] | None = None,
) -> BaseActor:
    """Instantiate the right actor class for ``actor_cfg``."""
    ActorClass = ACTOR_CLASS_BY_CFG[type(actor_cfg)]

    if isinstance(actor_cfg, MLPActorCfg):
        init_kwargs = _ortho_init_args(actor_cfg.init)
        return ActorClass(
            num_obs=num_obs,
            num_actions=num_actions,
            hidden_dims=list(actor_cfg.hidden_dims),
            activation=actor_cfg.activation.value,
            use_layer_norm=actor_cfg.use_layer_norm,
            key=key,
            **init_kwargs,
        )

    if isinstance(actor_cfg, SpaceTimeTransformerActorCfg):
        if kinematic_tree is None:
            raise ValueError("SpaceTimeTransformerActorCfg requires kinematic_tree.")
        return ActorClass(
            kinematic_tree=kinematic_tree,
            num_obs=num_obs,
            num_actions=num_actions,
            tracked_body_names=actor_cfg.tracked_body_names,
            future_offsets=actor_cfg.future_offsets,
            actuated_joint_names=actor_cfg.actuated_joint_names
            if actor_cfg.actuated_joint_names is not None
            else actuated_joint_names,
            ref_feature_dim=actor_cfg.ref_feature_dim,
            embed_dim=actor_cfg.embed_dim,
            num_heads=actor_cfg.num_heads,
            num_layers=actor_cfg.num_layers,
            dim_feedforward=actor_cfg.dim_feedforward,
            dropout=actor_cfg.dropout,
            bottleneck_dim=actor_cfg.bottleneck_dim,
            tokenizer_hidden_dim=actor_cfg.tokenizer_hidden_dim,
            decoder_hidden_dim=actor_cfg.decoder_hidden_dim,
            decoder_activation=actor_cfg.decoder_activation.value,
            use_kinematic_mask=actor_cfg.use_kinematic_mask,
            pe_type=actor_cfg.pe_type,
            use_relational_bias=actor_cfg.use_relational_bias,
            re_use_laplacian=actor_cfg.re_use_laplacian,
            re_use_spd=actor_cfg.re_use_spd,
            re_use_ppr=actor_cfg.re_use_ppr,
            re_ppr_alpha=actor_cfg.re_ppr_alpha,
            attention_mode=actor_cfg.attention_mode,
            key=key,
        )

    raise TypeError(f"Unknown ActorCfg type: {type(actor_cfg).__name__}")


def build_critic(
    critic_cfg: CriticCfg,
    *,
    num_obs: int,
    key: jax.Array,
    kinematic_tree: KinematicTree | None = None,
) -> BaseCritic:
    """Instantiate the right critic class for ``critic_cfg``."""
    CriticClass = CRITIC_CLASS_BY_CFG[type(critic_cfg)]

    if isinstance(critic_cfg, MLPCriticCfg):
        # Critic-side OrthoInit always uses output_gain=1.0 inside
        # MLPCritic (it ignores per-cfg output_gain), but we still
        # honor DefaultInit by toggling ``ortho_init``.
        init_kwargs = {"ortho_init": isinstance(critic_cfg.init, OrthoInit)}
        return CriticClass(
            num_obs=num_obs,
            hidden_dims=list(critic_cfg.hidden_dims),
            activation=critic_cfg.activation.value,
            use_layer_norm=critic_cfg.use_layer_norm,
            key=key,
            **init_kwargs,
        )

    if isinstance(critic_cfg, SpaceTimeTransformerCriticCfg):
        if kinematic_tree is None:
            raise ValueError("SpaceTimeTransformerCriticCfg requires kinematic_tree.")
        return CriticClass(
            kinematic_tree=kinematic_tree,
            num_obs=num_obs,
            tracked_body_names=critic_cfg.tracked_body_names,
            future_offsets=critic_cfg.future_offsets,
            ref_feature_dim=critic_cfg.ref_feature_dim,
            embed_dim=critic_cfg.embed_dim,
            num_heads=critic_cfg.num_heads,
            num_layers=critic_cfg.num_layers,
            dim_feedforward=critic_cfg.dim_feedforward,
            dropout=critic_cfg.dropout,
            tokenizer_hidden_dim=critic_cfg.tokenizer_hidden_dim,
            use_kinematic_mask=critic_cfg.use_kinematic_mask,
            pe_type=critic_cfg.pe_type,
            use_relational_bias=critic_cfg.use_relational_bias,
            re_use_laplacian=critic_cfg.re_use_laplacian,
            re_use_spd=critic_cfg.re_use_spd,
            re_use_ppr=critic_cfg.re_use_ppr,
            re_ppr_alpha=critic_cfg.re_ppr_alpha,
            attention_mode=critic_cfg.attention_mode,
            key=key,
        )

    raise TypeError(f"Unknown CriticCfg type: {type(critic_cfg).__name__}")
