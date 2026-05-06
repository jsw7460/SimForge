"""SpaceTimeTransformer actor.

Stacks :class:`SpaceTimeTokenizer` → :class:`SpaceTimeTransformerEncoder`
→ :class:`ParentLinkToJointActionDecoder`. The encoded token at
``(t=0, b)`` for each kinematic-tree body is treated as that body's
"current state informed by motion future" representation, and the
existing parent-link decoder maps per-body features to per-joint actions.

Works as a drop-in for :class:`MLPActor` on any preset; non-motion-
tracking presets pass ``num_future_frames=0`` so the tokenizer produces
a single time token and the temporal attention layers become no-ops,
reducing the architecture to a pure body-axis Body-Transformer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.architectures.base import BaseActor
from rlworld.rl.modules.architectures.morphology_utils import (
    ParentLinkToJointActionDecoder,
)
from rlworld.rl.modules.architectures.space_time_transformer.encoder import (
    SpaceTimeTransformerEncoder,
)
from rlworld.rl.modules.architectures.space_time_transformer.tokenizer import (
    SpaceTimeTokenizer,
)

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


__all__ = ["SpaceTimeTransformerActor"]


def _resolve_body_indices(
    kinematic_tree: KinematicTree,
    body_names: Sequence[str],
) -> list[int]:
    """Look up each name in the kinematic tree's link list, preserving order."""
    name_to_idx = {link["name"]: i for i, link in enumerate(kinematic_tree.links)}
    out: list[int] = []
    for n in body_names:
        if n not in name_to_idx:
            raise KeyError(f"Tracked body name {n!r} not found in kinematic tree. Available: {sorted(name_to_idx)}")
        out.append(name_to_idx[n])
    return out


class SpaceTimeTransformerActor(BaseActor):
    tokenizer: SpaceTimeTokenizer
    encoder: SpaceTimeTransformerEncoder
    bottleneck: eqx.nn.Linear
    decoder: ParentLinkToJointActionDecoder

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    bottleneck_dim: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        num_obs: int,
        num_actions: int,
        tracked_body_names: Sequence[str],
        future_offsets: Sequence[int],
        actuated_joint_names: Sequence[str] | None = None,
        ref_feature_dim: int = 9,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        bottleneck_dim: int = 32,
        tokenizer_hidden_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_activation: str = "elu",
        use_kinematic_mask: bool = True,
        pe_type: str = "learned",
        use_relational_bias: bool = False,
        re_use_laplacian: bool = True,
        re_use_spd: bool = True,
        re_use_ppr: bool = True,
        re_ppr_alpha: float = 0.15,
        attention_mode: str = "factorized",
        *,
        key: jax.Array,
        **kwargs,
    ):
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.bottleneck_dim = bottleneck_dim

        tracked_body_indices = _resolve_body_indices(
            kinematic_tree,
            tracked_body_names,
        )
        num_future_frames = len(future_offsets)
        future_window_dim = num_future_frames * len(tracked_body_names) * ref_feature_dim
        proprio_dim = num_obs - future_window_dim
        if proprio_dim < 0:
            raise ValueError(
                f"num_obs ({num_obs}) is smaller than computed future_window_dim "
                f"({future_window_dim}). Check that the observation group places "
                f"motion_future_reference_window at the end and that "
                f"future_offsets / tracked_body_names match the MotionCommand config."
            )

        key_tok, key_enc, key_bn, key_dec = jax.random.split(key, 4)

        self.tokenizer = SpaceTimeTokenizer(
            num_bodies_all=kinematic_tree.num_bodies,
            tracked_body_indices=tracked_body_indices,
            proprio_dim=proprio_dim,
            num_future_frames=num_future_frames,
            ref_feature_dim=ref_feature_dim,
            embed_dim=embed_dim,
            hidden_dim=tokenizer_hidden_dim,
            key=key_tok,
        )

        self.encoder = SpaceTimeTransformerEncoder(
            num_bodies_all=kinematic_tree.num_bodies,
            num_time_tokens=num_future_frames + 1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            kinematic_tree=kinematic_tree,
            use_kinematic_mask=use_kinematic_mask,
            pe_type=pe_type,
            use_relational_bias=use_relational_bias,
            re_use_laplacian=re_use_laplacian,
            re_use_spd=re_use_spd,
            re_use_ppr=re_use_ppr,
            re_ppr_alpha=re_ppr_alpha,
            attention_mode=attention_mode,
            key=key_enc,
        )

        # Information bottleneck: pool encoder output (across all tokens)
        # into a small ``z`` (NPMP-style global motion-intent latent).
        # Concatenated to every per-body t=0 feature so each body decoder
        # head sees both its own state and the abstracted clip context.
        self.bottleneck = eqx.nn.Linear(embed_dim, bottleneck_dim, key=key_bn)

        decoder_input_dim = embed_dim + bottleneck_dim
        if decoder_hidden_dim is None:
            decoder_hidden_dim = decoder_input_dim * 2

        self.decoder = ParentLinkToJointActionDecoder(
            kinematic_tree=kinematic_tree,
            hidden_dim=decoder_input_dim,
            activation=decoder_activation,
            action_hidden_dim=decoder_hidden_dim,
            output_gain=1.0,
            actuated_joint_names=actuated_joint_names,
            key=key_dec,
        )

    def _forward_single(
        self,
        observation: jax.Array,
        key: jax.Array | None,
    ) -> jax.Array:
        tokens = self.tokenizer(observation)
        encoded = self.encoder(tokens, key=key)  # (T+1, B_all, D)
        # Global bottleneck: pool every token to one vector, project to z.
        pooled = encoded.reshape(-1, encoded.shape[-1]).mean(axis=0)
        z = self.bottleneck(pooled)  # (D_z,)
        # Per-body t=0 features fused with broadcast z.
        per_body = encoded[0]  # (B_all, D)
        z_broadcast = jnp.broadcast_to(z, (per_body.shape[0], z.shape[0]))
        fused = jnp.concatenate([per_body, z_broadcast], axis=-1)
        return self.decoder(fused)

    def __call__(
        self,
        observation: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, dict]:
        """Forward with automatic batching.

        Args:
            observation: ``(num_obs,)`` or ``(batch, num_obs)``.
            key: PRNG key (split per-batch when ``observation.ndim == 2``).

        Returns:
            action: ``(num_actions,)`` or ``(batch, num_actions)`` mean action.
            aux: empty dict.
        """
        if observation.ndim == 1:
            return self._forward_single(observation, key), {}
        if key is None:
            keys = jnp.broadcast_to(
                jax.random.PRNGKey(0),
                (observation.shape[0], 2),
            )
        else:
            keys = jax.random.split(key, observation.shape[0])
        return jax.vmap(self._forward_single)(observation, keys), {}
