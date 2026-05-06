"""SpaceTimeTransformer critic.

Shares the tokenizer and encoder architecture with
:class:`SpaceTimeTransformerActor` (independently initialised — no
weight sharing) and mean-pools the encoded token grid to a scalar
value. Accepts the same kwargs as the actor so presets can configure
actor and critic in lockstep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.architectures.base import BaseCritic
from rlworld.rl.modules.architectures.space_time_transformer.actor import (
    _resolve_body_indices,
)
from rlworld.rl.modules.architectures.space_time_transformer.encoder import (
    SpaceTimeTransformerEncoder,
)
from rlworld.rl.modules.architectures.space_time_transformer.tokenizer import (
    SpaceTimeTokenizer,
)

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


__all__ = ["SpaceTimeTransformerCritic"]


class SpaceTimeTransformerCritic(BaseCritic):
    tokenizer: SpaceTimeTokenizer
    encoder: SpaceTimeTransformerEncoder
    value_head: eqx.nn.Linear

    num_obs: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        num_obs: int,
        tracked_body_names: Sequence[str],
        future_offsets: Sequence[int],
        ref_feature_dim: int = 9,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        tokenizer_hidden_dim: int | None = None,
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
                f"({future_window_dim}). Check observation group ordering."
            )

        key_tok, key_enc, key_head = jax.random.split(key, 3)

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

        self.value_head = eqx.nn.Linear(embed_dim, 1, key=key_head)

    def _forward_single(self, observation: jax.Array) -> jax.Array:
        tokens = self.tokenizer(observation)
        encoded = self.encoder(tokens, key=None)
        pooled = jnp.mean(encoded.reshape(-1, encoded.shape[-1]), axis=0)
        return self.value_head(pooled)

    def __call__(self, observation: jax.Array) -> tuple[jax.Array, dict]:
        """Forward with automatic batching.

        Args:
            observation: ``(num_obs,)`` or ``(batch, num_obs)``.

        Returns:
            value: ``(1,)`` or ``(batch, 1)``.
            aux: empty dict.
        """
        if observation.ndim == 1:
            return self._forward_single(observation), {}
        return jax.vmap(self._forward_single)(observation), {}
