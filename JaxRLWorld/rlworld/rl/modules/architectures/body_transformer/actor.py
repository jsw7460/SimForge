from typing import TYPE_CHECKING

import equinox as eqx
import jax

from rlworld.rl.modules.architectures.base import BaseActor
from rlworld.rl.modules.architectures.body_transformer.encoder import BodyTransformerEncoder
from rlworld.rl.modules.architectures.body_transformer.tokenizer import BodyTokenizer
from rlworld.rl.modules.architectures.morphology_utils import ParentLinkToJointActionDecoder

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["BodyTransformerActor"]


class BodyTransformerActor(BaseActor):
    """
    Body Transformer actor.
    Processes unbatched input. Use jax.vmap for batched input.
    """
    tokenizer: BodyTokenizer
    encoder: BodyTransformerEncoder
    decoder: ParentLinkToJointActionDecoder

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        num_obs: int,
        num_actions: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        tokenizer_hidden_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_activation: str = "elu",
        use_mixed_attention: bool = True,
        first_masked_layer: int = 1,
        *,
        key: jax.Array,
        **kwargs
    ):
        self.num_obs = num_obs
        self.num_actions = num_actions

        key_tok, key_enc, key_dec = jax.random.split(key, 3)

        self.tokenizer = BodyTokenizer(
            kinematic_tree=kinematic_tree,
            obs_dim=num_obs,
            embed_dim=embed_dim,
            hidden_dim=tokenizer_hidden_dim,
            key=key_tok,
        )

        self.encoder = BodyTransformerEncoder(
            kinematic_tree=kinematic_tree,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_mixed_attention=use_mixed_attention,
            first_masked_layer=first_masked_layer,
            key=key_enc,
        )

        if decoder_hidden_dim is None:
            decoder_hidden_dim = embed_dim * 2

        self.decoder = ParentLinkToJointActionDecoder(
            kinematic_tree=kinematic_tree,
            hidden_dim=embed_dim,
            activation=decoder_activation,
            action_hidden_dim=decoder_hidden_dim,
            key=key_dec,
        )

    def __call__(self, observation: jax.Array, *, key: jax.Array | None = None) -> tuple[jax.Array, dict]:
        """
        Args:
            observation: (num_obs,) - unbatched

        Returns:
            action: (num_actions,)
        """
        tokens = self.tokenizer(observation)  # (num_bodies, embed_dim)
        encoded = self.encoder(tokens, key=key)  # (num_bodies, embed_dim)
        action = self.decoder(encoded)  # (num_actions,)
        return action, {}
