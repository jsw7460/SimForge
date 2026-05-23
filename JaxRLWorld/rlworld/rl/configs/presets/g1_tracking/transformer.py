"""Unitree G1 motion tracking with the SpaceTimeTransformer actor/critic.

Subclass of :class:`G1TrackingConfig` that swaps the MLP policy for a
factorized space-time transformer. All task-level settings (motion
files, body list, rewards, terminations, action mode) are inherited;
this file only adds transformer hyperparameters and the policy
construction override.

The base class keeps ``future_offsets`` empty (no future-motion preview,
since the MLP baseline has no architectural use for it). This subclass
overrides to a non-empty tuple so the transformer's time axis carries
real reference content.

Usage::

    from rlworld.rl.configs.presets.g1_tracking.transformer import (
        G1TrackingTransformerConfig,
    )
    cfgs = G1TrackingTransformerConfig(sim_type="newton").build()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlworld.rl.configs.common_config_classes import (
    Activation,
    DistributionType,
    MLPCriticCfg,
    NNConfig,
    OrthoInit,
    PPOPolicyConfig,
    SpaceTimeTransformerActorCfg,
    SpaceTimeTransformerCriticCfg,
    StdType,
)

from .base import G1TrackingConfig

__all__ = ["G1TrackingTransformerConfig"]


@dataclass
class G1TrackingTransformerConfig(G1TrackingConfig):
    """G1 tracking with factorized space x time transformer policy."""

    num_envs: int = 512
    # ── Future motion reference window ────────────────────────────────
    # Sparse offsets in motion frames exposed to the transformer's time
    # axis (and to ``MotionCommandCfg.future_offsets`` so the obs term
    # has data). 4-frame sparse preview spanning ~0.16s at 50Hz.
    future_offsets: tuple[int, ...] = (1, 2, 4, 8)

    # ── Transformer hyperparameters ───────────────────────────────────
    transformer_embed_dim: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 512

    # NPMP-style information bottleneck. Encoder output is pooled to a
    # ``bottleneck_dim``-wide z and broadcast back to every per-body
    # decoder head, forcing the policy to abstract motion context into a
    # small latent rather than passing per-body features through.
    transformer_bottleneck_dim: int = 8

    # Per-joint decoder MLP hidden width. The decoder keeps one head per
    # actuated joint (so the morphological prior is preserved); set this
    # explicitly to keep heads compact instead of the
    # ``2 * (embed_dim + bottleneck_dim)`` default.
    transformer_decoder_hidden_dim: int = 32

    # ── Body-axis structural priors (SWAT-style) ──────────────────────
    # ``pe_type``: "learned" = single learnable (B, D) table (no
    # structural prior). "traversal" = SWAT-style; concatenates
    # pre/in/post-order DFS lookups so bodies nearby in the kinematic
    # tree share PE structure.
    pe_type: Literal["learned", "traversal"] = "learned"

    # When True, builds a GraphRelationalEmbedding that adds a learnable
    # per-head (H, B, B) bias to spatial attention scores derived from
    # graph features (Laplacian / SPD / PPR). Soft generalization of the
    # binary kinematic mask; combinable with ``use_kinematic_mask``.
    use_relational_bias: bool = False
    re_use_laplacian: bool = True
    re_use_spd: bool = True
    re_use_ppr: bool = True
    re_ppr_alpha: float = 0.15

    # Encoder attention mode. "factorized" = TimeSformer-style spatial
    # then temporal attention per layer. "joint" = single attention over
    # the flattened (T*B,) sequence per layer (better GPU fusion for
    # small T*B, and strictly more expressive).
    attention_mode: Literal["factorized", "joint"] = "joint"

    # Asymmetric actor-critic: the actor is the SpaceTimeTransformer (which
    # carries the research story), while the critic is a plain MLP over the
    # critic obs vector. Cheap and standard — value estimation doesn't need
    # the per-body / future-window structural prior, and dropping the
    # transformer on the critic side halves the forward/backward cost.
    # Set False to keep the transformer critic (e.g. for an ablation).
    use_mlp_critic: bool = True
    mlp_critic_hidden_dims: tuple[int, ...] = (512, 256, 128)

    # ── Build override ────────────────────────────────────────────────
    def _build_nn_config(self) -> NNConfig:
        if not self.future_offsets:
            raise ValueError(
                "G1TrackingTransformerConfig requires non-empty future_offsets. "
                "Set future_offsets to a tuple of motion-frame offsets, e.g. "
                "(1, 2, 4, 8), or use G1TrackingConfig (MLP baseline) instead."
            )
        actor_cfg = SpaceTimeTransformerActorCfg(
            tracked_body_names=self.body_names,
            future_offsets=self.future_offsets,
            ref_feature_dim=self.ref_feature_dim,
            embed_dim=self.transformer_embed_dim,
            num_heads=self.transformer_num_heads,
            num_layers=self.transformer_num_layers,
            dim_feedforward=self.transformer_dim_feedforward,
            bottleneck_dim=self.transformer_bottleneck_dim,
            decoder_hidden_dim=self.transformer_decoder_hidden_dim,
            use_kinematic_mask=False,
            pe_type=self.pe_type,
            use_relational_bias=self.use_relational_bias,
            re_use_laplacian=self.re_use_laplacian,
            re_use_spd=self.re_use_spd,
            re_use_ppr=self.re_use_ppr,
            re_ppr_alpha=self.re_ppr_alpha,
            attention_mode=self.attention_mode,
        )
        if self.use_mlp_critic:
            critic_cfg = MLPCriticCfg(
                activation=Activation.ELU,
                init=OrthoInit(output_gain=1.0),
                hidden_dims=list(self.mlp_critic_hidden_dims),
            )
        else:
            critic_cfg = SpaceTimeTransformerCriticCfg(
                tracked_body_names=self.body_names,
                future_offsets=self.future_offsets,
                ref_feature_dim=self.ref_feature_dim,
                embed_dim=self.transformer_embed_dim,
                num_heads=self.transformer_num_heads,
                num_layers=self.transformer_num_layers,
                dim_feedforward=self.transformer_dim_feedforward,
                use_kinematic_mask=False,
                pe_type=self.pe_type,
                use_relational_bias=self.use_relational_bias,
                re_use_laplacian=self.re_use_laplacian,
                re_use_spd=self.re_use_spd,
                re_use_ppr=self.re_use_ppr,
                re_ppr_alpha=self.re_ppr_alpha,
                attention_mode=self.attention_mode,
            )
        return NNConfig(
            policy=PPOPolicyConfig(
                actor=actor_cfg,
                critic=critic_cfg,
                init_noise_std=1.0,
                distribution_type=DistributionType.GAUSSIAN,
                std_type=StdType.STATE_INDEPENDENT,
            ),
        )
