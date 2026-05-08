"""T1 motion tracking with the SpaceTimeTransformer actor/critic.

Subclass of :class:`T1TrackingConfig` that swaps the MLP policy for a
factorized space-time transformer. All task-level settings (motion
files, body list, rewards, terminations, action mode) are inherited;
this file only adds transformer hyperparameters and the policy
construction override.

The base class keeps ``future_offsets`` empty (no future-motion
preview, since the MLP baseline doesn't have an architectural use for
it). This subclass overrides to a non-empty tuple so the transformer's
time axis carries real reference content.

Usage::

    from rlworld.rl.configs.presets.t1_tracking.transformer import (
        T1TrackingTransformerConfig,
    )
    cfgs = T1TrackingTransformerConfig(sim_type="newton").build()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig

from .base import T1TrackingConfig

__all__ = ["T1TrackingTransformerConfig"]

from ...algorithms import PPOConfig


@dataclass
class T1TrackingTransformerConfig(T1TrackingConfig):
    """T1 tracking with factorized space x time transformer policy."""

    num_envs: int = 512
    # ── Future motion reference window ────────────────────────────────
    # Sparse offsets in motion frames exposed to the transformer's
    # time axis (and to ``MotionCommandCfg.future_offsets`` so the obs
    # term has data). Default 5-frame sparse preview spanning ~0.32s
    # at 50Hz control rate.
    future_offsets: tuple[int, ...] = (1, 2, 4, 8)

    # ── Transformer hyperparameters ───────────────────────────────────
    transformer_embed_dim: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 512

    # NPMP-style information bottleneck. Encoder output is pooled to a
    # ``bottleneck_dim``-wide z and broadcast back to every per-body
    # decoder head. Forces the policy to abstract motion context into
    # a small latent rather than passing per-body features through.
    transformer_bottleneck_dim: int = 32

    # Per-joint decoder MLP hidden width. The decoder keeps one head per
    # actuated joint (so morphological prior is preserved), but the
    # default of ``2 * (embed_dim + bottleneck_dim)`` makes each head
    # large — set this explicitly to keep heads compact. With
    # ``embed_dim=128`` and ``bottleneck_dim=32`` the input is 160; a
    # hidden of 128 gives ~20k params per head (vs ~52k at the default).
    transformer_decoder_hidden_dim: int = 128

    # ── Body-axis structural priors (SWAT-style) ──────────────────────
    # ``pe_type``: "learned" = single learnable (B, D) table (no
    # structural prior, matches previous body_pe behavior). "traversal"
    # = SWAT-style; concatenates pre/in/post-order DFS lookups so
    # bodies that are nearby in the kinematic tree share PE structure.
    pe_type: Literal["learned", "traversal"] = "learned"

    # When True, builds a :class:`GraphRelationalEmbedding` that adds a
    # learnable, per-head ``(H, B, B)`` bias to spatial attention scores
    # derived from graph features (Laplacian / SPD / PPR). Soft,
    # head-specific generalization of the binary kinematic mask. The
    # mask is independently controlled via ``use_kinematic_mask`` and
    # the two can be combined.
    use_relational_bias: bool = False
    re_use_laplacian: bool = True
    re_use_spd: bool = True
    re_use_ppr: bool = True
    re_ppr_alpha: float = 0.15

    # Attention mode for the encoder.
    # ``"factorized"`` (default): TimeSformer-style spatial then temporal
    # attention per layer. Lower compute per layer but ``T + B`` separate
    # small attention calls hurt GPU utilization.
    # ``"joint"``: single attention over the flattened ``(T*B,)`` token
    # sequence per layer. Higher peak FLOPs but typically faster wall
    # time for small ``T*B`` (better fusion), and strictly more
    # expressive (cross-time-and-space dependencies are direct, not
    # mediated by stacking layers).
    attention_mode: Literal["factorized", "joint"] = "factorized"

    # ── Build override ────────────────────────────────────────────────
    def _build_nn_config(self) -> NNConfig:
        if not self.future_offsets:
            raise ValueError(
                "T1TrackingTransformerConfig requires non-empty future_offsets. "
                "Set future_offsets to a tuple of motion-frame offsets, e.g. "
                "(1, 2, 4, 8, 16), or use T1TrackingConfig (MLP baseline) instead."
            )
        transformer_kwargs = {
            "tracked_body_names": self.body_names,
            "future_offsets": self.future_offsets,
            "ref_feature_dim": self.ref_feature_dim,
            "embed_dim": self.transformer_embed_dim,
            "num_heads": self.transformer_num_heads,
            "num_layers": self.transformer_num_layers,
            "dim_feedforward": self.transformer_dim_feedforward,
            "bottleneck_dim": self.transformer_bottleneck_dim,
            "decoder_hidden_dim": self.transformer_decoder_hidden_dim,
            "use_kinematic_mask": True,
            "pe_type": self.pe_type,
            "use_relational_bias": self.use_relational_bias,
            "re_use_laplacian": self.re_use_laplacian,
            "re_use_spd": self.re_use_spd,
            "re_use_ppr": self.re_use_ppr,
            "re_ppr_alpha": self.re_ppr_alpha,
            "attention_mode": self.attention_mode,
        }
        return NNConfig(
            policy=PPOPolicyConfig(
                actor_class_name="SpaceTimeTransformerActor",
                critic_class_name="SpaceTimeTransformerCritic",
                actor_kwargs=transformer_kwargs,
                critic_kwargs=transformer_kwargs,
                init_noise_std=1.0,
                distribution_type="gaussian",
                std_type="state_independent",
            ),
        )

    def _build_algorithm_config(self) -> PPOConfig:
        return PPOConfig(
            algorithm_name=self.algorithm_name,
            clip_param=0.2,
            obs_normalization=True,
            use_early_stop=False,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            actor_lr=1e-3,
            critic_lr=1e-3,
            estimator_learning_rate=5e-4,
            max_grad_norm=1.0,
            num_learning_epochs=4,
            num_mini_batches=4,
            num_steps_per_env=16,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
        )
