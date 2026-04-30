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

from rlworld.rl.configs.common_config_classes import NNConfig, PPOPolicyConfig

from .base import T1TrackingConfig


__all__ = ["T1TrackingTransformerConfig"]


@dataclass
class T1TrackingTransformerConfig(T1TrackingConfig):
    """T1 tracking with factorized space x time transformer policy."""

    # ── Future motion reference window ────────────────────────────────
    # Sparse offsets in motion frames exposed to the transformer's
    # time axis (and to ``MotionCommandCfg.future_offsets`` so the obs
    # term has data). Default 5-frame sparse preview spanning ~0.32s
    # at 50Hz control rate.
    future_offsets: tuple[int, ...] = (1, 2, 4, 8, 16)

    # ── Transformer hyperparameters ───────────────────────────────────
    transformer_embed_dim: int = 64
    transformer_num_heads: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 128

    # NPMP-style information bottleneck. Encoder output is pooled to a
    # ``bottleneck_dim``-wide z and broadcast back to every per-body
    # decoder head. Forces the policy to abstract motion context into
    # a small latent rather than passing per-body features through.
    transformer_bottleneck_dim: int = 32

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
            "use_kinematic_mask": True,
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
