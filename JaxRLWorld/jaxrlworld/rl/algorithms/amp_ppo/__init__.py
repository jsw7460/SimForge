"""PPO with an adversarial motion prior (AMP).

``expert_motion`` renders reference clips into the discriminator's feature
layout (read from the env's ``amp`` observation group) and holds the clip
augmentations; ``discriminator`` is the network and its losses; ``amp`` the
prior's state (expert windows, replay, policy history); ``amp_ppo`` the
algorithm that plugs it into PPO.
"""

from jaxrlworld.rl.algorithms.amp_ppo.amp import AdversarialMotionPrior, expert_frame_probabilities
from jaxrlworld.rl.algorithms.amp_ppo.amp_ppo import AmpPPO
from jaxrlworld.rl.algorithms.amp_ppo.discriminator import (
    Discriminator,
    DiscriminatorStats,
    discriminator_logits,
    discriminator_loss,
    discriminator_reward,
    minibatch_std,
)
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import (
    AmpFeatureTerm,
    ClipState,
    ExpertMotionSet,
    amp_feature_layout,
    clip_features,
    clip_variants,
    feature_dim,
    history_windows,
    layout_mirror_operator,
    load_clip,
    load_expert_motions,
    mirror_clip,
    mirror_features,
    resample_clip_speed,
)
from jaxrlworld.rl.algorithms.amp_ppo.metrics import AmpMetrics, AmpPPOMetrics

__all__ = [
    "AdversarialMotionPrior",
    "AmpFeatureTerm",
    "AmpMetrics",
    "AmpPPO",
    "AmpPPOMetrics",
    "ClipState",
    "Discriminator",
    "DiscriminatorStats",
    "ExpertMotionSet",
    "amp_feature_layout",
    "clip_features",
    "clip_variants",
    "discriminator_logits",
    "discriminator_loss",
    "discriminator_reward",
    "expert_frame_probabilities",
    "feature_dim",
    "history_windows",
    "layout_mirror_operator",
    "load_clip",
    "load_expert_motions",
    "minibatch_std",
    "mirror_clip",
    "mirror_features",
    "resample_clip_speed",
]
