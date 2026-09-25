from dataclasses import dataclass, field

from ..base_config import BaseConfig
from .ppo import PPOConfig


@dataclass
class AmpConfig(BaseConfig):
    """Adversarial motion prior (Peng et al. 2021) on top of PPO.

    A discriminator scores K-frame windows of the env's ``amp`` observation
    group against the same windows cut from reference clips, and its output
    is blended into the step reward::

        r = (1 - w) * r_task + w * dt * r_style,   r_style = D's reward of the
                                                    window ending at this step

    (``dt`` because the task rewards are per-step rates scaled by the control
    step). The discriminator takes one gradient step per PPO minibatch, on a
    policy batch made of the current rollout and a replay of older rollouts
    against an equally sized expert batch.
    """

    motion_files: tuple[str, ...] = ()
    """Reference clips (NPZ, tracking format) at the control rate."""
    root_body_name: str = ""
    """The robot's base link as the clips' ``body_names`` list it."""
    dataset_weights: tuple[float, ...] | None = None
    """Per-file sampling mass; ``None`` for equal. Every clip (and each of its
    augmented variants) spreads its mass uniformly over its frames, so frames
    of a short clip are drawn more often than frames of a long one."""
    mirror_augmentation: bool = True
    """Add the left/right reflection of every clip to the expert set."""
    speed_augmentations: tuple[float, ...] = (1.1, 0.9, 1.2, 0.8)
    """Replay every clip (and its mirror) at these speed factors as well."""
    amp_group: str = "amp"
    """Observation group the discriminator scores, single-frame terms only."""
    num_amp_obs_steps: int = 10
    """Frames per discriminator sample."""

    style_reward_weight: float = 0.3
    reward_scale: float = 1.0
    """Multiplies the discriminator's reward before the blend."""

    discriminator_hidden_dims: tuple[int, ...] = (256, 128)
    discriminator_activation: str = "relu"
    use_minibatch_std: bool = True
    """Append the batch's mean feature standard deviation to the last hidden
    layer (a mode-collapse guard; detached inside the gradient penalty)."""
    empirical_normalization: bool = True
    """Normalize windows by running statistics of expert and policy batches."""
    loss_type: str = "bce"
    """``bce`` / ``hinge`` (R1 penalty on expert samples) or ``wasserstein``
    (tanh-squashed critic, gradient penalty on interpolates)."""
    wasserstein_eta: float = 1.0
    grad_penalty_lambda: float = 10.0

    discriminator_lr: float | None = None
    """``None`` follows the actor's (adaptive) learning rate, as the source
    does with one optimizer over actor, critic and discriminator."""
    discriminator_weight_decay_trunk: float = 1e-3
    discriminator_weight_decay_head: float = 1e-1
    """Coupled L2 (Adam ``weight_decay``) on the hidden layers / the head."""

    replay_buffer_size: int = 200_000
    replay_insert_size: int = 1_000
    """Rollout windows kept once the replay is full; before that every window
    of a rollout is inserted."""


@dataclass
class AmpPPOConfig(PPOConfig):
    """PPO plus an adversarial motion prior; every PPO field applies unchanged."""

    algorithm_name: str = "AMP_PPO"
    amp: AmpConfig = field(default_factory=AmpConfig)
