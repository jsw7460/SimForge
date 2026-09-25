"""PPO with an adversarial motion prior (Peng et al. 2021), after booster_mjlab's ``AmpPPO``.

PPO is untouched: this subclass blends the discriminator's style reward into
the step reward through :meth:`shape_step_reward` (the hook the runner calls
on every step, identity in PPO) and takes the discriminator's gradient steps
inside :meth:`update`, after the PPO update. The rollout, the storage and
the PPO losses are the parent's.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import torch

from jaxrlworld.rl.algorithms.amp_ppo.amp import AdversarialMotionPrior
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import amp_feature_layout, feature_dim, load_expert_motions
from jaxrlworld.rl.algorithms.amp_ppo.metrics import AmpPPOMetrics
from jaxrlworld.rl.algorithms.ppo.ppo import PPO
from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpPPOConfig
from jaxrlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax_many


class AmpPPO(PPO):
    """PPO whose step reward carries a discriminator's style term."""

    def __init__(self, *, amp: AdversarialMotionPrior, **ppo_kwargs):
        super().__init__(**ppo_kwargs)
        self.amp = amp
        if self.recompute_gae_per_epoch:
            raise ValueError("AMP_PPO uses the minibatch update; recompute_gae_per_epoch is not supported")

    @classmethod
    def from_config(cls, cfg: AmpPPOConfig, actor_critic, env, key: jax.Array, **symmetry) -> AmpPPO:
        """Build the prior from the env (feature layout, joint order, control rate) and the config."""
        if not cfg.amp.motion_files:
            raise ValueError("AMP_PPO needs amp.motion_files")
        if not cfg.amp.root_body_name:
            raise ValueError("AMP_PPO needs amp.root_body_name (the base link as the clips name it)")
        key, k_amp = jax.random.split(key)
        layout = amp_feature_layout(env.obs_manager, cfg.amp.amp_group)
        joint_names = list(env.act_manager.actuated_joint_names)
        robot = env.get_entity_data(env.robot_entity_name)
        expert = load_expert_motions(
            cfg.amp.motion_files,
            layout,
            joint_names,
            robot.default_joint_pos,
            cfg.amp.root_body_name,
            float(env.control_dt),
            mirror=cfg.amp.mirror_augmentation,
            speed_factors=tuple(cfg.amp.speed_augmentations),
        )
        amp = AdversarialMotionPrior(
            cfg.amp,
            expert,
            feature_dim(layout),
            float(env.control_dt),
            cfg.amp.discriminator_lr if cfg.amp.discriminator_lr is not None else cfg.actor_lr,
            k_amp,
        )
        return cls(amp=amp, actor_critic=actor_critic, key=key, **cls._ppo_kwargs(cfg), **symmetry)

    # ── rollout ──────────────────────────────────────────────────────

    def shape_step_reward(self, rewards: torch.Tensor, obs_dict: dict, dones: torch.Tensor) -> torch.Tensor:
        """``(1 - w) r_task + w dt r_style`` for this step; see :class:`AdversarialMotionPrior`."""
        sources = {
            "amp": obs_dict[self.amp.cfg.amp_group],
            "dones": dones.to(torch.uint8),
            "rewards": rewards,
        }
        converted = torch_to_jax_many(sources)
        blended = self.amp.shape_rewards(converted["amp"], converted["dones"] != 0, converted["rewards"])
        return jax_to_torch(blended, rewards.device)

    # ── update ───────────────────────────────────────────────────────

    def update(self) -> AmpPPOMetrics:
        minibatch_size = (self.storage.num_envs * self.storage.num_steps) // self.num_mini_batches
        num_updates = self.num_learning_epochs * self.num_mini_batches
        metrics = super().update()
        amp_metrics = self.amp.update_discriminator(num_updates, minibatch_size)
        return AmpPPOMetrics(
            critic=metrics.critic,
            actor=metrics.actor,
            kl=metrics.kl,
            batch=metrics.batch,
            learning_rate=metrics.learning_rate,
            amp=amp_metrics,
        )

    def _set_actor_lr(self, lr: float) -> None:
        super()._set_actor_lr(lr)
        if self.amp.cfg.discriminator_lr is None:
            self.amp.set_learning_rate(lr)

    # ── checkpoint ───────────────────────────────────────────────────

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        metadata = super().save_train_state(checkpoint_dir)
        self.amp.save(checkpoint_dir)
        metadata["amp_discriminator_lr"] = self.amp.learning_rate
        return metadata

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        super().load_train_state(checkpoint_dir, metadata)
        self.amp.learning_rate = float(metadata.get("amp_discriminator_lr", self.amp.learning_rate))
        self.amp.load(checkpoint_dir)
