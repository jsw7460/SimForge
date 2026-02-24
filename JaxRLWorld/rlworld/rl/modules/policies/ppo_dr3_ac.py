"""
PPO-DR3 Actor-Critic.

Extends PPOActorCritic to use DR3Critic for feature extraction.
"""
from typing import TYPE_CHECKING

import jax

from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.rl.modules.architectures.dr3 import DR3Critic

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["PPODR3ActorCritic"]


class PPODR3ActorCritic(PPOActorCritic):
    """
    PPO Actor-Critic with DR3-compatible critic.

    Uses DR3Critic instead of MLPCritic to enable feature extraction
    for DR3 regularization.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_class_name: str = "MLPActor",
        init_noise_std: float = 1.0,
        std_type: str = "state_dependent",
        distribution_type: str = "gaussian",
        kinematic_tree: "KinematicTree | None" = None,
        *,
        key: jax.Array,
        **kwargs,
    ):
        """
        Args:
            num_actor_obs: Actor observation dimension
            num_critic_obs: Critic observation dimension
            num_actions: Action dimension
            actor_class_name: Name of actor class ("MLPActor", "BodyTransformerActor", etc.)
            init_noise_std: Initial action standard deviation
            std_type: "state_dependent", "state_independent", or "fixed"
            distribution_type: "gaussian" or "squashed_gaussian"
            kinematic_tree: Optional kinematic tree for dynamics-aware actors
            key: JAX random key
            **kwargs: Must contain "actor_kwargs" and "critic_kwargs"
        """

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_class_name=actor_class_name,
            init_noise_std=init_noise_std,
            std_type=std_type,
            distribution_type=distribution_type,
            kinematic_tree=kinematic_tree,
            key=key,
            **kwargs
        )

    def _build_networks(
        self,
        actor_class_name: str,
        kinematic_tree: "KinematicTree | None",
        key_actor: jax.Array,
        key_critic: jax.Array,
        **kwargs,
    ):
        """Build actor and DR3-compatible critic networks."""
        # Build DR3Critic instead of MLPCritic
        critic_kwargs = kwargs["critic_kwargs"]
        self.critic = DR3Critic(
            num_obs=self.critic_obs_dim,
            hidden_dims=critic_kwargs["hidden_dims"],
            activation=critic_kwargs["activation"],
            use_layer_norm=critic_kwargs.get("use_layer_norm", False),
            ortho_init=True,
            key=key_critic,
        )

        # Build actor using common logic from base class
        actor_kwargs = kwargs.get("actor_kwargs", {})
        self.actor = self._build_actor_common(
            actor_class_name=actor_class_name,
            actor_obs_dim=self.actor_obs_dim,
            num_actions=self.num_actions,
            actor_kwargs=actor_kwargs,
            kinematic_tree=kinematic_tree,
            key=key_actor,
        )

        print(f"🔬 DR3Critic initialized (feature_dim={self.critic.feature_dim})")

    @property
    def critic_feature_dim(self) -> int:
        """Get critic feature dimension."""
        return self.critic.feature_dim