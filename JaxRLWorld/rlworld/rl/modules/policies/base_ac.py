from typing import TYPE_CHECKING

import equinox as eqx
import jax

from rlworld.rl.modules.normalization import EmpiricalNormalization

if TYPE_CHECKING:
    pass

__all__ = ["BaseActorCritic"]


class BaseActorCritic(eqx.Module):
    """Base class for Actor-Critic networks.

    Subclasses (``PPOActorCritic`` / ``SACActorCritic`` / ``TD3ActorCritic``)
    receive typed ``actor_cfg`` / ``critic_cfg`` and dispatch to the
    cfg-type-keyed builders in ``rlworld.rl.modules.architectures.actor_registry``
    (``build_actor`` / ``build_critic``).
    """

    actor: eqx.Module
    critic: eqx.Module

    actor_obs_dim: int = eqx.field(static=True)
    critic_obs_dim: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    is_recurrent: bool = eqx.field(static=True, default=False)
    is_squashed: bool = eqx.field(static=True, default=False)

    # Observation normalizers (optional, used by subclasses that enable obs_normalization)
    actor_obs_normalizer: EmpiricalNormalization | None = None
    critic_obs_normalizer: EmpiricalNormalization | None = None

    def _normalize_actor_obs(self, observations: jax.Array) -> jax.Array:
        """Normalize actor observations using running statistics if available."""
        if self.actor_obs_normalizer is None:
            return observations
        return self.actor_obs_normalizer.normalize(observations)

    def _normalize_critic_obs(self, observations: jax.Array) -> jax.Array:
        """Normalize critic observations using running statistics if available."""
        if self.critic_obs_normalizer is None:
            return observations
        return self.critic_obs_normalizer.normalize(observations)
