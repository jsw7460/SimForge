from typing import TYPE_CHECKING

import equinox as eqx

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

    # Set when observations arrive as a dict of groups rather than one
    # array — the name of the group holding the state vector. Everything
    # else in the dict is an image, which normalization skips: running
    # statistics over pixels track whatever the camera happens to be
    # pointing at, and rsl_rl's CNNModel normalizes its 1D groups only.
    actor_vector_group: str | None = eqx.field(static=True, default=None)
    critic_vector_group: str | None = eqx.field(static=True, default=None)

    def _actor_vector(self, observations):
        """The state-vector part of an actor observation."""
        return observations if self.actor_vector_group is None else observations[self.actor_vector_group]

    def _critic_vector(self, observations):
        """The state-vector part of a critic observation."""
        return observations if self.critic_vector_group is None else observations[self.critic_vector_group]

    def _normalize_actor_obs(self, observations):
        """Normalize actor observations using running statistics if available."""
        if self.actor_vector_group is None:
            if self.actor_obs_normalizer is None:
                return observations
            return self.actor_obs_normalizer.normalize(observations)
        if self.actor_obs_normalizer is None:
            return observations
        vector = self.actor_obs_normalizer.normalize(observations[self.actor_vector_group])
        return {**observations, self.actor_vector_group: vector}

    def _normalize_critic_obs(self, observations):
        """Normalize critic observations using running statistics if available."""
        if self.critic_vector_group is None:
            if self.critic_obs_normalizer is None:
                return observations
            return self.critic_obs_normalizer.normalize(observations)
        if self.critic_obs_normalizer is None:
            return observations
        vector = self.critic_obs_normalizer.normalize(observations[self.critic_vector_group])
        return {**observations, self.critic_vector_group: vector}
