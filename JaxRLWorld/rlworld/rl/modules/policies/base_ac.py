from typing import Any, Union

import equinox as eqx
import jax

from rlworld.rl.modules.architectures.actor_registry import get_actor_class
from rlworld.rl.modules.architectures.base import BaseActor
from rlworld.rl.modules.normalization import EmpiricalNormalization

__all__ = ["BaseActorCritic"]


class BaseActorCritic(eqx.Module):
    """
    Base class for Actor-Critic networks.

    Equivalent to PyTorch BaseActorCritic.
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

    @staticmethod
    def _get_actor_class(actor_class_name: str):
        """Get actor class from name."""
        return get_actor_class(actor_class_name)

    @staticmethod
    def _build_actor_common(
        actor_class_name: str,
        actor_obs_dim: int,
        num_actions: int,
        actor_kwargs: dict[str, Any],
        kinematic_tree: "KinematicTree | None" = None,
        *,
        key: jax.Array,
    ) -> BaseActor:
        """Common logic for building actors."""
        ActorClass = get_actor_class(actor_class_name)
        actor_kwargs = actor_kwargs.copy()

        if actor_class_name == "MLPActor":
            actor_kwargs.update({
                "num_obs": actor_obs_dim,
                "num_actions": num_actions,
                "ortho_init": actor_kwargs["ortho_init"],
                "key": key,
            })
        else:
            if kinematic_tree is None:
                raise ValueError(
                    f"{actor_class_name} requires kinematic_tree."
                )
            actor_kwargs.update({
                "kinematic_tree": kinematic_tree,
                "num_obs": actor_obs_dim,
                "num_actions": num_actions,
                "key": key,
            })

        return ActorClass(**actor_kwargs)

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