from abc import abstractmethod

import equinox as eqx
import jax

__all__ = ["BaseActor", "BaseCritic"]


class BaseActor(eqx.Module):
    """Base class for all actors."""

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    @abstractmethod
    def __call__(self, obs: jax.Array, *, key: jax.Array) -> tuple[jax.Array, dict]:
        """
        Args:
            obs: (num_obs,) unbatched observation
            key: JAX random key

        Returns:
            actions: (num_actions,)
            aux: auxiliary info dict
        """
        raise NotImplementedError


class BaseCritic(eqx.Module):
    """Base class for all critics."""

    num_obs: int = eqx.field(static=True)

    @abstractmethod
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, dict]:
        """
        Args:
            obs: (num_obs,) unbatched observation

        Returns:
            value: scalar
            aux: auxiliary info dict
        """
        raise NotImplementedError
