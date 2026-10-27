from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax

__all__ = ["BaseActor", "BaseCritic"]


class BaseActor(eqx.Module):
    """Base class for all actors.

    An actor may carry an auxiliary objective of its own (a regulariser
    on its internal representation, say). It announces that with
    ``has_auxiliary_loss = True`` and implements
    :meth:`compute_auxiliary_loss`; the PPO update then takes one extra
    optimizer step on that loss after the policy update, over the last
    minibatch, and calls :meth:`post_update_step` once per update so the
    actor can advance any per-update schedule. Equinox modules are
    immutable, so both hooks are pure: the schedule step returns the
    actor to use from now on.
    """

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    has_auxiliary_loss: ClassVar[bool] = False

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

    def compute_auxiliary_loss(self, obs: jax.Array) -> jax.Array:
        """Scalar auxiliary loss for one (num_obs,) observation.

        Only called when ``has_auxiliary_loss`` is True.
        """
        raise NotImplementedError(f"{type(self).__name__} declares no auxiliary loss")

    def post_update_step(self) -> "BaseActor":
        """The actor after one PPO update: a hook for per-update schedules.

        The default has nothing to advance and returns ``self``.
        """
        return self


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
