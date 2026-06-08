"""Gymnasium-side env helpers — wrapper chains + factories.

Centralises the wrapper composition for Gymnasium-backed tasks (dm_control,
classic control, etc.) so the same chain is used for both training and
evaluation envs.  Bypasses the ``cfgs.env``-only path that the physics-sim
envs use, because Gymnasium wrappers are *functional* (Python callables)
and can't be expressed declaratively in a dataclass.

Each factory in this package returns a ``(seed) -> gym.Env`` callable.
Pass the result to either:

* ``SyncVectorEnv([factory(i) for i in range(num_envs)], ...)`` when the
  user script owns the training env (current TDMPC2 benchmark pattern).
* ``runner.gym_env_factory = factory`` to let ``BaseRunner`` build the
  eval env through the same chain — keeps train and eval consistent.
"""

from rlworld.rl.envs.gymnasium.dmc_factory import make_dmc_env_factory
from rlworld.rl.envs.gymnasium.wrappers import ActionRepeatWrapper

__all__ = ["ActionRepeatWrapper", "make_dmc_env_factory"]
