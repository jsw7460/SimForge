"""dm_control (via shimmy + gymnasium) env factory.

``make_dmc_env_factory`` returns a ``(seed) -> gym.Env`` callable that
applies the canonical TDMPC2 / DMC-benchmark wrapper chain (action
repeat → flatten dict observation → seed) on top of
``gym.make("dm_control/<domain>-<task>-v0")``.

The factory is consumed by both:

* The user's training-env construction (``SyncVectorEnv([factory(i) for ...])``).
* :attr:`rlworld.rl.runners.BaseRunner.gym_env_factory` so the runner
  builds the eval env through the *same* chain — no wrapper drift
  between train and eval.

Observation normalization deliberately lives in the algorithm (running
mean/std inside the TDMPC2 world model / SAC actor+critic), not here.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

# Importing shimmy registers the ``dm_control/*`` namespace in
# ``gymnasium``'s env registry; ``gym.make("dm_control/cheetah-run-v0")``
# raises ``NamespaceNotFound`` until this import has run.
import shimmy  # noqa: F401
from gymnasium.wrappers import FlattenObservation

from rlworld.rl.envs.gymnasium.wrappers import ActionRepeatWrapper


def make_dmc_env_factory(
    task_name: str,
    action_repeat: int = 2,
    max_episode_steps: int = 1000,
) -> Callable[[int], gym.Env]:
    """Return a ``(seed) -> gym.Env`` factory for one DMC task.

    Args:
        task_name: gymnasium env id such as
            ``"dm_control/cheetah-run-v0"``.
        action_repeat: frame-skip applied via
            :class:`ActionRepeatWrapper`.  DMC convention is 2.
        max_episode_steps: forwarded to ``gym.make``; the dm_control
            suite's native episode length is 1000 substeps and that
            value is the standard benchmark setting.

    The returned factory is pure (deterministic in ``seed``) and safe
    to call repeatedly — one call per worker for ``SyncVectorEnv``
    construction, one call when the runner spins up the eval env.
    """

    def _factory(seed: int) -> gym.Env:
        env = gym.make(task_name, max_episode_steps=max_episode_steps)
        env = ActionRepeatWrapper(env, repeat=action_repeat)
        env = FlattenObservation(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _factory
